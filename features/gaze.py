"""Gaze direction detection — Wang et al. (2025).

Pipeline per eye:
  grayscale crop → GaussianBlur → medianBlur → binary threshold (THRESH_BINARY_INV)
  → split into thirds → dark-pixel count → left / center / right

Fusion: if both eyes agree → that direction; else → "off-center" (distracted).
"""

import cv2
import numpy as np
from config import GAZE_BINARIZE_THRESHOLD, RIGHT_EYE_IDS, LEFT_EYE_IDS
from models import GazePosition


def _eye_region(
    frame_gray: np.ndarray,
    landmarks,
    eye_ids: list[int],
    w: int,
    h: int,
    padding: int = 5,
) -> np.ndarray:
    coords = np.array(
        [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_ids]
    )
    x0 = max(coords[:, 0].min() - padding, 0)
    x1 = min(coords[:, 0].max() + padding, w)
    y0 = max(coords[:, 1].min() - padding, 0)
    y1 = min(coords[:, 1].max() + padding, h)
    return frame_gray[y0:y1, x0:x1]


def _iris_position(eye_crop: np.ndarray) -> GazePosition:
    if eye_crop is None or eye_crop.size == 0:
        return GazePosition.UNKNOWN
    blurred = cv2.medianBlur(cv2.GaussianBlur(eye_crop, (5, 5), 0), 3)
    _, binary = cv2.threshold(blurred, GAZE_BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    _, crop_w = binary.shape
    if crop_w < 3:
        return GazePosition.UNKNOWN
    t = crop_w // 3
    counts = [
        int(np.sum(binary[:, :t])),
        int(np.sum(binary[:, t : 2 * t])),
        int(np.sum(binary[:, 2 * t :])),
    ]
    return [GazePosition.LEFT, GazePosition.CENTER, GazePosition.RIGHT][int(np.argmax(counts))]


def compute_gaze(frame_gray: np.ndarray, landmarks, w: int, h: int) -> GazePosition:
    right = _iris_position(_eye_region(frame_gray, landmarks, RIGHT_EYE_IDS, w, h))
    left  = _iris_position(_eye_region(frame_gray, landmarks, LEFT_EYE_IDS,  w, h))
    if right == GazePosition.UNKNOWN or left == GazePosition.UNKNOWN:
        return GazePosition.UNKNOWN
    return right if right == left else GazePosition.OFF_CENTER
