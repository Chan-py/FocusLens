"""Head pose estimation via PnP algorithm — Wang et al. (2025).

Maps 6 face landmarks from 2-D camera coordinates to a 3-D reference model,
then extracts yaw, pitch, roll in degrees from the resulting rotation matrix.
"""

import cv2
import numpy as np
from config import FACE_3D_POINTS, FACE_2D_LANDMARK_IDS


def compute_head_pose(
    landmarks, w: int, h: int
) -> tuple[float | None, float | None, float | None]:
    """Return (pitch, yaw, roll) in degrees, or (None, None, None) on failure."""
    face_2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in FACE_2D_LANDMARK_IDS],
        dtype=np.float64,
    )
    cam_matrix = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64
    )
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, face_2d, cam_matrix, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy >= 1e-6:
        pitch = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0

    return float(pitch), float(yaw), float(roll)
