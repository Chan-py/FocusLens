"""Real-time HUD overlay rendered onto the camera frame."""

from __future__ import annotations

import cv2
import numpy as np

from models import DistractionResult, Scenario


def draw_overlay(
    frame:         np.ndarray,
    result:        DistractionResult,
    total_blinks:  int,
    ear_threshold: float,
    drut_history:  list[float],
    scenario:      Scenario | None = None,
    calibrated:    bool = False,
) -> np.ndarray:
    h, w  = frame.shape[:2]
    color = (0, 255, 0) if not result.distracted else (0, 0, 255)
    f     = result.features

    lines: list[str] = []
    if scenario:
        lines.append(f"Scenario: {scenario.label}")
    if calibrated:
        lines.append("Calibrated: YES")
    lines += [
        f"Focused:  {not result.distracted}",
        f"Reason:   {'+'.join(r.value for r in result.reasons) if result.reasons else 'none'}",
        f"Blinks:   {total_blinks}",
    ]
    if f.yaw is not None:
        lines.append(f"Yaw: {f.yaw:.1f}  Pitch: {f.pitch:.1f}")
    if f.ear is not None:
        lines.append(f"EAR: {f.ear:.3f}  thr: {ear_threshold:.3f}")
    if f.gaze is not None:
        lines.append(f"Gaze: {f.gaze.value}")

    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _draw_drut_graph(frame, drut_history, w, h)

    cv2.putText(frame, "Press Q to end session & generate report",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def _draw_drut_graph(
    frame: np.ndarray, drut_history: list[float], w: int, h: int
) -> None:
    if len(drut_history) < 2:
        return
    gx, gy, gw, gh = w - 220, 20, 200, 60
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (50, 50, 50), -1)
    pts = drut_history[-20:]
    n   = len(pts)
    for i in range(1, n):
        x1 = gx + int((i - 1) / max(n - 1, 1) * gw)
        x2 = gx + int(i       / max(n - 1, 1) * gw)
        y1 = gy + gh - int(pts[i - 1] * gh)
        y2 = gy + gh - int(pts[i]     * gh)
        c  = (0, 255, 0) if pts[i] < 0.2 else (0, 0, 255)
        cv2.line(frame, (x1, y1), (x2, y2), c, 2)
    cv2.putText(frame, "DRUT", (gx, gy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
