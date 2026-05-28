"""EAR (Eye Aspect Ratio) computation — Wang et al. (2025).

EAR_right = (||p161-p163|| + ||p160-p144|| + ||p158-p153|| + ||p157-p154||)
            / (4 × ||p33-p133||)
EAR_left  = (||p384-p381|| + ||p385-p380|| + ||p387-p373|| + ||p388-p390||)
            / (3 × ||p362-p263||)   ← denominator 3 per Wang et al.
EAR = (EAR_right + EAR_left) / 2
"""

import numpy as np


def _coord(landmarks, idx: int, w: int, h: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * w, lm.y * h


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def compute_ear(landmarks, w: int, h: int) -> float:
    lm = lambda i: _coord(landmarks, i, w, h)

    r_num = (_dist(lm(161), lm(163)) + _dist(lm(160), lm(144)) +
             _dist(lm(158), lm(153)) + _dist(lm(157), lm(154)))
    r_den = 4 * _dist(lm(33), lm(133))
    ear_right = r_num / r_den if r_den > 0 else 0.0

    lf_num = (_dist(lm(384), lm(381)) + _dist(lm(385), lm(380)) +
              _dist(lm(387), lm(373)) + _dist(lm(388), lm(390)))
    lf_den = 3 * _dist(lm(362), lm(263))
    ear_left = lf_num / lf_den if lf_den > 0 else 0.0

    return (ear_right + ear_left) / 2
