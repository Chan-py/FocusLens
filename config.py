"""Algorithm constants — single source of truth for all tunable parameters."""

import numpy as np

# ── EAR dynamic threshold ─────────────────────────────────────────────────────
EAR_ALPHA:              float = 0.02   # subtracted from sliding-window mean
EAR_WINDOW_SIZE:        int   = 60     # frames in sliding window (hyperparameter)
EAR_WINDOW_MIN_SAMPLES: int   = 5      # window needs this many samples before dynamic threshold kicks in
CLOSED_EYE_FRAMES:      int   = 3      # consecutive sub-threshold frames → blink

# ── Head pose tolerance defaults (calibration overrides per-person) ───────────
DEFAULT_PITCH_TOLERANCE: float = 20.0   # degrees ± around baseline
DEFAULT_YAW_TOLERANCE:   float = 20.0   # for SCREEN scenario
PAPER_YAW_TOLERANCE:     float = 15.0   # tighter for PAPER/BOOK (head is usually stationary)

# ── DRUT (Distraction Rate per Unit Time) ────────────────────────────────────
DRUT_UNIT_FRAMES:     int   = 800   # Wang et al.; lower during development
DRUT_FOCUS_THRESHOLD: float = 0.2   # DRUT < this → focused

# ── Fatigue detection (FocusLens extension) ──────────────────────────────────
FATIGUE_BLINK_MULTIPLIER: float = 1.5   # blink rate ≥ baseline × this → fatigue

# ── Face geometry: 3-D reference model (Wang et al.) ─────────────────────────
FACE_3D_POINTS: np.ndarray = np.array(
    [
        [0.0,    0.0,    0.0  ],   # nose tip      (landmark 1)
        [0.0,  -330.0,  -65.0 ],   # chin          (152)
        [-225.0, 170.0, -135.0],   # left eye corner  (263)
        [225.0,  170.0, -135.0],   # right eye corner (33)
        [-150.0,-150.0, -125.0],   # left mouth corner  (287)
        [150.0, -150.0, -125.0],   # right mouth corner (57)
    ],
    dtype=np.float64,
)
FACE_2D_LANDMARK_IDS: list[int] = [1, 152, 263, 33, 287, 57]

# ── Eye landmark ids (shared by EAR and gaze) ────────────────────────────────
RIGHT_EYE_IDS: list[int] = [33, 133, 160, 159, 158, 144, 145, 153]
LEFT_EYE_IDS:  list[int] = [362, 263, 387, 386, 385, 373, 374, 380]

# ── Gaze detection ────────────────────────────────────────────────────────────
GAZE_BINARIZE_THRESHOLD: int = 50   # pixel intensity cutoff for iris binarization
