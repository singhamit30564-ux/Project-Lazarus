"""Central paths and constants for Project Lazarus."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Artefacts produced by the training scripts and consumed by the app / agents.
WEIGHTS_AUTHENTICITY = MODELS_DIR / "read_authenticity_mlp.pt"
WEIGHTS_DAMAGE_CNN = MODELS_DIR / "damage_profile_cnn.pt"
WEIGHTS_DQN = MODELS_DIR / "dqn_gapfill.pt"
METRICS_ML = MODELS_DIR / "ml_metrics.json"
HISTORY_RL = MODELS_DIR / "rl_history.json"
EVAL_RL = MODELS_DIR / "rl_eval.json"

DEFAULT_SEED = 42

BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for i, b in enumerate(BASES)}
UNKNOWN_BASE = "N"

# Read-authenticity feature vector layout (see lazarus/ml/features.py).
READ_FEATURE_DIM = 24

# Damage-profile model input: (C->T curve, G->A curve) x positions-from-end.
PROFILE_POSITIONS = 20

# UI palette (mirrored in lazarus/ui_common.py).
ACCENT = "#35d0a5"
ACCENT_2 = "#a3e635"
BG = "#0b1210"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
