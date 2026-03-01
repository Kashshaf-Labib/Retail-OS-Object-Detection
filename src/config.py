"""Central configuration for the retail shelf detection application."""

import os
from pathlib import Path


# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best.pt")))
EXPERIMENTS_PATH = PROJECT_ROOT / "experiments" / "experiment_log.json"
DRIFT_LOG_PATH = PROJECT_ROOT / "experiments" / "drift_log.json"

# ── Model Settings ──
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1280"))
DEVICE = os.getenv("DEVICE", "0")  # "0" for GPU, "cpu" for CPU

# ── Server Settings ──
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Drift Monitoring ──
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))
CONFIDENCE_DECAY_THRESHOLD = float(os.getenv("CONFIDENCE_DECAY_THRESHOLD", "0.15"))
MONITORING_WINDOW = int(os.getenv("MONITORING_WINDOW", "500"))
RETRAIN_COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "24"))

# ── Class Names (76 SKUs) ──
CLASS_NAMES = [
    'q1', 'q10', 'q100', 'q103', 'q106', 'q109', 'q112', 'q115', 'q118', 'q121',
    'q13', 'q130', 'q133', 'q136', 'q142', 'q145', 'q148', 'q151', 'q157', 'q16',
    'q163', 'q169', 'q175', 'q178', 'q184', 'q187', 'q19', 'q190', 'q193', 'q196',
    'q199', 'q202', 'q211', 'q214', 'q22', 'q220', 'q229', 'q232', 'q247', 'q25',
    'q250', 'q256', 'q262', 'q265', 'q268', 'q271', 'q274', 'q280', 'q286', 'q289',
    'q291', 'q293', 'q299', 'q31', 'q34', 'q37', 'q4', 'q40', 'q46', 'q49',
    'q52', 'q55', 'q58', 'q61', 'q64', 'q67', 'q7', 'q70', 'q73', 'q76',
    'q79', 'q82', 'q88', 'q91', 'q94', 'q97',
]
NUM_CLASSES = len(CLASS_NAMES)
