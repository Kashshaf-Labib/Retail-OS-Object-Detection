"""Drift monitoring thresholds and settings."""

# ── Population Stability Index (PSI) ──
# PSI measures how much the prediction class distribution has shifted
# from the training baseline.
#   PSI < 0.1  → No significant change
#   PSI 0.1-0.2 → Moderate shift (WARNING)
#   PSI > 0.2  → Significant shift (DRIFT_DETECTED)
PSI_THRESHOLD_WARNING = 0.1
PSI_THRESHOLD_DRIFT = 0.2

# ── Confidence Score Decay ──
# If the average confidence of predictions drops by more than this
# percentage compared to baseline, it indicates the model is seeing
# unfamiliar inputs.
CONFIDENCE_BASELINE = 0.65  # Expected avg confidence from training
CONFIDENCE_DECAY_THRESHOLD = 0.15  # 15% drop triggers warning

# ── Monitoring Window ──
# Number of recent predictions to analyze for drift detection.
MONITORING_WINDOW = 500

# ── Retraining Cooldown ──
# Minimum hours between auto-retrain triggers to prevent thrashing.
RETRAIN_COOLDOWN_HOURS = 24

# ── Drift Status Labels ──
STATUS_HEALTHY = "HEALTHY"
STATUS_WARNING = "WARNING"
STATUS_DRIFT = "DRIFT_DETECTED"
