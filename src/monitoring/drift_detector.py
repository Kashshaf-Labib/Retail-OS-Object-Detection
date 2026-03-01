"""Data drift detection engine using PSI and confidence monitoring."""

import json
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import CLASS_NAMES, NUM_CLASSES, DRIFT_LOG_PATH
from src.monitoring.drift_config import (
    PSI_THRESHOLD_WARNING,
    PSI_THRESHOLD_DRIFT,
    CONFIDENCE_BASELINE,
    CONFIDENCE_DECAY_THRESHOLD,
    MONITORING_WINDOW,
    STATUS_HEALTHY,
    STATUS_WARNING,
    STATUS_DRIFT,
)


class DriftMonitor:
    """
    Monitors prediction distribution drift using PSI and confidence decay.

    Every detection call logs predictions into a rolling buffer. The monitor
    compares the recent prediction distribution against the training baseline
    to detect concept drift.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._predictions: deque = deque(maxlen=MONITORING_WINDOW)
        self._confidences: deque = deque(maxlen=MONITORING_WINDOW)
        self._last_retrain_time: float = 0
        self._total_logged: int = 0

        # Training baseline distribution (from class imbalance analysis)
        # Normalized to probabilities
        self._training_distribution = self._compute_training_baseline()

    def _compute_training_baseline(self) -> np.ndarray:
        """
        Compute the expected class distribution from training data.
        Counts from the original training set analysis.
        """
        # Training counts per class (from the class imbalance analysis)
        train_counts = np.array([
            21, 6, 21, 12, 21, 93, 48, 12, 99, 384,
            6, 18, 21, 57, 45, 9, 18, 48, 12, 99,
            48, 225, 33, 12, 30, 30, 24, 60, 9, 39,
            15, 15, 357, 15, 39, 87, 45, 18, 18, 99,
            81, 69, 162, 12, 9, 21, 42, 21, 60, 114,
            21, 60, 393, 60, 42, 12, 18, 9, 9, 6,
            6, 221, 42, 18, 6, 6, 99, 0, 0, 24,
            12, 0, 6, 9, 12, 0,
        ], dtype=np.float64)

        # Add small epsilon to avoid division by zero
        train_counts = train_counts + 1e-6
        return train_counts / train_counts.sum()

    def log_predictions(self, detections: list[dict]) -> None:
        """
        Log a batch of predictions from a single inference call.

        Args:
            detections: List of detection dicts with 'class_id' and 'confidence'
        """
        with self._lock:
            for det in detections:
                self._predictions.append(det.get("class_id", 0))
                self._confidences.append(det.get("confidence", 0.0))
                self._total_logged += 1

    def compute_psi(self) -> float:
        """
        Compute Population Stability Index between training and recent predictions.

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
        Where P = recent prediction distribution, Q = training distribution
        """
        with self._lock:
            if len(self._predictions) < 50:
                return 0.0

            # Build recent distribution
            counts = Counter(self._predictions)
            recent = np.zeros(NUM_CLASSES, dtype=np.float64)
            for cls_id, count in counts.items():
                if 0 <= cls_id < NUM_CLASSES:
                    recent[cls_id] = count

            # Add epsilon and normalize
            recent = recent + 1e-6
            recent = recent / recent.sum()

            # Compute PSI
            psi = np.sum(
                (recent - self._training_distribution)
                * np.log(recent / self._training_distribution)
            )
            return float(psi)

    def compute_avg_confidence(self) -> float:
        """Compute average confidence of recent predictions."""
        with self._lock:
            if not self._confidences:
                return 0.0
            return float(np.mean(list(self._confidences)))

    def get_status(self) -> dict:
        """
        Get the current drift monitoring status.

        Returns:
            dict with status, psi_score, avg_confidence, predictions_logged, message
        """
        psi = self.compute_psi()
        avg_conf = self.compute_avg_confidence()
        confidence_drop = max(0, CONFIDENCE_BASELINE - avg_conf)

        # Determine status
        if psi > PSI_THRESHOLD_DRIFT or confidence_drop > CONFIDENCE_DECAY_THRESHOLD:
            status = STATUS_DRIFT
            message = (
                f"Data drift detected! PSI={psi:.4f} "
                f"(threshold={PSI_THRESHOLD_DRIFT}), "
                f"confidence drop={confidence_drop:.2%}. "
                f"Auto-retraining recommended."
            )
        elif psi > PSI_THRESHOLD_WARNING:
            status = STATUS_WARNING
            message = (
                f"Moderate distribution shift. PSI={psi:.4f}. "
                f"Monitoring closely."
            )
        else:
            status = STATUS_HEALTHY
            message = (
                f"Model predictions are stable. PSI={psi:.4f}, "
                f"avg confidence={avg_conf:.2%}."
            )

        return {
            "status": status,
            "psi_score": round(psi, 6),
            "avg_confidence": round(avg_conf, 4),
            "predictions_logged": self._total_logged,
            "message": message,
        }

    def should_retrain(self) -> bool:
        """Check if auto-retraining should be triggered."""
        status = self.get_status()
        if status["status"] != STATUS_DRIFT:
            return False

        # Check cooldown
        from src.monitoring.drift_config import RETRAIN_COOLDOWN_HOURS
        hours_since_last = (time.time() - self._last_retrain_time) / 3600
        if hours_since_last < RETRAIN_COOLDOWN_HOURS:
            return False

        return True

    def mark_retrained(self) -> None:
        """Mark that retraining has been triggered."""
        self._last_retrain_time = time.time()

        # Save drift event to log
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "retrain_triggered",
            **self.get_status(),
        }
        self._save_drift_event(event)

    def _save_drift_event(self, event: dict) -> None:
        """Append a drift event to the drift log file."""
        log_path = DRIFT_LOG_PATH
        log_path.parent.mkdir(parents=True, exist_ok=True)

        events = []
        if log_path.exists():
            with open(log_path, "r") as f:
                try:
                    events = json.load(f)
                except json.JSONDecodeError:
                    events = []

        events.append(event)
        with open(log_path, "w") as f:
            json.dump(events, f, indent=2)


# Module-level singleton
drift_monitor = DriftMonitor()
