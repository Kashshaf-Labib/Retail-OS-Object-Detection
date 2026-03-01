"""YOLOv11 model wrapper with singleton pattern for efficient inference."""

import io
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import MODEL_PATH, CONFIDENCE_THRESHOLD, IMAGE_SIZE, DEVICE


class ShelfDetector:
    """Thread-safe singleton wrapper around the YOLOv11 model."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._initialized = True

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load the YOLO model weights."""
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. "
                f"Please place your trained best.pt in the models/ directory."
            )
        self._model = YOLO(str(path))
        print(f"✅ Model loaded from {path}")

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self.load_model()
        return self._model

    def detect(
        self,
        image: np.ndarray,
        confidence: float | None = None,
        imgsz: int | None = None,
    ) -> dict:
        """
        Run object detection on an image.

        Args:
            image: BGR numpy array (OpenCV format)
            confidence: Minimum confidence threshold
            imgsz: Inference image size

        Returns:
            dict with 'detections' list and 'annotated_image' as bytes
        """
        conf = confidence or CONFIDENCE_THRESHOLD
        size = imgsz or IMAGE_SIZE

        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=size,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        # Extract detections
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            det = {
                "class_id": cls_id,
                "class_name": result.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1),
                },
            }
            detections.append(det)

        # Generate annotated image
        annotated = result.plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        annotated_bytes = buffer.tobytes()

        return {
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_bytes,
        }


# Module-level singleton
detector = ShelfDetector()
