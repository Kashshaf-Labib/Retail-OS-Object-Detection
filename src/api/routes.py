"""API route definitions for the retail shelf detection service."""

import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    EXPERIMENTS_PATH,
)
from src.inference.detector import detector
from src.inference.share_of_shelf import compute_share_of_shelf
from src.monitoring.drift_detector import drift_monitor
from src.api.schemas import (
    DetectResponse,
    DetectionResult,
    ShareOfShelfResponse,
    ShareOfShelfItem,
    HealthResponse,
    DriftStatusResponse,
)

router = APIRouter()


async def _read_image(file: UploadFile) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG file.")
    return image


@router.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.01, le=1.0),
    return_image: bool = Query(default=True),
):
    """
    Upload an image and get object detections.

    - **file**: Image file (JPEG/PNG)
    - **confidence**: Optional confidence threshold override
    - **return_image**: If true, returns annotated image; if false, returns JSON only
    """
    try:
        image = await _read_image(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

    result = detector.detect(image, confidence=confidence)

    # Log to drift monitor
    drift_monitor.log_predictions(result["detections"])

    if return_image:
        return StreamingResponse(
            io.BytesIO(result["annotated_image"]),
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(result["count"]),
                "X-Detections": json.dumps(result["detections"]),
            },
        )

    return DetectResponse(
        count=result["count"],
        detections=[DetectionResult(**d) for d in result["detections"]],
    )


@router.post("/api/share-of-shelf", response_model=ShareOfShelfResponse)
async def share_of_shelf(
    file: UploadFile = File(...),
    confidence: float = Query(default=None, ge=0.01, le=1.0),
):
    """
    Upload a shelf image and get Share of Shelf analytics.

    Returns percentage breakdown of each detected SKU.
    """
    try:
        image = await _read_image(file)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})

    result = detector.detect(image, confidence=confidence)
    analytics = compute_share_of_shelf(result["detections"])

    return ShareOfShelfResponse(
        total_products=analytics["total_products"],
        top_skus=[ShareOfShelfItem(**s) for s in analytics.get("top_skus", [])],
        all_skus=[ShareOfShelfItem(**s) for s in analytics.get("all_skus", [])],
    )


@router.get("/api/metrics")
async def get_metrics():
    """Return the experiment log with all model runs and metrics."""
    if EXPERIMENTS_PATH.exists():
        with open(EXPERIMENTS_PATH, "r") as f:
            experiments = json.load(f)
        return {"success": True, "experiments": experiments}
    return {"success": True, "experiments": [], "message": "No experiments logged yet."}


@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    model_loaded = detector._model is not None
    return HealthResponse(
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
    )


@router.get("/api/drift-status", response_model=DriftStatusResponse)
async def drift_status():
    """Return current data drift monitoring status."""
    status = drift_monitor.get_status()
    return DriftStatusResponse(**status)
