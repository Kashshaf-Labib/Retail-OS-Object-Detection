"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """Single detection bounding box."""
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: dict


class DetectResponse(BaseModel):
    """Response from /api/detect endpoint."""
    success: bool = True
    count: int
    detections: list[DetectionResult]


class ShareOfShelfItem(BaseModel):
    """Single SKU in share of shelf breakdown."""
    sku: str
    count: int
    percentage: float


class ShareOfShelfResponse(BaseModel):
    """Response from /api/share-of-shelf endpoint."""
    success: bool = True
    total_products: int
    top_skus: list[ShareOfShelfItem]
    all_skus: list[ShareOfShelfItem]


class HealthResponse(BaseModel):
    """Response from /api/health endpoint."""
    status: str = "healthy"
    model_loaded: bool
    model_path: str


class DriftStatusResponse(BaseModel):
    """Response from /api/drift-status endpoint."""
    status: str  # HEALTHY, WARNING, DRIFT_DETECTED
    psi_score: float
    avg_confidence: float
    predictions_logged: int
    message: str
