"""FastAPI application entry point for the retail shelf detection service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import PROJECT_ROOT, MODEL_PATH
from src.inference.detector import detector
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, cleanup on shutdown."""
    print("Starting Retail Shelf Detection API...")
    try:
        detector.load_model()
        print(f"Model loaded: {MODEL_PATH}")
    except FileNotFoundError as e:
        print(f"{e}")
        print("   The API will start, but /api/detect will fail until weights are provided.")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="Retail Shelf Detection API",
    description=(
        "YOLOv11-based object detection for retail shelf products. "
        "Detects 76 SKU classes and computes Share of Shelf analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Register API routes
app.include_router(router)

# Serve frontend static files
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Retail Shelf Detection API is running. Visit /docs for API documentation."}
