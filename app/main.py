"""
XAI Fake Image Detection System — FastAPI Application
"""
from fastapi import FastAPI
from fastapi.responses import Response

from app.api.routes import router
from app.core.detector import FakeImageDetector

app = FastAPI(
    title="XAI Fake Image Detector",
    description=(
        "Explainable AI system for detecting AI-generated and manipulated images. "
        "Returns REAL/FAKE classification with attention maps, Grad-CAM visualizations, "
        "metadata analysis, and a forensic PDF report."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def cors_middleware(request, call_next):
    if request.method == "OPTIONS":
        return Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


app.include_router(router)


@app.on_event("startup")
async def startup():
    """Pre-load the model on startup so the first request isn't slow."""
    FakeImageDetector.get()
