"""
FastAPI Routes — Phase 8.

POST /analyze     Upload an image → returns full JSON analysis
GET  /report/{id} Download the PDF for a completed analysis
"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config import REPORT_DIR, UPLOAD_DIR
from app.core.pipeline import run_analysis

router = APIRouter()

# In-memory store: analysis_id → EvidenceReport
_results: dict = {}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image (JPEG, PNG, WEBP).
    Returns the full evidence report as JSON.
    """
    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WEBP."
        )

    # Save upload
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        report = run_analysis(upload_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    _results[report.analysis_id] = report

    # Serialise report (exclude file paths from JSON response)
    response = {
        "analysis_id":       report.analysis_id,
        "filename":          report.filename,
        "timestamp":         report.timestamp,
        "prediction":        report.prediction,
        "confidence":        report.confidence,
        "probabilities":     report.probabilities,
        "attention_regions": report.attention_regions,
        "gradcam_regions":   report.gradcam_regions,
        "metadata_findings": report.metadata_findings,
        "evidence_items":    report.evidence_items,
        "nl_explanation":    report.nl_explanation,
        "pdf_download_url":  f"/report/{report.analysis_id}",
    }
    return JSONResponse(content=response)


@router.get("/report/{analysis_id}")
async def download_report(analysis_id: str):
    """Download the forensic PDF report for a completed analysis."""
    pdf_path = REPORT_DIR / f"report_{analysis_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No report found for analysis ID '{analysis_id}'."
        )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"forensic_report_{analysis_id}.pdf",
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
