"""
Analysis Pipeline.

Single entry point that orchestrates:
    Detector → Attention → Grad-CAM → Metadata → Evidence → PDF
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from app.config import REPORT_DIR, UPLOAD_DIR
from app.core.detector import FakeImageDetector
from app.evidence.engine import EvidenceReport, build_evidence_report
from app.explainability.attention import extract_attention_rollout
from app.explainability.gradcam import extract_gradcam
from app.metadata.analyser import analyse_metadata
from app.reporting.pdf_generator import generate_pdf


def _save_heatmap(array: np.ndarray, filename: str) -> str:
    """Save an H×W×3 numpy array as PNG and return the path."""
    path = REPORT_DIR / filename
    Image.fromarray(array).save(str(path))
    return str(path)


def run_analysis(image_path: str | Path) -> EvidenceReport:
    """
    Full pipeline for a single image file.
    Returns a completed EvidenceReport with PDF path set.
    """
    image_path = Path(image_path)
    analysis_id = uuid.uuid4().hex[:12]
    image = Image.open(image_path).convert("RGB")

    detector = FakeImageDetector.get()

    # ── Phase 1: Detection ────────────────────────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Running detection…")
    prediction = detector.predict(image)
    print(f"[Pipeline:{analysis_id}] {prediction['label']} ({prediction['confidence']:.1f}%)")

    # Resolve class index for Grad-CAM target
    id2label = detector.vit_model.config.id2label
    label2id = {v.upper(): k for k, v in id2label.items()}
    target_class_idx = label2id[prediction["label"]]

    # ── Phase 2: Attention Maps ───────────────────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Extracting attention maps…")
    attention_overlay, raw_mask, attention_regions = extract_attention_rollout(
        detector.vit_model, detector.vit_processor, image
    )
    attention_path = _save_heatmap(
        attention_overlay, f"attention_{analysis_id}.png"
    )
    h = raw_mask.shape[0]
    zones = [(0, h // 3), (h // 3, 2 * h // 3), (2 * h // 3, h)]
    zone_names = ["upper region", "central region", "lower region"]
    attention_scores = {
        zone_names[i]: float(raw_mask[s:e].mean())
        for i, (s, e) in enumerate(zones)
    }

    # ── Phase 3: Grad-CAM ─────────────────────────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Generating Grad-CAM…")
    gradcam_overlay, gradcam_regions = extract_gradcam(
        detector.vit_model, detector.vit_processor, image, target_class_idx
    )
    gradcam_path = _save_heatmap(
        gradcam_overlay, f"gradcam_{analysis_id}.png"
    )

    # ── Phase 4: Metadata ─────────────────────────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Analysing metadata…")
    metadata = analyse_metadata(image_path)

    # ── Phase 5 + 6: Evidence + NL Explanation ────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Building evidence report…")
    report = build_evidence_report(
        analysis_id=analysis_id,
        filename=image_path.name,
        prediction=prediction,
        attention_regions=attention_regions,
        attention_scores=attention_scores,
        gradcam_regions=gradcam_regions,
        metadata=metadata,
    )
    report.attention_heatmap_path = attention_path
    report.gradcam_heatmap_path   = gradcam_path

    # ── Phase 7: PDF ──────────────────────────────────────────────────────────
    print(f"[Pipeline:{analysis_id}] Generating PDF report…")
    report.report_pdf_path = generate_pdf(report)

    print(f"[Pipeline:{analysis_id}] Done. Report: {report.report_pdf_path}")
    return report
