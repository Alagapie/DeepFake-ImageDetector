"""
Evidence Aggregation Engine.

Combines outputs from detector, attention, Grad-CAM, and metadata
into a single structured EvidenceReport with confidence tiers and
detailed natural language explanations.

NOTE: All strings use only latin-1 safe characters (no em-dashes, smart
quotes, or other unicode) so fpdf2 can render them without a Unicode font.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


def confidence_tier(conf: float) -> tuple[str, str]:
    if conf >= 90:
        return "VERY HIGH", "The model is highly certain about this classification."
    elif conf >= 75:
        return "HIGH", "The model is confident in this classification."
    elif conf >= 60:
        return "MODERATE", "The model shows moderate confidence. Review supporting evidence carefully."
    elif conf >= 45:
        return "LOW", "The model is uncertain. This image is borderline and requires manual review."
    else:
        return "VERY LOW", "The model has very low confidence. Classification may not be reliable."


@dataclass
class EvidenceReport:
    analysis_id:      str
    filename:         str
    timestamp:        str
    prediction:       str
    confidence:       float
    confidence_tier:  str  = ""
    confidence_note:  str  = ""
    probabilities:    dict = field(default_factory=dict)
    attention_regions: list = field(default_factory=list)
    attention_scores:  dict = field(default_factory=dict)
    gradcam_regions:   list = field(default_factory=list)
    metadata_findings: list = field(default_factory=list)
    evidence_items:    list = field(default_factory=list)
    nl_explanation:    str  = ""
    attention_heatmap_path: str = ""
    gradcam_heatmap_path:   str = ""
    report_pdf_path:        str = ""


def _safe(text: str) -> str:
    """Replace characters outside latin-1 range with ASCII equivalents."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: encode to latin-1, replacing any remaining unknowns
    return text.encode("latin-1", errors="replace").decode("latin-1")


def build_evidence_report(
    *,
    analysis_id:       str,
    filename:          str,
    prediction:        dict,
    attention_regions: list,
    attention_scores:  dict,
    gradcam_regions:   list,
    metadata:          dict,
) -> EvidenceReport:

    timestamp  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    tier, note = confidence_tier(prediction["confidence"])

    evidence_items = _aggregate_evidence(
        prediction=prediction,
        tier=tier,
        attention_regions=attention_regions,
        attention_scores=attention_scores,
        gradcam_regions=gradcam_regions,
        metadata_findings=metadata["findings"],
    )

    report = EvidenceReport(
        analysis_id=analysis_id,
        filename=filename,
        timestamp=timestamp,
        prediction=prediction["label"],
        confidence=prediction["confidence"],
        confidence_tier=tier,
        confidence_note=note,
        probabilities=prediction["probabilities"],
        attention_regions=attention_regions,
        attention_scores=attention_scores,
        gradcam_regions=gradcam_regions,
        metadata_findings=metadata["findings"],
        evidence_items=evidence_items,
    )

    report.nl_explanation = generate_explanation(report)
    return report


def _aggregate_evidence(
    *,
    prediction:        dict,
    tier:              str,
    attention_regions: list,
    attention_scores:  dict,
    gradcam_regions:   list,
    metadata_findings: list,
) -> list:
    items = []
    conf  = prediction["confidence"]
    label = prediction["label"]

    items.append({
        "source":   "CLASSIFIER",
        "strength": tier,
        "detail": _safe(
            f"Image classified as {label} with {conf:.1f}% confidence "
            f"(tier: {tier}). "
            f"FAKE probability: {prediction['probabilities'].get('FAKE', 0):.1f}%, "
            f"REAL probability: {prediction['probabilities'].get('REAL', 0):.1f}%."
        ),
    })

    if attention_regions:
        top_zone  = max(attention_scores, key=attention_scores.__getitem__) if attention_scores else None
        top_score = attention_scores.get(top_zone, 0) if top_zone else 0
        detail = f"Model attention concentrated on: {_join(attention_regions)}. "
        if top_zone:
            detail += f"Strongest activation in '{top_zone}' (score: {top_score:.3f}). "
        detail += (
            "High attention on facial features is consistent with deepfake detection "
            "patterns, as generative models often introduce artifacts in these areas."
        )
        items.append({"source": "ATTENTION MAP", "strength": "supporting", "detail": _safe(detail)})

    if gradcam_regions:
        detail = (
            f"Grad-CAM activation strongest in: {_join(gradcam_regions)}. "
            "These regions most directly influenced the final classification score. "
            "Activation in skin texture, hair boundaries, or background edges may "
            "indicate GAN or diffusion model blending artifacts."
        )
        items.append({"source": "GRAD-CAM", "strength": "supporting", "detail": _safe(detail)})

    high_flags   = [f for f in metadata_findings if f["severity"] == "high"]
    medium_flags = [f for f in metadata_findings if f["severity"] == "medium"]
    info_flags   = [f for f in metadata_findings if f["severity"] == "info"]

    for f in high_flags:
        items.append({
            "source":   "METADATA",
            "strength": "HIGH CONCERN",
            "detail":   _safe(f["detail"] + _metadata_context(f["flag"])),
        })
    for f in medium_flags:
        items.append({
            "source":   "METADATA",
            "strength": "MODERATE CONCERN",
            "detail":   _safe(f["detail"] + _metadata_context(f["flag"])),
        })
    for f in info_flags:
        items.append({
            "source":   "METADATA",
            "strength": "INFO",
            "detail":   _safe(f["detail"]),
        })

    return items


def _metadata_context(flag: str) -> str:
    ctx = {
        "NO_EXIF":          " - AI-generated images typically lack EXIF data as they are not captured by a physical camera.",
        "NO_CAMERA":        " - Absence of camera make/model is a common indicator of synthetic or heavily edited imagery.",
        "AI_SOFTWARE":      " - Presence of known AI generation software is a strong indicator of synthetic origin.",
        "EDITING_SOFTWARE": " - Image editing software detected. This does not confirm manipulation but is a supporting signal.",
        "FUTURE_DATE":      " - A future creation date is technically impossible for a real capture and suggests metadata tampering.",
        "GPS_PRESENT":      " - GPS data is present. Real cameras embed location; this alone is not a manipulation indicator.",
        "CAMERA_PRESENT":   " - Camera metadata is present, which is consistent with a real photograph.",
    }
    return ctx.get(flag, "")


def generate_explanation(report: EvidenceReport) -> str:
    parts = []

    verdict    = "AI-generated or manipulated" if report.prediction == "FAKE" else "likely authentic"
    tier_lower = report.confidence_tier.lower()
    parts.append(
        f"This image has been classified as {verdict} with {tier_lower} confidence "
        f"({report.confidence:.1f}%). {report.confidence_note}"
    )

    if report.attention_regions:
        top_zones = report.attention_regions[:2]
        parts.append(
            f"The model's attention focused most strongly on the {_join(top_zones)} of the image. "
            f"These are the regions the Vision Transformer weighted most heavily when forming its decision."
        )
        face_zones = [z for z in report.attention_regions if any(
            kw in z for kw in ["eye", "forehead", "cheek", "mouth", "chin", "jaw", "nose", "face"]
        )]
        if face_zones:
            parts.append(
                f"Attention on facial zones ({_join(face_zones)}) is forensically significant: "
                f"deepfake models frequently produce subtle inconsistencies in skin texture, "
                f"eye reflections, and facial boundary blending."
            )

    if report.gradcam_regions:
        parts.append(
            f"Grad-CAM identified the {_join(report.gradcam_regions)} as contributing most directly "
            f"to the classification score, providing a second independent spatial signal."
        )

    high_meta = [f for f in report.metadata_findings if f["severity"] == "high"]
    med_meta  = [f for f in report.metadata_findings if f["severity"] == "medium"]
    cam_meta  = [f for f in report.metadata_findings if f["flag"] == "CAMERA_PRESENT"]

    if high_meta:
        concerns = "; ".join(f["detail"] for f in high_meta)
        parts.append(
            f"Metadata inspection raised significant forensic concerns: {concerns}. "
            f"These findings independently support the classifier verdict."
        )
    elif med_meta:
        concerns = "; ".join(f["detail"] for f in med_meta)
        parts.append(f"Metadata inspection noted: {concerns}.")
    elif cam_meta:
        parts.append(
            f"Metadata found camera information ({cam_meta[0]['detail']}), "
            f"consistent with a real photograph. This partially counters the classifier verdict."
        )
    else:
        parts.append("No significant metadata anomalies were detected.")

    is_fake = report.prediction == "FAKE"
    conf    = report.confidence

    if is_fake and conf >= 75 and len(high_meta) >= 1:
        parts.append(
            "Multiple independent signals converge on a FAKE classification: "
            "classifier, spatial activation maps, and metadata findings are all consistent. "
            "This image warrants serious scrutiny."
        )
    elif is_fake and conf >= 60:
        parts.append(
            "The classifier and visual activation maps both indicate manipulation, "
            "though confidence is moderate. Manual review is recommended."
        )
    elif is_fake:
        parts.append(
            "The classifier leans toward FAKE but with low confidence. "
            "Treat findings as investigative leads rather than definitive conclusions."
        )
    else:
        parts.append(
            "The classifier indicates this image is likely authentic. "
            "Review the activation maps to confirm no unexpected patterns are present."
        )

    parts.append(
        "All findings are produced by automated analysis and should be considered "
        "alongside other investigative evidence. This report does not constitute legal proof."
    )

    return _safe(" ".join(parts))


def _join(items: list) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"
