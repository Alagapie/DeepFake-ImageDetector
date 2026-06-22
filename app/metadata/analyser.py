"""
Metadata Analysis Module.

Inspects EXIF data and returns structured flags.
Findings are treated as supporting evidence, not proof.
"""
from __future__ import annotations

from pathlib import Path

import piexif
from PIL import Image
from PIL.ExifTags import TAGS

# Software strings commonly associated with AI generation / editing
AI_SOFTWARE_KEYWORDS = [
    "midjourney", "stable diffusion", "dall-e", "dalle", "firefly",
    "photoshop", "lightroom", "gimp", "affinity", "canva",
    "topaz", "luminar", "capture one",
]

SEVERITY = {
    "NO_EXIF":           "high",
    "NO_CAMERA":         "high",
    "AI_SOFTWARE":       "high",
    "EDITING_SOFTWARE":  "medium",
    "METADATA_STRIPPED": "medium",
    "GPS_PRESENT":       "low",
    "FUTURE_DATE":       "medium",
    "CAMERA_PRESENT":    "info",
    "NO_SOFTWARE":       "info",
}


def analyse_metadata(image_path: str | Path) -> dict:
    """
    Returns:
        {
            "findings": [
                {"flag": "NO_CAMERA", "detail": "No camera make/model in EXIF", "severity": "high"},
                …
            ],
            "raw": { … }   # decoded EXIF tags (may be empty)
        }
    """
    image_path = Path(image_path)
    findings: list[dict] = []
    raw: dict = {}

    try:
        image = Image.open(image_path)
    except Exception as e:
        return {"findings": [{"flag": "READ_ERROR", "detail": str(e), "severity": "high"}], "raw": {}}

    # ── Try Pillow EXIF ───────────────────────────────────────────────────────
    exif_raw = image._getexif() if hasattr(image, "_getexif") else None

    if not exif_raw:
        # Try piexif for more aggressive extraction
        try:
            piexif_data = piexif.load(str(image_path))
            has_any = any(piexif_data.get(ifd) for ifd in ("0th", "1st", "Exif", "GPS"))
            if not has_any:
                findings.append(_flag("NO_EXIF", "No EXIF metadata found in image"))
                return {"findings": findings, "raw": {}}
        except Exception:
            findings.append(_flag("NO_EXIF", "No EXIF metadata found in image"))
            return {"findings": findings, "raw": {}}

    if exif_raw:
        raw = {TAGS.get(k, k): v for k, v in exif_raw.items() if isinstance(v, (str, int, float, bytes))}

    # ── Camera make / model ───────────────────────────────────────────────────
    has_camera = "Make" in raw or "Model" in raw
    if has_camera:
        make = raw.get("Make", "")
        model = raw.get("Model", "")
        findings.append(_flag("CAMERA_PRESENT", f"Camera: {make} {model}".strip(), "info"))
    else:
        findings.append(_flag("NO_CAMERA", "No camera make/model found in metadata"))

    # ── Software ──────────────────────────────────────────────────────────────
    software = str(raw.get("Software", "")).strip()
    if software:
        lower = software.lower()
        is_ai = any(kw in lower for kw in AI_SOFTWARE_KEYWORDS)
        if is_ai:
            # Check if it's purely an AI generator vs editor
            ai_gen_kw = ["midjourney", "stable diffusion", "dall-e", "dalle", "firefly"]
            if any(kw in lower for kw in ai_gen_kw):
                findings.append(_flag("AI_SOFTWARE", f"AI generation software detected: {software}"))
            else:
                findings.append(_flag("EDITING_SOFTWARE", f"Image editing software detected: {software}"))
        else:
            findings.append(_flag("NO_SOFTWARE", f"Software tag present: {software}", "info"))
    else:
        findings.append(_flag("NO_SOFTWARE", "No software tag in metadata", "info"))

    # ── GPS ───────────────────────────────────────────────────────────────────
    if "GPSInfo" in (exif_raw or {}):
        findings.append(_flag("GPS_PRESENT", "GPS location data embedded in image", "low"))

    # ── DateTime sanity check ─────────────────────────────────────────────────
    from datetime import datetime
    dt_str = raw.get("DateTimeOriginal") or raw.get("DateTime")
    if dt_str:
        try:
            dt = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
            if dt.year > datetime.now().year:
                findings.append(_flag("FUTURE_DATE", f"Image date is in the future: {dt_str}"))
        except ValueError:
            pass

    return {"findings": findings, "raw": raw}


def _flag(flag: str, detail: str, severity: str | None = None) -> dict:
    return {
        "flag": flag,
        "detail": detail,
        "severity": severity or SEVERITY.get(flag, "info"),
    }
