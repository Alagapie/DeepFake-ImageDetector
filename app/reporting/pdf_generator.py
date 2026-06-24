"""
Forensic PDF Report Generator.
All strings are sanitised to latin-1 safe characters before rendering.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from app.config import REPORT_DIR, REPORT_DISCLAIMER
from app.evidence.engine import EvidenceReport

C_DARK    = (15,  23,  42)
C_FAKE    = (220, 38,  38)
C_REAL    = (22,  163, 74)
C_ACCENT  = (99,  102, 241)
C_WARN    = (234, 88,  12)
C_MID     = (100, 116, 139)
C_ROW_ALT = (241, 245, 249)

TIER_COLORS = {
    "VERY HIGH": C_FAKE,
    "HIGH":      (234, 88,  12),
    "MODERATE":  (202, 138, 4),
    "LOW":       C_MID,
    "VERY LOW":  C_MID,
}


def _s(text: str) -> str:
    """Sanitise string to latin-1 safe characters for fpdf2 core fonts."""
    replacements = {
        "\u2014": "-", "\u2013": "-",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2022": "*", "\u2026": "...",
        "\u00e2\u0080\u0094": "-",
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ForensicReport(FPDF):
    def __init__(self, report: EvidenceReport):
        super().__init__()
        self.report = report
        self.set_auto_page_break(auto=True, margin=22)
        self.add_page()

    def header(self):
        self.set_fill_color(*C_DARK)
        self.rect(0, 0, 210, 18, "F")
        self.set_y(4)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "XAI FORENSIC IMAGE ANALYSIS REPORT", align="C")
        self.set_text_color(*C_DARK)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_MID)
        self.multi_cell(0, 4, _s(REPORT_DISCLAIMER), align="C")
        self.cell(0, 4, _s(f"Page {self.page_no()}  |  ID: {self.report.analysis_id}"), align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_fill_color(*C_ACCENT)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, _s(f"  {title}"), fill=True, ln=True)
        self.set_text_color(*C_DARK)
        self.ln(2)

    def kv_row(self, key: str, value: str, value_color=None):
        self.set_font("Helvetica", "B", 9)
        self.set_x(15)
        self.cell(52, 6, _s(key))
        if value_color:
            self.set_text_color(*value_color)
            self.set_font("Helvetica", "B", 9)
        else:
            self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 6, _s(value))
        self.set_text_color(*C_DARK)

    def prose(self, text: str, indent: int = 15):
        self.set_font("Helvetica", "", 9)
        self.set_x(indent)
        self.multi_cell(180, 5.5, _s(text))

    def image_block(self, path: str, caption: str, w: int = 88):
        if not path or not Path(path).exists():
            return
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_MID)
        self.cell(0, 5, _s(caption), ln=True, align="C")
        self.set_text_color(*C_DARK)
        self.ln(2)


def generate_pdf(report: EvidenceReport) -> str:
    pdf = ForensicReport(report)
    pdf.set_margin(15)
    pdf.set_y(22)

    # ── Section 1: Verdict ────────────────────────────────────────────────────
    pdf.section_title("1. VERDICT & CONFIDENCE")

    verdict_color = C_FAKE if report.prediction == "FAKE" else C_REAL
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*verdict_color)
    pdf.cell(0, 14, _s(report.prediction), align="C", ln=True)
    pdf.set_text_color(*C_DARK)

    tier_color = TIER_COLORS.get(report.confidence_tier, C_MID)
    pdf.set_fill_color(*tier_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(70)
    pdf.cell(70, 7,
        _s(f"Confidence: {report.confidence:.1f}%  |  {report.confidence_tier}"),
        fill=True, align="C", ln=True)
    pdf.set_text_color(*C_DARK)
    pdf.ln(1)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*C_MID)
    pdf.set_x(15)
    pdf.multi_cell(180, 5, _s(report.confidence_note))
    pdf.set_text_color(*C_DARK)
    pdf.ln(2)

    pdf.kv_row("File:", report.filename)
    pdf.kv_row("Analysis ID:", report.analysis_id)
    pdf.kv_row("Timestamp:", report.timestamp)
    pdf.ln(2)

    # Probability bars
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(15)
    pdf.cell(0, 5, "Class probabilities:", ln=True)
    for label, prob in sorted(report.probabilities.items()):
        color = C_FAKE if label == "FAKE" else C_REAL
        bar_w = int((prob / 100) * 155)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(28, 5, _s(f"{label}:"))
        pdf.set_fill_color(*color)
        y_bar = pdf.get_y() + 1
        x_bar = pdf.get_x()
        pdf.rect(x_bar, y_bar, bar_w, 4, "F")
        pdf.set_x(x_bar + max(bar_w, 2) + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(20, 5, _s(f"{prob:.1f}%"), ln=True)
    pdf.ln(3)

    # ── Section 2: Explanation ────────────────────────────────────────────────
    pdf.section_title("2. FORENSIC EXPLANATION")
    pdf.prose(report.nl_explanation)

    # ── Section 3: Attention Map ──────────────────────────────────────────────
    if report.attention_heatmap_path:
        pdf.section_title("3. ATTENTION MAP (Attention Rollout)")
        pdf.image_block(
            report.attention_heatmap_path,
            "Heatmap showing which regions the ViT attended to most strongly"
        )
        if report.attention_regions:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(15)
            pdf.cell(0, 5, "High-attention zones (ranked by activation strength):", ln=True)
            for i, region in enumerate(report.attention_regions, 1):
                score = report.attention_scores.get(region, 0)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(20)
                pdf.cell(0, 5, _s(f"{i}.  {region}  (score: {score:.3f})"), ln=True)
            pdf.ln(2)

        if report.attention_scores:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(15)
            pdf.cell(0, 5, "Full attention zone scores (3x3 spatial grid):", ln=True)
            pdf.ln(1)

            pdf.set_fill_color(*C_ACCENT)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_x(15)
            pdf.cell(120, 6, "Zone", fill=True)
            pdf.cell(40,  6, "Score", fill=True, align="C")
            pdf.cell(0,   6, "", ln=True)
            pdf.set_text_color(*C_DARK)

            sorted_zones = sorted(report.attention_scores.items(), key=lambda x: x[1], reverse=True)
            for idx, (zone, score) in enumerate(sorted_zones):
                fill = idx % 2 == 0
                if fill:
                    pdf.set_fill_color(*C_ROW_ALT)
                pdf.set_font("Helvetica", "B" if idx == 0 else "", 8)
                pdf.set_x(15)
                pdf.cell(120, 5.5, _s(f"  {zone}"), fill=fill)
                bar_w = int(score * 35)
                pdf.set_fill_color(*(C_FAKE if report.prediction == "FAKE" else C_REAL))
                if bar_w > 0:
                    pdf.rect(pdf.get_x() + 1, pdf.get_y() + 1, bar_w, 3.5, "F")
                if fill:
                    pdf.set_fill_color(*C_ROW_ALT)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_x(15 + 120 + 5)
                pdf.cell(35, 5.5, _s(f"{score:.4f}"), align="R", ln=True)

    # ── Section 4: Grad-CAM ───────────────────────────────────────────────────
    if report.gradcam_heatmap_path:
        pdf.section_title("4. GRAD-CAM ACTIVATION MAP")
        pdf.image_block(
            report.gradcam_heatmap_path,
            "Gradient-weighted activation map - regions most influential to classification"
        )
        if report.gradcam_regions:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(15)
            pdf.cell(0, 5, "Peak activation regions:", ln=True)
            for region in report.gradcam_regions:
                pdf.set_font("Helvetica", "", 9)
                pdf.set_x(20)
                pdf.cell(0, 5, _s(f"*  {region}"), ln=True)

    # ── Section 5: Metadata ───────────────────────────────────────────────────
    pdf.section_title("5. METADATA ANALYSIS")
    if not report.metadata_findings:
        pdf.prose("No metadata findings.")
    else:
        sev_colors = {
            "high":   C_FAKE,
            "medium": C_WARN,
            "low":    C_MID,
            "info":   C_ACCENT,
        }
        for finding in report.metadata_findings:
            if pdf.will_page_break(15):
                pdf.add_page()
            sev = finding.get("severity", "info")
            pdf.set_fill_color(*sev_colors.get(sev, C_DARK))
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_x(15)
            pdf.cell(20, 5, _s(f"  {sev.upper()}"), fill=True)
            pdf.set_text_color(*C_DARK)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.ln(5)
            pdf.set_x(20)
            pdf.multi_cell(175, 5, _s(finding["detail"]))
            pdf.ln(3)
        pdf.ln(2)

    # ── Section 6: Evidence Summary ───────────────────────────────────────────
    pdf.section_title("6. EVIDENCE SUMMARY")
    for item in report.evidence_items:
        if pdf.will_page_break(15):
            pdf.add_page()
        pdf.set_fill_color(*C_DARK)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_x(15)
        pdf.cell(32, 5, _s(f"  {item['source']}"), fill=True)
        strength = item.get("strength", "")
        s_color  = C_FAKE if "HIGH" in strength else (C_WARN if "MODERATE" in strength else C_MID)
        pdf.set_fill_color(*s_color)
        pdf.cell(30, 5, _s(f"  {strength}"), fill=True)
        pdf.set_text_color(*C_DARK)
        pdf.ln(5)
        pdf.set_x(20)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.multi_cell(170, 5, _s(item["detail"]))
        pdf.ln(3)

    out_path = REPORT_DIR / f"report_{report.analysis_id}.pdf"
    pdf.output(str(out_path))
    return str(out_path)
