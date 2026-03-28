"""PDF report generation for FRA diagnostic summaries (ReportLab)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(
    result: dict[str, Any],
    *,
    out_path: Optional[str] = None,
    title: str = "FRA Diagnostic Report",
) -> str:
    """
    Build a PDF report from a pipeline or :func:`src.analyzer.advanced_analysis` result.

    Embeds plot PNG paths when present under ``result['plots']``.

    Parameters
    ----------
    result
        Must include ``diagnosis`` or top-level fault fields; may include ``plots`` URLs/paths.
    out_path
        Output ``.pdf`` path. Default: ``reports/fra_report_<timestamp>.pdf``.
    title
        Document title on the cover block.

    Returns
    -------
    str
        Absolute path to the written PDF.
    """
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = out_path or os.path.join("reports", f"fra_report_{ts}.pdf")

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0ea5e9"),
        spaceAfter=16,
    )

    diag = result.get("diagnosis") or {}
    fault = str(diag.get("fault", result.get("fault_type", "N/A")))
    conf = float(diag.get("confidence", result.get("confidence", 0)))
    sev = str(diag.get("severity", result.get("severity", "N/A")))
    rec = str(diag.get("recommendation", result.get("recommendation", "")))
    expl = str(diag.get("explanation", result.get("explanation", "")))
    an = result.get("anomaly") or {}
    an_score = an.get("anomaly_score", result.get("anomaly_score", "N/A"))

    content: list = []
    content.append(Paragraph(title, title_style))
    content.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    content.append(Spacer(1, 12))
    content.append(Paragraph("-" * 72, styles["Normal"]))
    content.append(Spacer(1, 12))

    data = [
        ["Field", "Value"],
        ["Fault assessment", fault],
        ["Confidence (0–100)", f"{conf:.1f}"],
        ["Severity", sev],
        ["Anomaly score (0–100)", str(an_score)],
        ["Correlation (vs ref)", f"{float(result.get('correlation', 0)):.4f}"],
        ["Max deviation", f"{float(result.get('shift', result.get('max_deviation_db', 0))):.2f} dB"],
    ]

    table = Table(data, colWidths=[150, 320])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    content.append(table)
    content.append(Spacer(1, 16))

    plots = result.get("plots") or {}
    for label, key in (
        ("Single FRA (test)", "single"),
        ("Comparison", "comparison"),
        ("Difference map", "difference"),
    ):
        p = plots.get(key)
        if p and os.path.isfile(p):
            content.append(Paragraph(f"<b>{label}</b>", styles["Heading3"]))
            content.append(Spacer(1, 6))
            try:
                img = Image(p, width=6.2 * inch, height=3.1 * inch)
                content.append(img)
            except OSError:
                content.append(Paragraph("(image not available)", styles["Normal"]))
            content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Recommendation</b>", styles["Heading3"]))
    content.append(Paragraph(rec or "N/A", styles["Normal"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Explanation</b>", styles["Heading3"]))
    content.append(Paragraph(expl or "N/A", styles["Normal"]))

    doc.build(content)
    return os.path.abspath(file_path)


__all__ = ["generate_report"]
