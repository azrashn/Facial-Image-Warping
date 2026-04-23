"""Router for CSV and PDF report exports."""

from __future__ import annotations

import csv
import io
from typing import List, Tuple

from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/export", tags=["export"])


def _sample_metrics() -> List[Tuple[float, float, float]]:
    """Provide sample metric rows for report exports."""
    return [
        (10.0, 30.0, 0.85),
        (8.0, 32.0, 0.88),
        (6.5, 34.2, 0.91),
    ]


@router.get("/csv")
async def export_csv() -> Response:
    """Export sample quality metrics as a CSV attachment."""
    rows = _sample_metrics()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["mse", "psnr", "ssim"])
    writer.writerows(rows)

    return Response(
        content=csv_buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=metrics_report.csv"},
    )


@router.get("/pdf")
async def export_pdf() -> Response:
    """Export a simple image processing report as a PDF attachment."""
    rows = _sample_metrics()
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="fpdf2 is required for PDF export. Install with: pip install fpdf2",
        ) from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Image Processing Report", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(50, 8, "MSE", border=1, align="C")
    pdf.cell(50, 8, "PSNR", border=1, align="C")
    pdf.cell(50, 8, "SSIM", border=1, align="C", ln=True)

    pdf.set_font("Helvetica", size=11)
    for mse, psnr, ssim in rows:
        pdf.cell(50, 8, f"{mse:.2f}", border=1, align="C")
        pdf.cell(50, 8, f"{psnr:.2f}", border=1, align="C")
        pdf.cell(50, 8, f"{ssim:.2f}", border=1, align="C", ln=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(
        0,
        8,
        "Placeholder for input/output visual examples.\n"
        "Image previews can be added here in a later integration step.",
    )

    pdf_bytes = bytes(pdf.output(dest="S"))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=image_processing_report.pdf"},
    )
