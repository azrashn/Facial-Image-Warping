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


import base64
import os
import tempfile
from fastapi import Form

@router.post("/pdf")
async def export_pdf(
    before_image: str = Form(...),
    after_image: str = Form(...),
    mse: float = Form(0.0),
    psnr: float = Form(0.0),
    ssim: float = Form(0.0),
) -> Response:
    """Export an image processing report with before/after images as a PDF attachment."""
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
    pdf.cell(0, 10, "Image Processing Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(50, 8, "MSE", border=1, align="C")
    pdf.cell(50, 8, "PSNR", border=1, align="C")
    pdf.cell(50, 8, "SSIM", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", size=11)
    pdf.cell(50, 8, f"{mse:.2f}", border=1, align="C")
    pdf.cell(50, 8, f"{psnr:.2f}", border=1, align="C")
    pdf.cell(50, 8, f"{ssim:.2f}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    
    def decode_b64(b64_str, filename):
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        with open(filename, "wb") as f:
            f.write(base64.b64decode(b64_str))

    with tempfile.TemporaryDirectory() as tmpdir:
        before_path = os.path.join(tmpdir, "before.png")
        after_path = os.path.join(tmpdir, "after.png")
        
        decode_b64(before_image, before_path)
        decode_b64(after_image, after_path)
        
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(90, 10, "Before", align="C")
        pdf.cell(90, 10, "After", align="C", new_x="LMARGIN", new_y="NEXT")
        
        # Add images side by side
        y_pos = pdf.get_y()
        pdf.image(before_path, x=15, y=y_pos, w=80)
        pdf.image(after_path, x=105, y=y_pos, w=80)

    pdf_bytes = bytes(pdf.output())
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=image_processing_report.pdf"},
    )
