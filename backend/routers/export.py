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
from typing import Optional

from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class PDFRequest(BaseModel):
    """Schema matching the JSON payload the frontend may send."""
    before_image: str              # base64 data-URL or raw base64
    after_image: str               # base64 data-URL or raw base64
    mse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    summary: Optional[str] = ""
    history: Optional[list] = []
    orig_spectrum: Optional[str] = None
    proc_spectrum: Optional[str] = None


def _decode_b64_to_file(b64_str: str, filepath: str) -> None:
    """Strip an optional data-URL header and write decoded bytes to *filepath*."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    with open(filepath, "wb") as fh:
        fh.write(raw)


@router.post("/pdf")
async def export_pdf(payload: PDFRequest) -> StreamingResponse:
    """Export an image-processing report with before/after images as a PDF.

    Accepts a JSON body whose fields are defined by :class:`PDFRequest`.
    Images arrive as base64 strings; they are decoded to temporary PNG files
    before being handed to *fpdf2* (which requires file paths, not raw b64).
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="fpdf2 is required for PDF export. Install with: pip install fpdf2",
        ) from exc

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Decode images to temp files ──
        before_path = os.path.join(tmpdir, "before.png")
        after_path = os.path.join(tmpdir, "after.png")
        _decode_b64_to_file(payload.before_image, before_path)
        _decode_b64_to_file(payload.after_image, after_path)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Title ──
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "FaceDSP - Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, f"Generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(6)

        # ── Metrics table ──
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Quality Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "MSE", border=1, align="C")
        pdf.cell(50, 8, "PSNR", border=1, align="C")
        pdf.cell(50, 8, "SSIM", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", size=11)
        pdf.cell(50, 8, f"{payload.mse:.2f}", border=1, align="C")
        pdf.cell(50, 8, f"{payload.psnr:.2f} dB", border=1, align="C")
        pdf.cell(50, 8, f"{payload.ssim:.3f}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

        # ── Before / After images side by side ──
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(90, 10, "Before", align="C")
        pdf.cell(90, 10, "After", align="C", new_x="LMARGIN", new_y="NEXT")

        y_pos = pdf.get_y()
        pdf.image(before_path, x=15, y=y_pos, w=80)
        pdf.image(after_path, x=105, y=y_pos, w=80)
        pdf.ln(85)

        # ── Analysis Summary ──
        if payload.summary:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Analysis Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 5, payload.summary)
            pdf.ln(4)

        # ── Operation History ──
        if payload.history:
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Operation History", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
            for idx, op in enumerate(payload.history, 1):
                pdf.cell(0, 5, f"  {idx}. {op}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

        # ── Spectrum images (optional) ──
        if payload.orig_spectrum or payload.proc_spectrum:
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "FFT Spectrum", new_x="LMARGIN", new_y="NEXT")
            spec_y = pdf.get_y()

            if payload.orig_spectrum:
                orig_spec_path = os.path.join(tmpdir, "orig_spectrum.png")
                _decode_b64_to_file(payload.orig_spectrum, orig_spec_path)
                pdf.image(orig_spec_path, x=15, y=spec_y, w=80)

            if payload.proc_spectrum:
                proc_spec_path = os.path.join(tmpdir, "proc_spectrum.png")
                _decode_b64_to_file(payload.proc_spectrum, proc_spec_path)
                pdf.image(proc_spec_path, x=105, y=spec_y, w=80)

        # ── Output ──
        pdf_bytes = bytes(pdf.output())

    buf = io.BytesIO(pdf_bytes)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=FaceDSP_Report.pdf"},
    )

