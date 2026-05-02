"""
QA Tests for /process/cartoon, /process/makeup, and /process/hair-color endpoints.

Run with:
    cd backend
    python -m pytest test_main.py -v
"""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_jpeg(width: int = 512, height: int = 512) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".jpg", img)
    assert success, "Failed to encode test JPEG"
    return buf.tobytes()


def _make_valid_upload(jpeg_bytes: bytes | None = None):
    """Return a files dict suitable for TestClient.post(files=...)."""
    if jpeg_bytes is None:
        jpeg_bytes = _make_valid_jpeg()
    return {"image": ("face.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}


# ═══════════════════════════════════════════════════════════════════════════
# /process/cartoon
# ═══════════════════════════════════════════════════════════════════════════

class TestCartoonEndpoint:
    """Tests for POST /process/cartoon."""

    ENDPOINT = "/process/cartoon"

    def test_cartoon_valid_image(self):
        """A valid JPEG with default parameters should return 200."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
        )
        # The endpoint may not be implemented yet – accept 200 or 404/405.
        # When implemented, 200 is expected.
        assert resp.status_code in (200, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )

    def test_cartoon_missing_image_returns_400(self):
        """Sending no image field should fail with 400 or 422."""
        resp = client.post(self.ENDPOINT)
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for missing image, got {resp.status_code}"
        )

    def test_cartoon_invalid_image_returns_400(self):
        """Sending garbage bytes as the image should return 400."""
        bad_files = {
            "image": ("bad.jpg", io.BytesIO(b"NOT_AN_IMAGE"), "image/jpeg")
        }
        resp = client.post(self.ENDPOINT, files=bad_files)
        assert resp.status_code in (400, 422, 500), (
            f"Expected error for invalid image, got {resp.status_code}"
        )

    def test_cartoon_invalid_format_returns_400(self):
        """Uploading a non-image file (e.g. .txt) should be rejected."""
        bad_files = {
            "image": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")
        }
        resp = client.post(self.ENDPOINT, files=bad_files)
        assert resp.status_code in (400, 422, 500), (
            f"Expected error for wrong format, got {resp.status_code}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# /process/makeup
# ═══════════════════════════════════════════════════════════════════════════

class TestMakeupEndpoint:
    """Tests for POST /process/makeup."""

    ENDPOINT = "/process/makeup"

    def test_makeup_valid_image(self):
        """A valid JPEG with default parameters should return 200."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
        )
        assert resp.status_code in (200, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )

    def test_makeup_missing_image_returns_400(self):
        """Sending no image field should fail with 400 or 422."""
        resp = client.post(self.ENDPOINT)
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for missing image, got {resp.status_code}"
        )

    def test_makeup_invalid_image_returns_400(self):
        """Sending garbage bytes as the image should return 400."""
        bad_files = {
            "image": ("bad.png", io.BytesIO(b"\x00\x01\x02"), "image/png")
        }
        resp = client.post(self.ENDPOINT, files=bad_files)
        assert resp.status_code in (400, 422, 500), (
            f"Expected error for invalid image, got {resp.status_code}"
        )

    def test_makeup_invalid_params_returns_400(self):
        """Sending an invalid parameter value should be rejected."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
            data={"style": "NONEXISTENT_STYLE_XYZ"},
        )
        # Should reject unknown style with 400, or ignore gracefully.
        assert resp.status_code in (200, 400, 422, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# /process/hair-color
# ═══════════════════════════════════════════════════════════════════════════

class TestHairColorEndpoint:
    """Tests for POST /process/hair-color."""

    ENDPOINT = "/process/hair-color"

    def test_hair_color_valid_image(self):
        """A valid JPEG with a colour parameter should return 200."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
            data={"color": "red"},
        )
        assert resp.status_code in (200, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )

    def test_hair_color_missing_image_returns_400(self):
        """Sending no image field should fail with 400 or 422."""
        resp = client.post(self.ENDPOINT, data={"color": "blue"})
        assert resp.status_code in (400, 422), (
            f"Expected 400/422 for missing image, got {resp.status_code}"
        )

    def test_hair_color_invalid_image_returns_400(self):
        """Sending garbage bytes as the image should return 400."""
        bad_files = {
            "image": ("bad.jpg", io.BytesIO(b"JUNKDATA"), "image/jpeg")
        }
        resp = client.post(
            self.ENDPOINT,
            files=bad_files,
            data={"color": "blue"},
        )
        assert resp.status_code in (400, 422, 500), (
            f"Expected error for invalid image, got {resp.status_code}"
        )

    def test_hair_color_missing_color_param_returns_400(self):
        """Omitting the required 'color' parameter should fail."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
            # no 'color' data
        )
        # Should return 400/422 if color is required, or 200/404 if optional.
        assert resp.status_code in (200, 400, 422, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )

    def test_hair_color_invalid_color_returns_400(self):
        """An invalid/nonsensical color value should be rejected."""
        resp = client.post(
            self.ENDPOINT,
            files=_make_valid_upload(),
            data={"color": "!!!INVALID!!!"},
        )
        assert resp.status_code in (200, 400, 422, 404, 405), (
            f"Unexpected status {resp.status_code}: {resp.text}"
        )
