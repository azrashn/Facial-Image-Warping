import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from modules.frequency_module import apply_aging_filter

client = TestClient(app)


def _make_dummy_image(w=400, h=300):
    # Plain BGR image
    return np.ones((h, w, 3), dtype=np.uint8) * 128


def _make_dummy_landmarks():
    # 468 mock landmarks (just random coordinates)
    landmarks = np.zeros((468, 2), dtype=np.float32)
    # Give some reasonable values to prevent divide by zero
    landmarks[:, 0] = np.linspace(100, 300, 468)
    landmarks[:, 1] = np.linspace(50, 250, 468)
    return landmarks


def _make_valid_upload(img):
    success, buf = cv2.imencode(".jpg", img)
    assert success
    return {"image": ("face.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}


def test_apply_aging_filter_success():
    img = _make_dummy_image()
    landmarks = _make_dummy_landmarks()
    
    # Test with and without landmarks, and different intensities
    for intensity in [0.0, 0.5, 1.0]:
        res_with = apply_aging_filter(img, intensity=intensity, landmarks=landmarks)
        assert res_with.shape == img.shape
        assert res_with.dtype == np.uint8
        
        res_without = apply_aging_filter(img, intensity=intensity, landmarks=None)
        assert res_without.shape == img.shape
        assert res_without.dtype == np.uint8


def test_api_process_age_success():
    img = _make_dummy_image()
    files = _make_valid_upload(img)
    
    # Test POST /process/age
    response = client.post(
        "/process/age",
        files=files,
        data={"operation": "aging", "intensity": 60}
    )
    assert response.status_code == 200
    payload = response.json()
    assert "image_b64" in payload
    assert "metrics" in payload


def test_api_process_apply_aging_success():
    # Test POST /process/apply with unified payload
    img = _make_dummy_image()
    success, buf = cv2.imencode(".jpg", img)
    import base64
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    
    response = client.post(
        "/process/apply",
        json={
            "image_b64": img_b64,
            "filter_name": "aging",
            "intensity": 75.0,
            "skip_spectra": True
        }
    )
    assert response.status_code == 200
    payload = response.json()
    assert "image_b64" in payload
    assert "metrics" in payload
