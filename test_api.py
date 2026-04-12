"""Quick API smoke test using only stdlib."""
import urllib.request
import json
import os

# Build multipart form data manually
boundary = "----TestBoundary123456"
file_path = os.path.join("backend", "test_face.jpg")

with open(file_path, "rb") as f:
    file_data = f.read()

body_parts = []
# File field
body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
body_parts.append(file_data)
body_parts.append(b"\r\n")
# Operation field
body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"operation\"\r\n\r\naging\r\n".encode())
# Intensity field
body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"intensity\"\r\n\r\n50\r\n".encode())
# End
body_parts.append(f"--{boundary}--\r\n".encode())

body = b"".join(body_parts)

req = urllib.request.Request(
    "http://127.0.0.1:8000/apply_transformation",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read())

print(f"Status: {resp.status}")
print(f"Keys: {list(data.keys())}")
has_img = "processed_image" in data and len(data["processed_image"]) > 100
has_fft = "fft_spectrum" in data and len(data["fft_spectrum"]) > 100
lm_count = len(data.get("landmarks", []))
metrics = data.get("metrics", {})

print(f"Has processed image: {has_img}")
print(f"Has FFT spectrum: {has_fft}")
print(f"Landmarks count: {lm_count}")
print(f"Metrics: {json.dumps(metrics, indent=2)}")
print(f"Operation: {data.get('operation')}")

if has_img and has_fft:
    print("\n=== API TEST PASSED ===")
else:
    print("\n=== API TEST FAILED ===")
