import requests
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite('test.png', img)

url = 'http://127.0.0.1:8000/process/makeup'
files = {'image': open('test.png', 'rb')}
data = {'region': 'lips', 'hue': 10, 'opacity': 0.5}

try:
    response = requests.post(url, files=files, data=data)
    print("Status:", response.status_code)
    print("Response:", response.text[:200])
except Exception as e:
    print("Error:", e)
