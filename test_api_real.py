import requests

url = 'http://127.0.0.1:8000/process/makeup'
files = {'image': open('test_lips.png', 'rb')}
data = {'region': 'lips', 'hue': '10', 'opacity': '0.5'}

response = requests.post(url, files=files, data=data)
print("Status:", response.status_code)
print("Headers:", response.headers)
