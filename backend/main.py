from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64
import time

app = FastAPI(title="Facial Warping API - Group 14")

# CORS Ayarları: Frontend (Tarayıcı) ile Backend'in konuşabilmesi için zorunludur!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend API Sistemimiz Aktif!"}

# Rol 5'in istek atacağı ana uç nokta (Endpoint)
@app.post("/apply_transformation")
async def apply_transformation(
    file: UploadFile = File(...),
    operation: str = Form("Smile"),
    intensity: int = Form(50)
):
    # 1. Gelen resmi oku (Rol 1 burada devreye girecek)
    image_data = await file.read()

    # 2. İşlem yapıyormuş gibi bekle (Rol 2 ve 3 kodlarını buraya yazacak)
    time.sleep(1) # 1 saniye gecikme simülasyonu

    # 3. Metrikleri hesapla (Senin Görevin: Rol 6)
    # Şimdilik Rol 5 arayüzü çizebilsin diye sahte veriler dönüyoruz.
    dummy_metrics = {
        "mse": 14.57,
        "psnr": 27.74,
        "ssim": 0.885
    }

    # 4. Resmi Frontend'de gösterilebilecek Base64 formatına çevir
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/jpeg;base64,{base64_encoded}"

    # 5. Sonucu arayüze (Frontend) geri yolla
    return {
        "status": "success",
        "processed_image": image_url,
        "metrics": dummy_metrics
    }