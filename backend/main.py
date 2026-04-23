from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.process import router as process_router

app = FastAPI(title="Facial Warping API - Group 14")

# CORS Ayarları: Frontend (Tarayıcı) ile Backend'in konuşabilmesi için zorunludur!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_router)


@app.get("/")
def read_root():
    return {"message": "Backend API Sistemimiz Aktif!"}