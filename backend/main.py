from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from routers.process import router as process_router
from routers.metrics import router as metrics_router
from routers.export import router as export_router

app = FastAPI(title="Facial Warping API - Group 14")

# CORS Ayarları: Frontend (Tarayıcı) ile Backend'in konuşabilmesi için zorunludur!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Measure request latency and expose it in response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(process_router)
app.include_router(metrics_router)
app.include_router(export_router)


@app.get("/")
def read_root():
    return {"message": "Backend API Sistemimiz Aktif!"}