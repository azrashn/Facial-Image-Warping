import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.ai_router import router as ai_router

try:
    from routers.export import router as export_router
    from routers.metrics import router as metrics_router
    from routers.process import router as process_router
    from routers.upload import router as upload_router
except ModuleNotFoundError:
    from routers.export import router as export_router
    from routers.metrics import router as metrics_router
    from routers.process import router as process_router
    from routers.upload import router as upload_router


app = FastAPI(title="Facial Warping API - Group 14")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

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


app.include_router(upload_router)
app.include_router(process_router)
app.include_router(metrics_router)
app.include_router(export_router)
app.include_router(ai_router)

