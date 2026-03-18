"""
main.py
Entry point của FastAPI application.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import settings
from app.services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HiddenLayer Backend",
    description="API backend cho hệ thống vận hành kho bãi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Khởi động HiddenLayer Backend (%s)", settings.APP_ENV)
    model_service.load(settings.MODEL_PATH)


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "env": settings.APP_ENV}


# ── Routes ─────────────────────────────────────────────────────────────────

app.include_router(api_router)
