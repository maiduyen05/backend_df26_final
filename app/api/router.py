from fastapi import APIRouter

from app.api import batches

router = APIRouter(prefix="/api")
router.include_router(batches.router)
