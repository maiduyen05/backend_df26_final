from fastapi import APIRouter

from app.api.v1.endpoints import batches

router = APIRouter(prefix="/api/v1")
router.include_router(batches.router)
