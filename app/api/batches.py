"""
batches.py
Endpoints:
  POST /api/batches/upload          → UploadResponse
  GET  /api/batches/{id}/overview   → OverviewResponse
  GET  /api/batches/{id}/orders     → OrdersResponse
"""

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.schemas import OrdersResponse, OverviewResponse, UploadResponse
from app.services.batch_service import get_orders, get_overview, process_upload

router = APIRouter(prefix="/batches", tags=["batches"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload CSV dataset",
)
async def upload_dataset(
    file: UploadFile = File(..., description="File CSV: id + feature_1..feature_66"),
) -> UploadResponse:
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".csv"):
            return JSONResponse(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                content={"detail": "Chỉ nhận file CSV (.csv)."},
            )
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": f"File quá lớn. Tối đa {MAX_FILE_SIZE // 1024 // 1024} MB."},
        )
    if len(content) == 0:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "File rỗng."},
        )
    return process_upload(content)


@router.get(
    "/{batch_id}/overview",
    response_model=OverviewResponse,
    summary="Dữ liệu Overview page",
)
async def get_batch_overview(batch_id: str) -> OverviewResponse:
    return get_overview(batch_id)


@router.get(
    "/{batch_id}/orders",
    response_model=OrdersResponse,
    summary="Danh sách đơn hàng — Workbench & Warehouse page",
)
async def get_batch_orders(batch_id: str) -> OrdersResponse:
    return get_orders(batch_id)
