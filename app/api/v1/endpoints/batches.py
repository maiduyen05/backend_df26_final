"""
batches.py
Endpoints:
  POST /api/v1/batches/upload      → UploadResponse
  GET  /api/v1/batches/{id}/overview → OverviewResponse
"""

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.schemas import OverviewResponse, UploadResponse
from app.services.batch_service import get_overview, process_upload

router = APIRouter(prefix="/batches", tags=["batches"])

# ── Kích thước tối đa cho phép: 50 MB ─────────────────────────────────────
MAX_FILE_SIZE = 50 * 1024 * 1024  # bytes


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload CSV dataset",
    description=(
        "Nhận 1 file CSV có cột `id` + `feature_1..66`. "
        "Backend parse, chạy model, lưu kết quả và trả về `batch_id`."
    ),
)
async def upload_dataset(
    file: UploadFile = File(..., description="File CSV chứa id + feature_1..feature_66"),
) -> UploadResponse:
    # Validate content type
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # Vẫn cho qua nếu content_type không chuẩn, kiểm tra extension
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
    summary="Lấy dữ liệu Overview page",
    description=(
        "Trả về toàn bộ dữ liệu cần thiết cho trang Overview "
        "(KPI cards, charts, segments, profile) trong 1 request."
    ),
)
async def get_batch_overview(batch_id: str) -> OverviewResponse:
    return get_overview(batch_id)
