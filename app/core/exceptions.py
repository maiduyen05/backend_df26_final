from fastapi import HTTPException, status


class CSVValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV không hợp lệ: {detail}",
        )


class BatchNotFoundError(HTTPException):
    def __init__(self, batch_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' không tồn tại.",
        )


class ModelNotLoadedError(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model chưa được load. Kiểm tra MODEL_PATH trong .env.",
        )
