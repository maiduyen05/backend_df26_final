# HiddenLayer Backend

FastAPI backend cho hệ thống vận hành kho bãi.

## Cấu trúc thư mục

```
backend/
├── app/
│   ├── main.py                      # FastAPI app, CORS, startup
│   ├── core/
│   │   ├── config.py                # Settings từ .env
│   │   └── exceptions.py           # Custom HTTP exceptions
│   ├── api/v1/
│   │   ├── router.py                # Gom routers
│   │   └── endpoints/
│   │       └── batches.py          # POST /upload  GET /{id}/overview
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response schemas
│   ├── services/
│   │   ├── csv_parser.py           # Parse + validate CSV
│   │   ├── model_service.py        # Transformer inference wrapper
│   │   ├── metrics.py              # Tính entropy, risk, decision state
│   │   └── batch_service.py        # Orchestration pipeline
│   └── storage/
│       └── in_memory.py            # Lưu batch trong RAM
├── ml/
│   └── README.md                   # Hướng dẫn đặt model.pt
├── requirements.txt
├── .env.example
└── README.md
```

## Cài đặt

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu hình

```bash
cp .env.example .env
# Chỉnh sửa .env theo môi trường của bạn
```

## Chạy server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Sau khi chạy:
- API docs: http://localhost:8000/docs
- Health:   http://localhost:8000/health

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|-------|
| `POST` | `/api/v1/batches/upload` | Upload CSV, chạy model, trả `batch_id` |
| `GET`  | `/api/v1/batches/{batch_id}/overview` | Lấy toàn bộ dữ liệu Overview page |
| `GET`  | `/health` | Health check |

## Mock Mode

Khi chưa có `ml/model.pt`, backend tự động chạy **MOCK MODE** —
sinh dữ liệu giả để dev/test frontend mà không cần model thật.

## Test nhanh với curl

```bash
# Upload
curl -X POST http://localhost:8000/api/v1/batches/upload \
  -F "file=@sample.csv"

# Overview (thay <batch_id> bằng giá trị trả về ở trên)
curl http://localhost:8000/api/v1/batches/<batch_id>/overview
```
