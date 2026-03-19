# 🏭 HiddenLayer Backend

API backend cho hệ thống vận hành kho bãi, tích hợp mô hình AI **Transformer + RoPE** từ DataFlow 2026 để dự đoán và phân tích hành vi vận hành.

---

## ⚙️ Yêu cầu môi trường

| | |
|---|---|
| **Python** | >= 3.11 |
| **Framework** | FastAPI |
| **Server** | Uvicorn |

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1 — Clone repository

```bash
git clone https://github.com/maiduyen05/backend_df26_final.git
cd backend_df26_final
```

### Bước 2 — Tạo môi trường ảo và cài dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Bước 3 — Cấu hình biến môi trường

Sao chép file mẫu và chỉnh sửa theo môi trường của bạn:

```bash
cp .env.example .env
```

### Bước 4 — Chạy server

**Development (auto-reload):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server chạy tại [http://localhost:8000](http://localhost:8000)

Swagger UI tại [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Chạy bằng Docker

```bash
docker build -t hiddenlayer-backend .
docker run -p 8000:8000 hiddenlayer-backend
```

---

## 📡 API Endpoints

### System

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/health` | Kiểm tra trạng thái server |

### Batches

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/batches/upload` | Upload CSV dataset |
| `GET` | `/api/batches/{batch_id}/overview` | Dữ liệu trang Overview |
| `GET` | `/api/batches/{batch_id}/orders` | Danh sách đơn hàng — Workbench & Warehouse page |

> Tài liệu API đầy đủ xem tại `/docs` (Swagger UI) hoặc `/openapi.json` sau khi chạy server.

---

## 📁 Cấu trúc thư mục

```
├── app/                    # Source code chính
├── ml/                     # Model AI (Transformer + RoPE)
├── .env.example            # Mẫu biến môi trường
├── .gitattributes
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```
