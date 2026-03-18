# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Tách torch ra khỏi requirements để cài CPU-only ───────────────────────
# Bước 1: cài torch CPU-only trước (~200MB thay vì ~800MB bản GPU)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Bước 2: cài phần còn lại (đã bỏ torch trong requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────
COPY app/ ./app/
COPY ml/  ./ml/

# ── Environment variables ──────────────────────────────────────────────────
# HuggingFace Spaces bắt buộc port 7860
ENV APP_ENV=production
ENV APP_HOST=0.0.0.0
ENV APP_PORT=7860
ENV MODEL_PATH=ml/model.pt
ENV ALLOWED_ORIGINS=*
ENV PLANT_SIGNAL_A_COL=attr_3
ENV PLANT_SIGNAL_B_COL=attr_6
ENV PLANT_SIGNAL_MAX=99

# ── Port ───────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Start ──────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
