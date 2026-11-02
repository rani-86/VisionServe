# Serving image: ONNX Runtime + PaddleOCR + Tesseract. PyTorch is not installed;
# models are trained/exported outside the image and served from ONNX.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libglib2.0-0 libgomp1 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Bake the PP-OCRv5 weights (Paddle + ONNX) into the image so containers start without network access.
COPY docserve/ docserve/
RUN python -c "from docserve.ocr import get_engine; [get_engine(e) for e in ('paddle_onnx', 'paddle', 'tesseract')]"

COPY models/ models/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/storage && chown -R app /app /opt/paddlex
USER app

ENV DOCSERVE_MODEL_DIR=/app/models \
    DOCSERVE_STORAGE_DIR=/app/storage
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8000/health/live || exit 1

# One process per container: each process owns its micro-batcher and ONNX sessions.
# Scale horizontally (compose replicas / ECS tasks) instead of uvicorn --workers.
CMD ["uvicorn", "docserve.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
