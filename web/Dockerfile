# ---- Stage 1: build Vite frontend (same-origin: empty VITE_API_BASE_URL) ----
FROM node:22-slim AS frontend-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---- Stage 2: slim Python runtime for FastAPI + ONNX (CPU) ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=models/adc_student_full.onnx \
    STATIC_DIR=/app/static \
    CORS_ORIGINS=*

# onnxruntime needs libgomp1; curl is for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code only — model (*.onnx, 126MB) is NOT baked so the image stays
# under Artifact Registry free tier. Inject at runtime via:
#   local:  volume ./backend/models -> /app/backend/models (see compose)
#   GCP:    copy once with `gcloud storage cp` + mount, or bake with --build-arg in CI
COPY backend/app ./app
COPY backend/pytest.ini ./
RUN mkdir -p models static

# Vite build output served by FastAPI StaticFiles (see app/main.py)
COPY --from=frontend-build /web/dist /app/static

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT:-8000}/api/health || exit 1

# Single worker: Cloud Run scales with instances, not workers.
# ${PORT} is required by Cloud Run (8080); defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers"]
