from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from app.schemas.responses import HealthResponse, PredictionResponse

router = APIRouter()

PREDICT_REQUESTS = Counter("agricnxedge_predict_requests_total", "Total prediction requests", ["outcome"])
PREDICT_LATENCY = Histogram("agricnxedge_predict_latency_seconds", "Prediction latency in seconds")


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    service = request.app.state.inference
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": settings.app_name,
        "model_loaded": service.loaded,
    }


@router.get("/model/status")
def model_status(request: Request):
    return request.app.state.inference.status()


@router.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    import time

    settings = request.app.state.settings
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        PREDICT_REQUESTS.labels(outcome="rejected").inc()
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Upload a JPEG, PNG, or WEBP image.")

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        PREDICT_REQUESTS.labels(outcome="rejected").inc()
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Image must be {settings.max_upload_mb} MB or smaller.")

    started = time.perf_counter()
    try:
        result = request.app.state.inference.predict(content, file.filename or "upload")
        PREDICT_REQUESTS.labels(outcome="ok").inc()
        return result
    except RuntimeError as exc:
        PREDICT_REQUESTS.labels(outcome="error").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    finally:
        PREDICT_LATENCY.observe(time.perf_counter() - started)
