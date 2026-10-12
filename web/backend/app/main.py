from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
from app.services.inference import InferenceService


def create_app() -> FastAPI:
    import os
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0")
    app.state.settings = settings
    app.state.inference = InferenceService(settings.model_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router, prefix="/api")

    # Single-container production: serve Vite build from FastAPI when present.
    # Docker multi-stage build copies frontend/dist -> /app/static.
    # Local dev (npm run dev) is unaffected when the directory is absent.
    static_dir = Path(os.environ.get("STATIC_DIR", "static"))
    candidates = [static_dir, Path(__file__).resolve().parents[2] / "static", Path("/app/static")]
    dist = next((p for p in candidates if (p / "index.html").exists()), None)
    if dist is not None:
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_root():
            return FileResponse(dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str):
            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
