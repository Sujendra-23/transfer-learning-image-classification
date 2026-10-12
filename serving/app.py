"""FastAPI service: `POST /predict` (image upload) and `GET /health`.

Configuration is by environment variable so the same image runs any of the four backbones:

    MODEL_PATH        path to the exported .keras file (required)
    BACKBONE          ResNet50 | ResNet101 | EfficientNetB0 | VGG16 (default ResNet50)
    MAX_UPLOAD_BYTES  largest accepted upload (default 8000000)

    uvicorn serving.app:app --host 0.0.0.0 --port 8000

Run a single uvicorn process per container: the model lives in that process's memory and
predictions are serialized, so extra workers in one container only multiply memory use.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from serving.inference import ImageError, Predictor  # noqa: E402

DEFAULT_MAX_UPLOAD_BYTES = 8_000_000


def load_predictor(model_path: str, backbone: str) -> Predictor:
    """Load an exported model through scripts/load_model.py (registers MatchedCrop)."""
    from load_model import CLASS_NAMES, PREPROCESS, load_final_model

    if backbone not in PREPROCESS:
        raise RuntimeError(f"BACKBONE must be one of {list(PREPROCESS)}, got {backbone!r}")
    if not Path(model_path).is_file():
        raise RuntimeError(f"MODEL_PATH {model_path!r} does not exist")
    return Predictor(load_final_model(backbone, model_path), list(CLASS_NAMES), backbone)


def create_app(
    predictor: Predictor | None = None,
    *,
    model_path: str | None = None,
    backbone: str | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    """Build the app. Pass ``predictor`` to skip loading (tests); otherwise it is loaded at startup."""
    limit = max_upload_bytes or int(os.environ.get("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if predictor is not None:
            app.state.predictor = predictor
        else:
            path = model_path or os.environ.get("MODEL_PATH")
            if not path:
                raise RuntimeError("Set MODEL_PATH to the exported .keras model")
            app.state.predictor = await run_in_threadpool(
                load_predictor, path, backbone or os.environ.get("BACKBONE", "ResNet50")
            )
        yield

    app = FastAPI(
        title="Transfer-learning image classifier",
        description="Serves one exported backbone (classes C1-C5).",
        lifespan=lifespan,
    )
    app.state.predictor = None

    @app.get("/health")
    async def health() -> dict:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {
            "status": "ok",
            "backbone": loaded.backbone,
            "classes": loaded.class_names,
            "input_size": list(loaded.input_size),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(loaded.classify, data)
        except ImageError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    return app


app = create_app()
