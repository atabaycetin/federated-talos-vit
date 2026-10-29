"""FastAPI service: /predict, /health, /tasks, /drift, /metrics.

Configured by environment variables:
    MODEL_STORE   directory of task bundles             (default: model_store)
    BACKEND       onnx | taskvector                     (default: onnx)
    DEFAULT_TASK  task used when ?task= is omitted      (default: first task)
    THREADS       inference threads, 0 = runtime default
    DRIFT_WINDOW  requests kept for the drift check     (default: 500)
    API_KEY       if set, required as the X-API-Key header on everything but /health and the docs
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, Security, UploadFile
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from talos_mlops import __version__
from talos_mlops.drift import PSI_ALERT, DriftMonitor, drift_features
from talos_mlops.preprocessing import InvalidImageError, decode_image, preprocess, softmax
from talos_mlops.serving.metrics import ServiceMetrics
from talos_mlops.serving.predictors import OnnxPredictor, Predictor

MAX_UPLOAD_BYTES = 10 * 2**20
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 2**10  # image plus multipart framing
DRIFT_UPDATE_EVERY = 20
PROTECTED_PATHS = frozenset({"/predict", "/tasks", "/drift", "/metrics"})
# Declares the key in the OpenAPI schema so /docs gets an "Authorize" button; ApiKeyAuth enforces it.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
log = logging.getLogger("uvicorn.error")


@dataclass
class Settings:
    model_store: Path = field(default_factory=lambda: Path(os.getenv("MODEL_STORE", "model_store")))
    backend: str = field(default_factory=lambda: os.getenv("BACKEND", "onnx"))
    default_task: str | None = field(default_factory=lambda: os.getenv("DEFAULT_TASK") or None)
    threads: int = field(default_factory=lambda: int(os.getenv("THREADS", "0")))
    drift_window: int = field(default_factory=lambda: int(os.getenv("DRIFT_WINDOW", "500")))
    api_key: str | None = field(default_factory=lambda: os.getenv("API_KEY") or None)


class BodySizeLimit:
    """Reject oversized POST bodies by their Content-Length, before any of the body is read.

    FastAPI parses (and spools to disk) the whole multipart body before /predict runs, so the
    endpoint's own size check alone would still let a huge upload fill the container's disk.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST":
            length = dict(scope["headers"]).get(b"content-length")
            if length is None:
                return await _reject(send, 411, "Content-Length header required")
            if not length.isdigit() or int(length) > self.max_bytes:
                return await _reject(send, 413, f"request larger than {self.max_bytes} bytes")
        await self.app(scope, receive, send)


class ApiKeyAuth:
    """Require the X-API-Key header on PROTECTED_PATHS, checked before any of the body is read.

    A FastAPI dependency would run only after the multipart upload had been parsed, so
    unauthenticated clients could still make the service read and spool their uploads.
    """

    def __init__(self, app, key: str):
        self.app = app
        self.key = key.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] in PROTECTED_PATHS:
            sent = dict(scope["headers"]).get(b"x-api-key", b"")
            if not secrets.compare_digest(sent, self.key):
                return await _reject(send, 401, "missing or invalid X-API-Key header")
        await self.app(scope, receive, send)


async def _reject(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class Prediction(BaseModel):
    label: str
    index: int
    probability: float


class PredictResponse(BaseModel):
    task: str
    model_version: str
    predictions: list[Prediction]
    latency_ms: float


def load_predictor(settings: Settings, metrics: ServiceMetrics) -> Predictor:
    if not settings.model_store.is_dir():
        raise FileNotFoundError(f"model store not found: {settings.model_store}")
    if settings.backend == "onnx":
        return OnnxPredictor(settings.model_store, settings.threads)
    if settings.backend == "taskvector":
        from talos_mlops.serving.task_vectors import TaskVectorPredictor  # needs torch

        return TaskVectorPredictor(
            settings.model_store, settings.threads, on_swap=lambda _old, _new, s: metrics.task_swap.observe(s)
        )
    raise ValueError(f"unknown BACKEND {settings.backend!r}")


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings()
    metrics = ServiceMetrics()
    state: dict = {"predictor": predictor, "error": None, "drift": {}, "counter": 0}

    def init(pred: Predictor) -> None:
        state["predictor"] = pred
        for name, info in pred.tasks.items():
            state["drift"][name] = DriftMonitor(info.meta["drift_reference"], window=settings.drift_window)
            metrics.model_info.labels(name, info.version, info.meta.get("method", ""), pred.backend).set(1)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            init(state["predictor"] or load_predictor(settings, metrics))
        except Exception as exc:  # keep serving /health so the failure is visible
            state["error"] = f"{type(exc).__name__}: {exc}"
        if not settings.api_key:
            log.warning("API_KEY is not set: /predict, /tasks, /drift and /metrics are open to anyone")
        yield

    app = FastAPI(title="TaLoS ViT inference", version=__version__, lifespan=lifespan)
    app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)
    if settings.api_key:
        app.add_middleware(ApiKeyAuth, key=settings.api_key)  # added last, so it runs first
    auth = [Security(api_key_header)] if settings.api_key else []

    def ready_predictor() -> Predictor:
        pred = state["predictor"]
        if pred is None or not pred.tasks:
            raise HTTPException(503, detail=state["error"] or "no models loaded")
        return pred

    def resolve_task(pred: Predictor, task: str | None) -> str:
        name = task or settings.default_task or next(iter(pred.tasks))
        if name not in pred.tasks:
            raise HTTPException(404, detail=f"unknown task {name!r}; available: {sorted(pred.tasks)}")
        return name

    @app.get("/health")
    def health(response: Response) -> dict:
        pred = state["predictor"]
        ok = pred is not None and bool(pred.tasks)
        if not ok:
            response.status_code = 503
        return {
            "status": "ok" if ok else "unavailable",
            "backend": pred.backend if pred else settings.backend,
            "tasks": sorted(pred.tasks) if pred else [],
            "error": state["error"],
            "version": __version__,
        }

    @app.get("/tasks", dependencies=auth)
    def tasks() -> list[dict]:
        return [info.summary() for info in ready_predictor().tasks.values()]

    @app.post("/predict", response_model=PredictResponse, dependencies=auth)
    def predict(
        file: UploadFile = File(..., description="image file (JPEG, PNG, ...)"),
        task: str | None = Query(None, description="task name; defaults to DEFAULT_TASK"),
        top_k: int = Query(5, ge=1, le=20),
    ) -> PredictResponse:
        start = time.perf_counter()
        pred = ready_predictor()
        name = resolve_task(pred, task)
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            metrics.requests.labels(name, "too_large").inc()
            raise HTTPException(413, detail=f"image larger than {MAX_UPLOAD_BYTES} bytes")
        try:
            batch = preprocess(decode_image(data))[None]
        except InvalidImageError as exc:
            metrics.requests.labels(name, "invalid_image").inc()
            raise HTTPException(400, detail=str(exc)) from exc

        t0 = time.perf_counter()
        try:
            logits = pred.predict(batch, name)
        except Exception:
            metrics.requests.labels(name, "error").inc()
            raise
        metrics.inference_latency.labels(name).observe(time.perf_counter() - t0)

        probs = softmax(logits.astype(np.float32))
        info = pred.tasks[name]
        order = np.argsort(-probs[0])[:top_k]
        preds = [
            Prediction(label=info.classes[i], index=int(i), probability=float(probs[0, i])) for i in order
        ]
        _observe(name, batch, probs, preds[0])

        latency = time.perf_counter() - start
        metrics.request_latency.labels(name).observe(latency)
        metrics.requests.labels(name, "ok").inc()
        return PredictResponse(
            task=name, model_version=info.version, predictions=preds, latency_ms=latency * 1000
        )

    def _observe(name: str, batch: np.ndarray, probs: np.ndarray, top: Prediction) -> None:
        metrics.confidence.labels(name).observe(top.probability)
        metrics.predicted_class.labels(name, top.label).inc()
        monitor: DriftMonitor = state["drift"][name]
        monitor.update(drift_features(batch, probs))
        state["counter"] += 1
        if state["counter"] % DRIFT_UPDATE_EVERY == 0:
            _publish_drift(name, monitor)

    def _publish_drift(name: str, monitor: DriftMonitor) -> dict | None:
        scores = monitor.scores()
        if scores is not None:
            for feature, value in scores.items():
                metrics.drift_psi.labels(name, feature).set(value)
            metrics.drift_alert.labels(name).set(float(max(scores.values()) > PSI_ALERT))
        return scores

    @app.get("/drift", dependencies=auth)
    def drift() -> dict:
        ready_predictor()
        out = {}
        for name, monitor in state["drift"].items():
            scores = _publish_drift(name, monitor)
            out[name] = {
                "samples": monitor.n_samples,
                "min_samples": monitor.min_samples,
                "psi": scores,
                "alert": bool(scores and max(scores.values()) > PSI_ALERT),
                "threshold": PSI_ALERT,
            }
        return out

    @app.get("/metrics", dependencies=auth)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
