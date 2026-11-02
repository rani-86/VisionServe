"""FastAPI inference service.

Endpoints
  GET  /health/live              process is up
  GET  /health/ready             models + OCR engine loaded, Redis reachable (if configured)
  GET  /metrics                  Prometheus exposition
  GET  /v1/models                registered model versions and their eval metrics
  POST /v1/classify              1..N images -> document type (dynamically batched ONNX inference)
  POST /v1/documents             1 image -> type + OCR text + key-value fields
  POST /v1/documents/batch       1..N images -> full results, per-file errors
  POST /v1/jobs                  enqueue a document on the Redis queue (202), processed by worker.py
  GET  /v1/jobs/{job_id}         job status / result
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .batcher import MicroBatcher
from .config import Settings, get_settings
from .logging_utils import configure_logging, request_id_var
from .metrics import (CACHE_EVENTS, FAILURES, HTTP_LATENCY, HTTP_REQUESTS, INFLIGHT, JOB_EVENTS,
                      JOB_QUEUE_DEPTH, REJECTED, STAGE_LATENCY)
from .ocr import ENGINES, get_engine
from .pipeline import assemble, classification, doc_id, prepare, run_ocr_and_extract
from .preprocessing import ImageDecodeError
from .registry import LoadedModel, ModelNotFound, ModelRegistry
from .schemas import (ClassifyResponse, DocumentResult, JobAccepted, JobStatus, ModelsResponse)
from .store import JobQueue, MemoryCache, RedisCache, make_storage, redis_client

log = logging.getLogger("docserve.api")

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp", "image/webp",
                 "application/octet-stream"}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


class Admission:
    """Bounded concurrency: at most `max_pending` documents admitted (else 503), and at most
    `max_concurrency` running CPU-heavy stages (preprocess / OCR) at the same time."""

    def __init__(self, max_pending: int, max_concurrency: int):
        self.max_pending = max_pending
        self.pending = 0
        self.cpu = asyncio.Semaphore(max_concurrency)

    def __enter__(self):
        if self.pending >= self.max_pending:
            REJECTED.labels("overloaded").inc()
            raise APIError(503, "overloaded", "server is at capacity, retry shortly", {"Retry-After": "1"})
        self.pending += 1
        INFLIGHT.inc()
        return self

    def __exit__(self, *exc):
        self.pending -= 1
        INFLIGHT.dec()


class State:
    settings: Settings
    registry: ModelRegistry
    batchers: dict[str, MicroBatcher]
    admission: Admission
    cache: MemoryCache | RedisCache
    jobs: JobQueue | None
    storage: object
    ocr_ready: bool = False
    started_at: float


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    state = State()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.settings = settings
        state.started_at = time.time()
        state.registry = ModelRegistry(settings.model_dir, settings.model_name, settings.model_variant,
                                       settings.ort_device, settings.ort_intra_threads).load()
        state.batchers = {v: MicroBatcher(m, settings.batch_max_size, settings.batch_max_wait_ms)
                          for v, m in state.registry.models.items()}
        for b in state.batchers.values():
            b.start()
        state.admission = Admission(settings.max_pending, settings.max_concurrency)
        if settings.redis_url:
            client = redis_client(settings.redis_url)
            state.cache, state.jobs = RedisCache(client), JobQueue(client, settings.job_ttl_s)
        else:
            state.cache, state.jobs = MemoryCache(), None
        state.storage = make_storage(settings)
        if settings.preload_ocr:
            try:
                await asyncio.to_thread(get_engine, settings.ocr_engine)
                state.ocr_ready = True
            except Exception:
                log.exception("OCR engine failed to load", extra={"engine": settings.ocr_engine})
        log.info("service ready", extra={"versions": sorted(state.registry.models), "ocr_engine": settings.ocr_engine,
                                         "redis": bool(settings.redis_url)})
        yield
        for b in state.batchers.values():
            await b.stop()

    app = FastAPI(title="DocServe", version=__version__, lifespan=lifespan,
                  description="OCR & Document AI inference platform")
    app.state.docserve = state

    # ---- middleware / errors ------------------------------------------------
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            dt = time.perf_counter() - t0
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if route != "/metrics":
                HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                HTTP_LATENCY.labels(route).observe(dt)
                log.info("request", extra={"method": request.method, "route": route, "status": status,
                                           "latency_ms": round(dt * 1000, 2)})
            request_id_var.reset(token)

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError):
        if exc.status in (400, 413, 415, 422):
            REJECTED.labels("validation").inc()
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}, "request_id": request_id_var.get()},
                            status_code=exc.status, headers=exc.headers)

    # ---- helpers -------------------------------------------------------------
    def resolve_model(version: str | None) -> LoadedModel:
        try:
            return state.registry.resolve(version)
        except ModelNotFound as e:
            raise APIError(404, "model_not_found", str(e.args[0]))

    def check_engine(name: str | None) -> str:
        name = name or settings.ocr_engine
        if name not in ENGINES:
            raise APIError(422, "invalid_ocr_engine", f"ocr_engine must be one of {sorted(ENGINES)}")
        return name

    async def read_upload(f: UploadFile) -> bytes:
        ctype = (f.content_type or "application/octet-stream").split(";")[0].lower()
        if ctype not in ALLOWED_TYPES:
            raise APIError(415, "unsupported_media_type", f"{f.filename}: content type {ctype} is not an image")
        data = await f.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise APIError(413, "file_too_large", f"{f.filename}: exceeds {settings.max_upload_mb} MB")
        if not data:
            raise APIError(422, "empty_file", f"{f.filename}: file is empty")
        return data

    def check_count(files: list[UploadFile]) -> None:
        if not files:
            raise APIError(422, "no_files", "at least one file is required")
        if len(files) > settings.max_files_per_request:
            raise APIError(413, "too_many_files", f"at most {settings.max_files_per_request} files per request")

    async def process(data: bytes, model: LoadedModel, *, full: bool, engine: str, include_lines: bool = False,
                      preprocess_enabled: bool = True) -> dict:
        key = f"{doc_id(data)}:{model.version}:{model.variant}:{'full' if full else 'cls'}:{engine if full else '-'}" \
              f":{int(include_lines)}:{int(preprocess_enabled)}"
        if settings.cache_enabled:
            cached = await asyncio.to_thread(state.cache.get, key)
            if cached:
                CACHE_EVENTS.labels("hit").inc()
                return cached | {"cached": True}
            CACHE_EVENTS.labels("miss").inc()
        with state.admission:
            try:
                async with state.admission.cpu:
                    p = await asyncio.to_thread(prepare, data, model, max_pixels=settings.max_image_pixels,
                                                min_side=settings.min_image_side, preprocess_enabled=preprocess_enabled)
            except ImageDecodeError as e:
                raise APIError(422, "invalid_image", str(e))
            t0 = time.perf_counter()
            probs = await state.batchers[model.version].submit(p.tensor)
            dt = time.perf_counter() - t0
            STAGE_LATENCY.labels("classify").observe(dt)
            p.timings.ms["classify"] = round(dt * 1000, 2)
            cls = classification(model, probs)
            ocr = None
            if full:
                try:
                    async with state.admission.cpu:
                        ocr = await asyncio.to_thread(run_ocr_and_extract, p, cls["doc_type"], engine, include_lines)
                except Exception as e:
                    FAILURES.labels("ocr", type(e).__name__).inc()
                    log.exception("OCR failed", extra={"engine": engine})
                    raise APIError(503, "ocr_unavailable", f"OCR engine {engine!r} failed: {e}")
        result = assemble(p, model, cls, ocr)
        if settings.cache_enabled:
            await asyncio.to_thread(state.cache.set, key, result, settings.cache_ttl_s)
        if settings.store_uploads:
            day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            await asyncio.to_thread(state.storage.put, f"uploads/{day}/{p.document_id}", data)
        return result | {"cached": False}

    async def process_many(files: list[UploadFile], model: LoadedModel, **kw) -> list[dict]:
        async def one(f: UploadFile) -> dict:
            try:
                return {"filename": f.filename, **await process(await read_upload(f), model, **kw)}
            except APIError as e:
                if e.status in (400, 413, 415, 422):
                    REJECTED.labels("validation").inc()
                return {"filename": f.filename, "error": {"code": e.code, "message": e.message}}
        return list(await asyncio.gather(*(one(f) for f in files)))

    # ---- routes ----------------------------------------------------------------
    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready():
        checks = {"models": bool(state.registry.models), "ocr": state.ocr_ready}
        if state.jobs is not None:
            try:
                checks["redis"] = await asyncio.to_thread(state.cache.ping)
            except Exception:
                checks["redis"] = False
        ok = all(checks.values())
        body = {"status": "ready" if ok else "not_ready", "checks": checks, "inflight": state.admission.pending,
                "uptime_s": round(time.time() - state.started_at, 1)}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        if state.jobs is not None:
            try:
                JOB_QUEUE_DEPTH.set(await asyncio.to_thread(state.jobs.depth))
            except Exception:
                pass
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models", response_model=ModelsResponse, tags=["models"])
    async def models():
        return {"models": state.registry.describe(), "default_version": state.registry.latest}

    @app.post("/v1/classify", response_model=ClassifyResponse, response_model_exclude_unset=True, tags=["inference"])
    async def classify(files: Annotated[list[UploadFile], File(description="One or more document images")],
                       model_version: Annotated[str | None, Query(max_length=32)] = None):
        check_count(files)
        model = resolve_model(model_version)
        return {"results": await process_many(files, model, full=False, engine="-")}

    @app.post("/v1/documents", response_model=DocumentResult, response_model_exclude_unset=True, tags=["inference"])
    async def documents(file: Annotated[UploadFile, File(description="Document image")],
                        model_version: Annotated[str | None, Query(max_length=32)] = None,
                        ocr_engine: Annotated[str | None, Query()] = None,
                        include_lines: bool = False,
                        preprocess: bool = True):
        model, engine = resolve_model(model_version), check_engine(ocr_engine)
        data = await read_upload(file)
        return await process(data, model, full=True, engine=engine, include_lines=include_lines,
                             preprocess_enabled=preprocess)

    @app.post("/v1/documents/batch", response_model=ClassifyResponse, response_model_exclude_unset=True, tags=["inference"])
    async def documents_batch(files: Annotated[list[UploadFile], File()],
                              model_version: Annotated[str | None, Query(max_length=32)] = None,
                              ocr_engine: Annotated[str | None, Query()] = None):
        check_count(files)
        model, engine = resolve_model(model_version), check_engine(ocr_engine)
        return {"results": await process_many(files, model, full=True, engine=engine)}

    @app.post("/v1/jobs", response_model=JobAccepted, status_code=202, tags=["jobs"])
    async def submit_job(file: Annotated[UploadFile, File()],
                         model_version: Annotated[str | None, Query(max_length=32)] = None,
                         ocr_engine: Annotated[str | None, Query()] = None):
        if state.jobs is None:
            raise APIError(503, "jobs_disabled", "async jobs need DOCSERVE_REDIS_URL")
        model, engine = resolve_model(model_version), check_engine(ocr_engine)
        data = await read_upload(file)
        job_id = await asyncio.to_thread(state.jobs.enqueue, data,
                                         {"model_version": model.version, "ocr_engine": engine,
                                          "filename": file.filename})
        JOB_EVENTS.labels("queued").inc()
        return {"job_id": job_id, "status": "queued", "status_url": f"/v1/jobs/{job_id}"}

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus, response_model_exclude_unset=True, tags=["jobs"])
    async def job_status(job_id: str):
        if state.jobs is None:
            raise APIError(503, "jobs_disabled", "async jobs need DOCSERVE_REDIS_URL")
        job = await asyncio.to_thread(state.jobs.get, job_id)
        if not job:
            raise APIError(404, "job_not_found", f"no job {job_id}")
        return job

    return app

