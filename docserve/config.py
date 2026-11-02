"""Runtime configuration, read once from DOCSERVE_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"DOCSERVE_{name}", default)


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Model registry: <model_dir>/<model_name>/<version>/{model.onnx, model.int8.onnx, metadata.json}
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", str(ROOT / "models"))))
    model_name: str = field(default_factory=lambda: _env("MODEL_NAME", "doc_classifier"))
    default_version: str = field(default_factory=lambda: _env("MODEL_VERSION", "latest"))
    model_variant: str = field(default_factory=lambda: _env("MODEL_VARIANT", "fp32"))  # fp32 | int8
    ort_device: str = field(default_factory=lambda: _env("ORT_DEVICE", "auto"))  # auto | cpu | cuda
    ort_intra_threads: int = field(default_factory=lambda: _int("ORT_INTRA_THREADS", 0))  # 0 = ORT default

    # OCR
    ocr_engine: str = field(default_factory=lambda: _env("OCR_ENGINE", "paddle_onnx"))  # paddle_onnx | paddle | tesseract
    preload_ocr: bool = field(default_factory=lambda: _bool("PRELOAD_OCR", True))

    # Request validation
    max_upload_mb: float = field(default_factory=lambda: _float("MAX_UPLOAD_MB", 10))
    max_files_per_request: int = field(default_factory=lambda: _int("MAX_FILES_PER_REQUEST", 16))
    max_image_pixels: int = field(default_factory=lambda: _int("MAX_IMAGE_PIXELS", 40_000_000))
    min_image_side: int = field(default_factory=lambda: _int("MIN_IMAGE_SIDE", 64))

    # Dynamic batching for the classifier
    batch_max_size: int = field(default_factory=lambda: _int("BATCH_MAX_SIZE", 16))
    batch_max_wait_ms: float = field(default_factory=lambda: _float("BATCH_MAX_WAIT_MS", 8))

    # Bounded concurrency / load shedding
    max_concurrency: int = field(default_factory=lambda: _int("MAX_CONCURRENCY", 4))
    max_pending: int = field(default_factory=lambda: _int("MAX_PENDING", 64))

    # Redis (result cache + async job queue). Empty URL -> in-memory cache, job API disabled.
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    cache_enabled: bool = field(default_factory=lambda: _bool("CACHE_ENABLED", True))
    cache_ttl_s: int = field(default_factory=lambda: _int("CACHE_TTL_S", 3600))
    job_ttl_s: int = field(default_factory=lambda: _int("JOB_TTL_S", 86400))
    worker_batch_size: int = field(default_factory=lambda: _int("WORKER_BATCH_SIZE", 8))

    # Document storage: local | s3
    storage_backend: str = field(default_factory=lambda: _env("STORAGE_BACKEND", "local"))
    storage_dir: Path = field(default_factory=lambda: Path(_env("STORAGE_DIR", str(ROOT / "storage"))))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env("S3_PREFIX", "docserve"))
    store_uploads: bool = field(default_factory=lambda: _bool("STORE_UPLOADS", False))

    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
