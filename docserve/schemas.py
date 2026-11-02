"""Pydantic response models (also drive the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelRef(BaseModel):
    name: str
    version: str
    variant: str


class Preprocessing(BaseModel):
    steps: list[str]
    skew_angle: float
    perspective_corrected: bool
    original_size: list[int]
    processed_size: list[int]


class OCRLineOut(BaseModel):
    text: str
    confidence: float
    bbox: list[int]


class OCRInfo(BaseModel):
    engine: str
    line_count: int
    mean_confidence: float
    lines: list[OCRLineOut] | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str


class DocumentResult(BaseModel):
    filename: str | None = None
    document_id: str | None = None
    model: ModelRef | None = None
    doc_type: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    fields: dict[str, str | None] | None = None
    missing_fields: list[str] | None = None
    expected_fields: list[str] | None = None
    ocr: OCRInfo | None = None
    text: str | None = None
    preprocessing: Preprocessing | None = None
    timings_ms: dict[str, float] | None = None
    cached: bool | None = None
    error: ErrorInfo | None = None


class ClassifyResponse(BaseModel):
    results: list[DocumentResult]


class ModelInfo(BaseModel):
    name: str
    version: str
    variant: str
    provider: str
    labels: list[str]
    input_size: int
    base_model: str | None = None
    trained_at: str | None = None
    metrics: dict[str, Any] = {}
    default: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_version: str


class JobAccepted(BaseModel):
    job_id: str
    status: str
    status_url: str


class JobStatus(BaseModel):
    id: str
    status: str  # queued | processing | done | failed
    params: dict[str, Any] | None = None
    created_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    result: DocumentResult | None = None
    error: str | None = None
