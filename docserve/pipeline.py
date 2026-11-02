"""Document processing stages shared by the API and the Redis batch worker.

decode -> preprocess -> classify (batched ONNX) -> OCR -> key-value extraction
"""
from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from .extraction import SPECS, extract_fields
from .metrics import CLASSIFIER_CONFIDENCE, FIELDS_MISSING, OCR_CONFIDENCE, PREDICTIONS, STAGE_LATENCY
from .ocr import get_engine, lines_to_rows
from .preprocessing import PreprocessResult, classifier_tensor, decode_image, preprocess
from .registry import LoadedModel

log = logging.getLogger(__name__)


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]


@dataclass
class Timings:
    ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            STAGE_LATENCY.labels(name).observe(dt)
            self.ms[name] = round(self.ms.get(name, 0) + dt * 1000, 2)


@dataclass
class Prepared:
    document_id: str
    pre: PreprocessResult
    tensor: np.ndarray
    original_size: tuple[int, int]
    timings: Timings


def prepare(data: bytes, model: LoadedModel, *, max_pixels: int, min_side: int,
            preprocess_enabled: bool = True) -> Prepared:
    """CPU-bound: decode + OpenCV preprocessing + classifier input tensor."""
    t = Timings()
    with t.stage("decode"):
        img = decode_image(data, max_pixels=max_pixels, min_side=min_side)
    with t.stage("preprocess"):
        pre = preprocess(img, enabled=preprocess_enabled)
        tensor = classifier_tensor(pre.gray, model.input_size, model.mean, model.std)
    return Prepared(doc_id(data), pre, tensor, (img.shape[1], img.shape[0]), t)


def classification(model: LoadedModel, probs: np.ndarray) -> dict:
    idx = int(np.argmax(probs))
    doc_type, conf = model.labels[idx], float(probs[idx])
    PREDICTIONS.labels(model.version, doc_type).inc()
    CLASSIFIER_CONFIDENCE.observe(conf)
    return {
        "doc_type": doc_type,
        "confidence": round(conf, 4),
        "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(model.labels, probs)},
    }


def run_ocr_and_extract(p: Prepared, doc_type: str, engine_name: str, include_lines: bool = True) -> dict:
    """CPU-bound: OCR on the preprocessed page, then rule-based field extraction."""
    engine = get_engine(engine_name)
    with p.timings.stage("ocr"):
        lines = engine.recognize(p.pre.gray, p.pre.binary)
    with p.timings.stage("extract"):
        rows = lines_to_rows(lines)
        fields = extract_fields(doc_type, rows)
    mean_conf = float(np.mean([ln.confidence for ln in lines])) if lines else 0.0
    OCR_CONFIDENCE.labels(engine.name, doc_type).observe(mean_conf)
    missing = [k for k, v in fields.items() if v is None]
    for k in missing:
        FIELDS_MISSING.labels(doc_type, k).inc()
    out = {
        "fields": fields,
        "missing_fields": missing,
        "expected_fields": list(SPECS.get(doc_type, {})),
        "ocr": {"engine": engine.name, "line_count": len(lines), "mean_confidence": round(mean_conf, 4)},
        "text": "\n".join(rows),
    }
    if include_lines:
        out["ocr"]["lines"] = [{"text": ln.text, "confidence": round(ln.confidence, 4), "bbox": list(ln.bbox)}
                               for ln in lines]
    return out


def assemble(p: Prepared, model: LoadedModel, cls: dict, ocr: dict | None) -> dict:
    result = {
        "document_id": p.document_id,
        "model": {"name": model.name, "version": model.version, "variant": model.variant},
        **cls,
        "preprocessing": {
            "steps": p.pre.steps,
            "skew_angle": round(p.pre.skew_angle, 2),
            "perspective_corrected": p.pre.perspective_corrected,
            "original_size": list(p.original_size),
            "processed_size": [int(p.pre.gray.shape[1]), int(p.pre.gray.shape[0])],
        },
    }
    if ocr is not None:
        result.update(ocr)
    result["timings_ms"] = p.timings.ms
    return result
