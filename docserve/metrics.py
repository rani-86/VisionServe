"""Prometheus metrics for the service and the batch worker."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16)

HTTP_REQUESTS = Counter(
    "docserve_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "docserve_http_request_seconds", "End-to-end HTTP latency", ["route"], buckets=LATENCY_BUCKETS
)
STAGE_LATENCY = Histogram(
    "docserve_stage_seconds",
    "Latency of each pipeline stage",
    ["stage"],  # decode | preprocess | classify | ocr | extract
    buckets=LATENCY_BUCKETS,
)
INFLIGHT = Gauge("docserve_inflight_requests", "Documents currently admitted for processing")
REJECTED = Counter(
    "docserve_rejected_total", "Requests rejected before processing", ["reason"]
)  # overloaded | validation
FAILURES = Counter("docserve_failures_total", "Processing failures", ["stage", "error"])

BATCH_SIZE = Histogram(
    "docserve_batch_size", "Images per classifier inference call", buckets=(1, 2, 4, 8, 16, 32, 64)
)
BATCH_QUEUE_WAIT = Histogram(
    "docserve_batch_queue_wait_seconds", "Time an image waits in the micro-batcher", buckets=LATENCY_BUCKETS
)
MODEL_INFERENCE = Histogram(
    "docserve_model_inference_seconds", "ONNX Runtime run() latency per batch", ["version", "variant"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "docserve_predictions_total", "Classifier predictions", ["version", "doc_type"]
)
CLASSIFIER_CONFIDENCE = Histogram(
    "docserve_classifier_confidence", "Top-1 softmax probability", buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
)
OCR_CONFIDENCE = Histogram(
    "docserve_ocr_line_confidence", "Mean OCR line confidence per document", ["engine", "doc_type"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
)
FIELDS_MISSING = Counter(
    "docserve_fields_missing_total", "Expected key-value fields that could not be extracted", ["doc_type", "field"]
)
CACHE_EVENTS = Counter("docserve_cache_events_total", "Result cache lookups", ["result"])  # hit | miss
JOB_EVENTS = Counter("docserve_jobs_total", "Async job lifecycle events", ["event"])  # queued | done | failed
JOB_QUEUE_DEPTH = Gauge("docserve_job_queue_depth", "Jobs waiting in the Redis queue")
MODEL_INFO = Gauge("docserve_model_loaded", "Loaded model versions", ["name", "version", "variant", "provider"])
