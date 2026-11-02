# DocServe — OCR & Document AI Inference Platform

**Tech stack:** Python · PyTorch · Hugging Face Transformers · OpenCV · PaddleOCR (PP-OCRv5) · Tesseract ·
ONNX / ONNX Runtime · FastAPI · Docker · Redis · Prometheus · AWS (S3, ECR)

A containerised inference service for lending documents. It takes a scanned or phone-photographed
**invoice, bank statement, payslip, PAN-style ID card or receipt** and returns the document type, the
OCR text and typed key-value fields (invoice number, IFSC, net pay, PAN, …). The focus is the
*serving* side: ONNX export and quantisation, dynamic batching, bounded concurrency, a Redis job
queue, Prometheus metrics, and measured latency/throughput/quality trade-offs.

```
upload ─► validation ─► OpenCV preprocessing ─► classifier (ONNX Runtime) ─► OCR ─► key-value extraction
          type/size/     perspective · deskew    micro-batched across        PP-OCRv5    label → value rules,
          pixels/count   denoise · CLAHE · ROI   concurrent requests         on ORT      fuzzy labels, OCR fixes
                │                                     │                        │
                └──── 503 when over capacity ─────────┴── Prometheus metrics ──┴──► JSON + Redis cache
async path: POST /v1/jobs ─► Redis list ─► worker pops N jobs ─► one batched ONNX call ─► results in Redis
```

## Results

All numbers below were measured on this repository's code, on a laptop CPU (Intel 11th-gen Tiger Lake-H,
12 logical cores, no GPU used). Raw outputs are in [`reports/`](reports).

### Classifier (ResNet-18 from Hugging Face, fine-tuned on document images)

| Version | Training | Test accuracy | Macro-F1 | ONNX FP32 | ONNX INT8 |
|---|---|---:|---:|---:|---:|
| v1 | linear probe (frozen ImageNet backbone, 3 epochs) | 97.0% | 0.970 | 97.0% | 96.5% |
| v2 | full fine-tune (5 epochs) | **99.5%** | 0.995 | 99.5% | 99.5% |

200 held-out test documents (40 per type). PyTorch → ONNX parity: max |Δlogit| = 1.5e-5. INT8 static
quantisation (QDQ, per-channel, 64 calibration images) shrinks the model 44.7 MB → 11.3 MB.
The synthetic task is easy (validation hit 100% after one epoch); the point of v1/v2 is the
versioning workflow, not the accuracy number.

### Classifier inference: PyTorch vs ONNX Runtime ([`benchmark_inference.md`](reports/benchmark_inference.md))

| Config | p50 @ batch 1 | p95 @ batch 1 | Peak throughput | Peak RSS |
|---|---:|---:|---:|---:|
| PyTorch eager | 23.8 ms | 29.7 ms | 56.6 img/s | 672 MB |
| ONNX Runtime, graph optimisations off | 15.1 ms | 17.0 ms | 70.0 img/s | 397 MB |
| ONNX Runtime, `ORT_ENABLE_ALL` | 10.1 ms | 12.0 ms | 105.2 img/s | 397 MB |
| ONNX Runtime, static INT8 | **2.9 ms** | 3.5 ms | **408.7 img/s** | **184 MB** |

- ONNX Runtime (optimised) is **2.35× faster** than PyTorch at batch 1; INT8 is **8.1× faster** with the
  same test accuracy and **73% less memory**.
- **Batch size does not raise CPU throughput** here (105 → 96 img/s from batch 1 → 32, while p50 grows
  10 → 337 ms): ONNX Runtime already saturates all cores on one image. Batching is a GPU lever; on CPU
  the batcher is configured with a small wait (8 ms) so it costs little latency.

### OCR & extraction ([`ocr_eval.md`](reports/ocr_eval.md))

Untouched **holdout** split: 150 documents (30 per type, 96 scans + 54 phone photos), PP-OCRv5 on ONNX
Runtime. Field accuracy = exact match after normalisation, using the true document type so it measures
OCR + extraction only (450 fields).

| Pipeline | CER | WER | Field accuracy | OCR p50 | Preprocess p50 |
|---|---:|---:|---:|---:|---:|
| Raw image → OCR | 0.116 | 0.167 | 97.3% | 1.82 s | 1 ms |
| OpenCV preprocessing → OCR | **0.007** | **0.031** | **97.6%** | 1.55 s | 29 ms |

| Slice (with preprocessing) | CER | Field accuracy |
|---|---:|---:|
| ID card / payslip / invoice | 0.010 / 0.001 / 0.005 | 100% / 100% / 98.9% |
| Bank statement / receipt | 0.005 / 0.017 | 96.7% / 92.2% |
| Scans / phone photos | 0.005 / 0.011 | 98.6% / 95.7% |

7 of the 11 remaining misses are receipt numbers and dates on small-font (17–19 px) receipts, 5 of them
phone photos; every miss is a missing value (`None`), never a wrong value returned as correct.

- **OCR engine choice:** the same PP-OCRv5 mobile detector + English recogniser run **6.7× faster on ONNX
  Runtime** (1.05 s/page) than on native Paddle Inference (7.04 s/page), whose oneDNN path crashes on
  this Paddle 3.x build and falls back to reference kernels. Field accuracy was equal (21/21) in the
  side-by-side check, so `paddle_onnx` is the default engine.
- **Preprocessing** mostly fixes reading order and character errors on skewed pages (CER 10×+ lower).
- **Error analysis drove two fixes**, both found on val/test pages and then confirmed on the untouched
  holdout split above:
  1. A 3×3 median filter eroded thin glyphs (`/`, `#`) in small receipt fonts — receipt field accuracy
     88% → 95% after switching to *adaptive* denoising (median only when salt-and-pepper noise is detected).
  2. OCR typos inside labels ("Baiance as on date", "Perrmanent Account Number") and spaces inside
     amounts ("104, 784.81") — fixed with fuzzy label matching (edit distance, labels ≥ 8 chars only)
     and OCR-tolerant amount parsing, each covered by unit tests.

### HTTP load tests ([`load_test_*.md`](reports))

Real HTTP requests against a spawned server, result cache off, 1,000 requests per row (200 for OCR).

| Endpoint | Batching | Concurrency | Req/s | p50 | p95 | Success |
|---|---|---:|---:|---:|---:|---:|
| `/v1/classify` | off | 1 / 8 / 32 | 17.8 / 37.2 / 42.5 | 55 / 211 / 746 ms | 69 / 279 / 826 ms | 100% |
| `/v1/classify` | on (16, 8 ms) | 1 / 8 / 32 | 13.7 / 32.9 / 31.6 | 74 / 233 / 946 ms | 93 / 347 / 1346 ms | 100% |
| `/v1/documents` (full OCR) | on | 4 | 0.6 | 5.9 s | 12.6 s | 100% |

**Finding: dynamic batching did not help in this setup.** The mean server-side batch was ~1.0 even at
concurrency 64, because requests spend their time in CPU-bound OpenCV preprocessing and reach the batcher
one at a time, and CPU ONNX Runtime gains nothing from batching anyway (see the benchmark above). It
adds ~8 ms of wait, so it is slightly slower here. Batching is expected to pay off on a GPU or with
cheaper preprocessing; set `DOCSERVE_BATCH_MAX_SIZE=1` to disable it. Full-pipeline throughput is bounded
by OCR (~1 s of CPU per page), so scale out with replicas and the Redis worker.
Every request succeeded; the 503 load-shedding path is covered by unit tests, not by this load run.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | liveness |
| GET | `/health/ready` | models loaded, OCR engine loaded, Redis reachable (503 otherwise) |
| GET | `/metrics` | Prometheus exposition |
| GET | `/v1/models` | registered versions, variant, provider, eval metrics |
| POST | `/v1/classify` | 1–16 images → document type (dynamically batched) |
| POST | `/v1/documents` | 1 image → type + OCR + fields (`model_version`, `ocr_engine`, `include_lines`, `preprocess`) |
| POST | `/v1/documents/batch` | 1–16 images → full results, per-file errors |
| POST | `/v1/jobs` | enqueue on Redis (202) for the batch worker |
| GET | `/v1/jobs/{id}` | job status / result |

```bash
curl -F "file=@data/test/invoice/invoice_0000.jpg" "http://localhost:8000/v1/documents?model_version=v2"
```

```json
{
  "document_id": "4f1c…", "model": {"name": "doc_classifier", "version": "v2", "variant": "fp32"},
  "doc_type": "invoice", "confidence": 0.98,
  "fields": {"invoice_number": "INV-29824", "invoice_date": "12/09/2023", "total_amount": "127,773.28"},
  "missing_fields": [], "ocr": {"engine": "paddle_onnx", "line_count": 31, "mean_confidence": 0.97},
  "preprocessing": {"steps": ["grayscale", "deskew", "denoise", "clahe", "roi_crop"], "skew_angle": 1.3, …},
  "timings_ms": {"decode": 7.0, "preprocess": 40.2, "classify": 23.5, "ocr": 1080.1, "extract": 1.1},
  "cached": false
}
```

Errors share one shape — `{"error": {"code", "message"}, "request_id"}` — with 404 (unknown model
version), 413 (file too large / too many files), 415 (not an image), 422 (corrupt, tiny or oversized
image, bad OCR engine) and 503 + `Retry-After` (over capacity, Redis jobs disabled).

## Design notes

- **Model versioning.** `models/doc_classifier/<version>/` holds `model.onnx`, `model.int8.onnx` and
  `metadata.json` (labels, normalisation, training history, val/test metrics, parity). Every version is
  loaded at start-up; requests pin one with `?model_version=v1`, otherwise the highest version serves.
  `DOCSERVE_MODEL_VARIANT=int8` switches all versions to the quantised graphs.
- **No train/serve skew.** Training images go through the exact `preprocess()` + `classifier_tensor()`
  the service uses.
- **Dynamic batching** ([`batcher.py`](docserve/batcher.py)): requests submit tensors to a per-version
  queue; a background task flushes at `BATCH_MAX_SIZE` items or `BATCH_MAX_WAIT_MS`, runs one
  `session.run()` in a thread and resolves each future.
- **Bounded concurrency** ([`api.py`](docserve/api.py)): at most `MAX_PENDING` documents admitted (else
  503 + `Retry-After`), and a semaphore caps CPU-heavy stages (preprocess, OCR) at `MAX_CONCURRENCY`.
- **Redis**: result cache keyed by `sha256(image) + version + variant + options`, and a job queue
  (`LPUSH` / `BRPOP` + non-blocking drain) so the worker classifies several documents in one ONNX call.
  Without `DOCSERVE_REDIS_URL` the service falls back to an in-memory LRU and disables `/v1/jobs`.
- **Observability**: JSON logs with a per-request id (`x-request-id` in/out); Prometheus histograms for
  HTTP latency, per-stage latency, batch size, queue wait, ONNX run time; counters for predictions per
  type, rejections, failures by stage, missing fields; OCR-confidence histograms per document type.
  [`deploy/alerts.yml`](deploy/alerts.yml) alerts on p95 latency, 5xx ratio, load shedding, job backlog,
  OCR-confidence drift and fields going missing.
- **Data hygiene**: train / val / test / holdout splits use disjoint name pools; the generator verifies
  no duplicate images or shared person names across any pair ([`data/split_report.json`]).
  Extraction rules were tuned on val/test error analysis, so OCR results are reported on `holdout`.
- **Storage**: optional upload archiving to local disk or S3 (`DOCSERVE_STORAGE_BACKEND=s3`,
  server-side encrypted, since documents carry PII).

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py                         # 1,000 train/val/test docs + leakage report
python scripts/generate_data.py --only holdout --holdout 30
python scripts/train_classifier.py --version v1 --mode linear_probe --epochs 3
python scripts/train_classifier.py --version v2 --mode finetune --epochs 5

uvicorn docserve.api:create_app --factory --port 8000   # docs at http://localhost:8000/docs
pytest -q                                               # 42 tests
```

Benchmarks and evaluation:

```bash
python scripts/benchmark_inference.py --version v2      # PyTorch vs ONNX FP32 / optimised / INT8
python scripts/evaluate_ocr.py --split holdout --per-type 30
python scripts/load_test.py --spawn --matrix --endpoint classify --requests 1000
python scripts/load_test.py --spawn --endpoint documents --requests 1000 --concurrency 8
```

Docker (API + batch worker + Redis + Prometheus at :9090):

```bash
docker compose up --build
AWS_REGION=ap-south-1 ./scripts/push_ecr.sh            # build, tag, push to ECR
```

The serving image installs only [`requirements-serve.txt`](requirements-serve.txt) — no PyTorch — and
bakes the OCR weights in at build time so containers start offline.

## Configuration

All settings are `DOCSERVE_*` environment variables ([`config.py`](docserve/config.py)): `MODEL_DIR`,
`MODEL_VARIANT` (fp32/int8), `ORT_DEVICE` (auto/cpu/cuda), `OCR_ENGINE` (paddle_onnx/paddle/tesseract),
`MAX_UPLOAD_MB`, `MAX_FILES_PER_REQUEST`, `MAX_IMAGE_PIXELS`, `BATCH_MAX_SIZE`, `BATCH_MAX_WAIT_MS`,
`MAX_CONCURRENCY`, `MAX_PENDING`, `REDIS_URL`, `CACHE_ENABLED`, `CACHE_TTL_S`, `WORKER_BATCH_SIZE`,
`STORAGE_BACKEND`, `S3_BUCKET`, `LOG_LEVEL`.

## Limitations

- Documents are synthetic (rendered with real fonts, then degraded like scans and phone photos). Real
  documents have more layout variety; the extraction rules would need more label variants and the
  classifier real training data.
- Tesseract support is implemented and installed in the Docker image but was not installed on the
  benchmark machine, so it has no numbers here. The Docker image was not built on the benchmark
  machine (Docker daemon unavailable); the compose file validates.
- No GPU numbers: the laptop's RTX 2050 was not used (CPU-only PyTorch and ONNX Runtime wheels). With
  `onnxruntime-gpu`, `ORT_DEVICE=auto` picks the CUDA provider and the benchmark adds a `_cuda` config.
