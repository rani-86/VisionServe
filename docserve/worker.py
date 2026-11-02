"""Redis batch worker for async jobs: `python -m docserve.worker`.

Pops up to WORKER_BATCH_SIZE job ids at a time, preprocesses each document,
classifies the whole batch with ONE ONNX Runtime call, then runs OCR +
extraction per document and writes results back to Redis. Exposes Prometheus
metrics on :9101.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time

import numpy as np
from prometheus_client import start_http_server

from .config import get_settings
from .logging_utils import configure_logging
from .metrics import BATCH_SIZE, FAILURES, JOB_EVENTS, JOB_QUEUE_DEPTH, STAGE_LATENCY
from .ocr import get_engine
from .pipeline import assemble, classification, prepare, run_ocr_and_extract
from .registry import ModelRegistry
from .store import JobQueue, RedisCache, redis_client

log = logging.getLogger("docserve.worker")


class Worker:
    def __init__(self, settings, queue: JobQueue, registry: ModelRegistry, cache: RedisCache | None = None):
        self.s, self.queue, self.registry, self.cache = settings, queue, registry, cache
        self.running = True

    def process_batch(self, job_ids: list[str]) -> int:
        """Returns the number of jobs completed successfully."""
        prepared = {}  # job_id -> (Prepared, model, params)
        for jid in job_ids:
            job = self.queue.get(jid)
            data = self.queue.blob(jid)
            if not job or data is None:
                FAILURES.labels("worker", "expired").inc()
                continue
            self.queue.update(jid, status="processing", started_at=time.time())
            try:
                model = self.registry.resolve(job["params"].get("model_version"))
                p = prepare(data, model, max_pixels=self.s.max_image_pixels, min_side=self.s.min_image_side)
                prepared[jid] = (p, model, job["params"])
            except Exception as e:
                self._fail(jid, "prepare", e)

        # One inference call per model version present in the batch.
        by_version: dict[str, list[str]] = {}
        for jid, (_, model, _) in prepared.items():
            by_version.setdefault(model.version, []).append(jid)
        done = 0
        for version, jids in by_version.items():
            model = self.registry.resolve(version)
            BATCH_SIZE.observe(len(jids))
            t0 = time.perf_counter()
            try:
                probs = model.predict(np.stack([prepared[j][0].tensor for j in jids]))
            except Exception as e:
                for j in jids:
                    self._fail(j, "classify", e)
                continue
            dt = time.perf_counter() - t0
            STAGE_LATENCY.labels("classify").observe(dt)
            for j, pr in zip(jids, probs):
                p, _, params = prepared[j]
                p.timings.ms["classify"] = round(dt * 1000, 2)
                try:
                    cls = classification(model, pr)
                    ocr = run_ocr_and_extract(p, cls["doc_type"], params.get("ocr_engine") or self.s.ocr_engine,
                                              include_lines=False)
                    result = assemble(p, model, cls, ocr) | {"filename": params.get("filename"), "cached": False}
                    self.queue.update(j, status="done", finished_at=time.time(), result=result)
                    self.queue.drop_blob(j)
                    JOB_EVENTS.labels("done").inc()
                    done += 1
                except Exception as e:
                    self._fail(j, "ocr", e)
        return done

    def _fail(self, jid: str, stage: str, exc: Exception) -> None:
        FAILURES.labels(stage, type(exc).__name__).inc()
        JOB_EVENTS.labels("failed").inc()
        log.exception("job failed", extra={"job_id": jid, "stage": stage})
        self.queue.update(jid, status="failed", finished_at=time.time(), error=f"{stage}: {exc}")
        self.queue.drop_blob(jid)

    def run(self) -> None:
        log.info("worker started", extra={"batch_size": self.s.worker_batch_size})
        while self.running:
            try:
                JOB_QUEUE_DEPTH.set(self.queue.depth())
                ids = self.queue.pop_batch(self.s.worker_batch_size, timeout=2)
                if ids:
                    n = self.process_batch(ids)
                    log.info("batch processed", extra={"jobs": len(ids), "succeeded": n})
            except Exception:
                log.exception("worker loop error")
                time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-port", type=int, default=9101)
    args = ap.parse_args()
    s = get_settings()
    configure_logging(s.log_level)
    if not s.redis_url:
        raise SystemExit("DOCSERVE_REDIS_URL is required for the worker")
    start_http_server(args.metrics_port)
    client = redis_client(s.redis_url)
    registry = ModelRegistry(s.model_dir, s.model_name, s.model_variant, s.ort_device, s.ort_intra_threads).load()
    get_engine(s.ocr_engine)  # fail fast if OCR is unavailable
    worker = Worker(s, JobQueue(client, s.job_ttl_s), registry, RedisCache(client))

    def stop(*_):
        worker.running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.run()


if __name__ == "__main__":
    main()
