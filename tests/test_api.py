from __future__ import annotations

import asyncio

import fakeredis
import pytest
from fastapi.testclient import TestClient

from docserve import api as api_mod
from docserve.config import Settings
from docserve.registry import ModelRegistry
from docserve.store import JobQueue
from docserve.worker import Worker

from .conftest import blank_png


def make_client(model_dir, **overrides) -> TestClient:
    s = Settings(model_dir=model_dir, ocr_engine="fake", cache_enabled=True, **overrides)
    return TestClient(api_mod.create_app(s))


@pytest.fixture
def client(model_dir):
    with make_client(model_dir) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"] == {"models": True, "ocr": True}
    body = client.get("/v1/models").json()
    assert body["default_version"] == "v2"
    assert {m["version"] for m in body["models"]} == {"v1", "v2"}


def test_classify_batch_of_files(client, doc_pngs):
    files = [("files", (f"d{i}.jpg", b, "image/jpeg")) for i, b in enumerate(doc_pngs)]
    r = client.post("/v1/classify", files=files)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == len(doc_pngs)
    for res in results:
        assert res["model"]["version"] == "v2"
        assert abs(sum(res["probabilities"].values()) - 1) < 1e-3
        assert "fields" not in res  # classify-only


def test_documents_full_pipeline_and_cache(client, doc_png):
    r = client.post("/v1/documents", files={"file": ("inv.png", doc_png, "image/png")},
                    params={"model_version": "v1", "include_lines": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["version"] == "v1"
    assert body["ocr"]["engine"] == "fake" and len(body["ocr"]["lines"]) == 4
    assert body["cached"] is False
    assert {"decode", "preprocess", "classify", "ocr", "extract"} <= set(body["timings_ms"])
    assert "x-request-id" in r.headers
    again = client.post("/v1/documents", files={"file": ("inv.png", doc_png, "image/png")},
                        params={"model_version": "v1", "include_lines": "true"})
    assert again.json()["cached"] is True


@pytest.mark.parametrize("files,params,status,code", [
    ({"file": ("x.txt", b"hello", "text/plain")}, {}, 415, "unsupported_media_type"),
    ({"file": ("x.png", b"not an image", "image/png")}, {}, 422, "invalid_image"),
    ({"file": ("x.png", blank_png(20, 20), "image/png")}, {}, 422, "invalid_image"),
    ({"file": ("x.png", blank_png(200, 200), "image/png")}, {"model_version": "v9"}, 404, "model_not_found"),
    ({"file": ("x.png", blank_png(200, 200), "image/png")}, {"ocr_engine": "nope"}, 422, "invalid_ocr_engine"),
])
def test_validation_errors(client, files, params, status, code):
    r = client.post("/v1/documents", files=files, params=params)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code


def test_upload_size_and_file_count_limits(model_dir, doc_png):
    with make_client(model_dir, max_upload_mb=0.001, max_files_per_request=2) as c:
        r = c.post("/v1/documents", files={"file": ("a.png", doc_png, "image/png")})
        assert r.status_code == 413
        files = [("files", (f"{i}.png", blank_png(100, 100), "image/png")) for i in range(3)]
        assert c.post("/v1/classify", files=files).json()["error"]["code"] == "too_many_files"


def test_load_shedding_returns_503(model_dir, doc_png):
    with make_client(model_dir, max_pending=0) as c:
        r = c.post("/v1/documents", files={"file": ("a.png", doc_png, "image/png")})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
        assert 'docserve_rejected_total{reason="overloaded"}' in c.get("/metrics").text


def test_metrics_exposed(client, doc_png):
    client.post("/v1/classify", files=[("files", ("a.png", doc_png, "image/png"))])
    text = client.get("/metrics").text
    for name in ("docserve_http_requests_total", "docserve_stage_seconds_bucket", "docserve_batch_size_bucket",
                 "docserve_model_inference_seconds_bucket", "docserve_predictions_total"):
        assert name in text


def test_jobs_disabled_without_redis(client, doc_png):
    r = client.post("/v1/jobs", files={"file": ("a.png", doc_png, "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "jobs_disabled"


def test_async_job_roundtrip_with_worker(model_dir, doc_png, monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(api_mod, "redis_client", lambda url: fake)
    with make_client(model_dir, redis_url="redis://fake") as c:
        r = c.post("/v1/jobs", files={"file": ("a.png", doc_png, "image/png")})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert c.get(f"/v1/jobs/{job_id}").json()["status"] == "queued"

        s = Settings(model_dir=model_dir, ocr_engine="fake", redis_url="redis://fake")
        queue = JobQueue(fake)
        worker = Worker(s, queue, ModelRegistry(model_dir, "doc_classifier").load())
        assert worker.process_batch(queue.pop_batch(8, timeout=1)) == 1

        job = c.get(f"/v1/jobs/{job_id}").json()
        assert job["status"] == "done"
        assert job["result"]["ocr"]["engine"] == "fake"
        assert c.get("/v1/jobs/doesnotexist").status_code == 404


def test_micro_batcher_groups_concurrent_requests(model_dir):
    import numpy as np

    from docserve.batcher import MicroBatcher

    model = ModelRegistry(model_dir, "doc_classifier").load().resolve("v1")
    calls = []
    orig = model.predict
    model.predict = lambda b: (calls.append(len(b)), orig(b))[1]

    async def go():
        b = MicroBatcher(model, max_batch_size=8, max_wait_ms=50)
        b.start()
        x = np.zeros((3, model.input_size, model.input_size), np.float32)
        out = await asyncio.gather(*(b.submit(x) for _ in range(20)))
        await b.stop()
        return out

    out = asyncio.run(go())
    assert len(out) == 20 and all(o.shape == (5,) for o in out)
    assert sum(calls) == 20 and max(calls) == 8 and len(calls) <= 4
