"""HTTP load test against a running DocServe instance (or one it spawns).

    # against a running server
    python scripts/load_test.py --url http://localhost:8000 --endpoint classify --requests 1000 --concurrency 32

    # spawn the server per scenario: dynamic batching off vs on, several concurrency levels
    python scripts/load_test.py --spawn --matrix --requests 1000

Uses real test-split images. Reports p50/p95/p99 latency, throughput, error rate,
status codes and the server's own batch-size histogram (scraped from /metrics).
Writes reports/load_test.{json,md}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_images(data: Path, n: int) -> list[tuple[str, bytes]]:
    paths = sorted((data / "test").glob("*/*.jpg"))
    random.Random(0).shuffle(paths)
    return [(p.name, p.read_bytes()) for p in paths[:n]]


def batch_hist(metrics_text: str) -> dict:
    """Mean batch size and inference-call count from the docserve_batch_size histogram."""
    s = re.search(r"^docserve_batch_size_sum ([\d.e+]+)", metrics_text, re.M)
    c = re.search(r"^docserve_batch_size_count ([\d.e+]+)", metrics_text, re.M)
    rej = sum(float(v) for v in re.findall(r'^docserve_rejected_total\{reason="overloaded"\} ([\d.e+]+)',
                                            metrics_text, re.M))
    if not (s and c) or float(c.group(1)) == 0:
        return {"inference_calls": 0, "mean_batch_size": 0, "rejected_overloaded": rej}
    return {"inference_calls": int(float(c.group(1))),
            "mean_batch_size": round(float(s.group(1)) / float(c.group(1)), 2), "rejected_overloaded": rej}


async def run_load(url: str, endpoint: str, images: list[tuple[str, bytes]], n: int, concurrency: int,
                   params: dict) -> dict:
    path = {"classify": "/v1/classify", "documents": "/v1/documents"}[endpoint]
    field = "files" if endpoint == "classify" else "file"
    lat, codes = [], {}
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        queue.put_nowait(i)
    async with httpx.AsyncClient(base_url=url, timeout=120,
                                 limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        before = batch_hist((await client.get("/metrics")).text)

        async def worker():
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                name, data = images[i % len(images)]
                t0 = time.perf_counter()
                try:
                    r = await client.post(path, files={field: (name, data, "image/jpeg")}, params=params,
                                          headers={"x-request-id": f"load-{i}"})
                    code = r.status_code
                except httpx.HTTPError as e:
                    code = type(e).__name__
                lat.append((time.perf_counter() - t0) * 1000)
                codes[str(code)] = codes.get(str(code), 0) + 1

        t_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_start
        after = batch_hist((await client.get("/metrics")).text)
    a = np.array(lat)
    ok = codes.get("200", 0)
    calls = after["inference_calls"] - before["inference_calls"]
    return {
        "endpoint": endpoint, "requests": n, "concurrency": concurrency, "wall_s": round(wall, 2),
        "throughput_rps": round(n / wall, 1), "success_rate": round(ok / n, 4), "status_codes": codes,
        "p50_ms": round(float(np.percentile(a, 50)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1), "mean_ms": round(float(a.mean()), 1),
        "server_inference_calls": calls,
        "server_mean_batch_size": round(ok / calls, 2) if calls else None,
        "server_rejected_overloaded": after["rejected_overloaded"] - before["rejected_overloaded"],
    }


class Server:
    """Starts uvicorn with scenario-specific env vars; the result cache is off so every request does real work."""

    def __init__(self, port: int, env: dict):
        self.port, self.env = port, env
        self.proc = None

    def __enter__(self):
        env = os.environ | {"DOCSERVE_CACHE_ENABLED": "0", "DOCSERVE_PRELOAD_OCR": "0", "DOCSERVE_LOG_LEVEL": "WARNING"} \
            | self.env
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "docserve.api:create_app", "--factory", "--port", str(self.port),
             "--log-level", "warning"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(120):
            try:
                if httpx.get(f"http://127.0.0.1:{self.port}/health/live", timeout=1).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.5)
        self.proc.kill()
        raise RuntimeError(self.proc.stderr.read().decode()[-2000:])

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait(10)


def to_markdown(rows: list[dict]) -> str:
    out = ["# HTTP load test", "",
           "| Scenario | Endpoint | Requests | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | "
           "Success | Mean server batch |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r.get('scenario', '-')} | {r['endpoint']} | {r['requests']} | {r['concurrency']} | "
                   f"{r['throughput_rps']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | "
                   f"{r['success_rate'] * 100:.1f}% | {r['server_mean_batch_size']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", choices=["classify", "documents"], default="classify")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[32])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--spawn", action="store_true", help="start uvicorn for each scenario")
    ap.add_argument("--matrix", action="store_true", help="batching off vs on x concurrency 1/8/32/64")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    images = load_images(Path(args.data), 200)
    rows = []
    if args.spawn:
        scenarios = {"batching_off": {"DOCSERVE_BATCH_MAX_SIZE": "1"},
                     "batching_on": {"DOCSERVE_BATCH_MAX_SIZE": "16", "DOCSERVE_BATCH_MAX_WAIT_MS": "8"}}
        if not args.matrix:
            scenarios = {"batching_on": scenarios["batching_on"]}
        levels = [1, 8, 32, 64] if args.matrix else args.concurrency
        for name, env in scenarios.items():
            env = env | {"DOCSERVE_MAX_PENDING": "256", "DOCSERVE_MAX_CONCURRENCY": str(os.cpu_count() or 4)}
            with Server(args.port, env):
                url = f"http://127.0.0.1:{args.port}"
                asyncio.run(run_load(url, args.endpoint, images, 20, 4, {}))  # warm-up
                for c in levels:
                    r = asyncio.run(run_load(url, args.endpoint, images, args.requests, c, {}))
                    rows.append({"scenario": name} | r)
                    print(json.dumps(rows[-1]), flush=True)
    else:
        for c in args.concurrency:
            rows.append(asyncio.run(run_load(args.url, args.endpoint, images, args.requests, c, {})))
            print(json.dumps(rows[-1]), flush=True)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"load_test_{args.endpoint}"
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
