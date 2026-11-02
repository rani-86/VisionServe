"""Offline classifier benchmark: PyTorch eager vs ONNX Runtime configurations across batch sizes.

    python scripts/benchmark_inference.py --version v2 --batch-sizes 1 2 4 8 16 32

Configurations
  pytorch_eager      HF model in torch.inference_mode()
  onnx_fp32_noopt    ONNX Runtime, graph optimisations disabled (baseline)
  onnx_fp32_opt      ONNX Runtime, ORT_ENABLE_ALL (constant folding, Conv+BN/ReLU fusion)
  onnx_int8          static-quantised QDQ model, ORT_ENABLE_ALL
  *_cuda             same on CUDAExecutionProvider when onnxruntime-gpu is installed

Each configuration runs in its own subprocess so peak memory is measured cleanly.
Writes reports/benchmark_inference.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def peak_rss_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)  # Windows
    if peak is None:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB
    return peak / 1e6


def run_config(cfg: str, model_dir: Path, batch_sizes: list[int], images: int, threads: int) -> dict:
    """Executed inside the child process."""
    rng = np.random.default_rng(0)
    base_mem = peak_rss_mb()
    if cfg == "pytorch_eager":
        import torch
        from transformers import AutoModelForImageClassification

        if threads:
            torch.set_num_threads(threads)
        model = AutoModelForImageClassification.from_pretrained(model_dir / "hf").eval()

        def infer(x):
            with torch.inference_mode():
                return model(pixel_values=torch.from_numpy(x)).logits.numpy()
    else:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if "noopt" in cfg
                                         else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        if threads:
            opts.intra_op_num_threads = threads
        path = model_dir / ("model.int8.onnx" if "int8" in cfg else "model.onnx")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cfg.endswith("_cuda") else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)

        def infer(x):
            return sess.run(None, {"pixel_values": x})[0]

    meta = json.loads((model_dir / "metadata.json").read_text())
    size = meta["input_size"]
    load_mem = peak_rss_mb()
    rows = []
    for bs in batch_sizes:
        x = rng.standard_normal((bs, 3, size, size), dtype=np.float32)
        for _ in range(3):  # warm-up (allocator, kernel selection)
            infer(x)
        iters = max(5, images // bs)
        lat = []
        t_start = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            infer(x)
            lat.append((time.perf_counter() - t0) * 1000)
        wall = time.perf_counter() - t_start
        lat = np.array(lat)
        rows.append({"batch_size": bs, "iterations": iters, "images": iters * bs,
                     "p50_ms": round(float(np.percentile(lat, 50)), 2),
                     "p95_ms": round(float(np.percentile(lat, 95)), 2),
                     "per_image_ms": round(float(np.percentile(lat, 50)) / bs, 2),
                     "throughput_ips": round(iters * bs / wall, 1)})
    return {"config": cfg, "model_load_mb": round(load_mem - base_mem, 1), "peak_rss_mb": round(peak_rss_mb(), 1),
            "results": rows}


def to_markdown(report: dict) -> str:
    lines = [f"# Classifier inference benchmark ({report['model_version']})", "",
             f"Host: {report['host']['cpu']} · {report['host']['cores']} logical cores · "
             f"ONNX Runtime {report['host']['onnxruntime']} · torch {report['host']['torch']}", "",
             "| Config | Batch | p50 (ms) | p95 (ms) | ms / image | Throughput (img/s) | Peak RSS (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for c in report["configs"]:
        for r in c["results"]:
            lines.append(f"| {c['config']} | {r['batch_size']} | {r['p50_ms']} | {r['p95_ms']} | {r['per_image_ms']} | "
                         f"{r['throughput_ips']} | {c['peak_rss_mb']} |")
    if report.get("summary"):
        lines += ["", "## Summary", ""] + [f"- {s}" for s in report["summary"]]
    return "\n".join(lines) + "\n"


def summarize(configs: list[dict]) -> list[str]:
    by = {c["config"]: {r["batch_size"]: r for r in c["results"]} for c in configs}
    out = []
    if "pytorch_eager" in by and "onnx_fp32_opt" in by:
        for bs in (1, max(by["onnx_fp32_opt"])):
            a, b = by["pytorch_eager"].get(bs), by["onnx_fp32_opt"].get(bs)
            if a and b:
                out.append(f"Batch {bs}: ONNX Runtime (optimised) p50 {b['p50_ms']} ms vs PyTorch {a['p50_ms']} ms "
                           f"({a['p50_ms'] / b['p50_ms']:.2f}x faster)")
    if "onnx_fp32_noopt" in by and "onnx_fp32_opt" in by:
        b1n, b1o = by["onnx_fp32_noopt"][1], by["onnx_fp32_opt"][1]
        out.append(f"Graph optimisations at batch 1: {b1n['p50_ms']} -> {b1o['p50_ms']} ms")
    if "onnx_int8" in by and "onnx_fp32_opt" in by:
        bs = max(by["onnx_int8"])
        out.append(f"INT8 vs FP32 at batch {bs}: {by['onnx_int8'][bs]['throughput_ips']} vs "
                   f"{by['onnx_fp32_opt'][bs]['throughput_ips']} img/s")
    if "onnx_fp32_opt" in by:
        r = by["onnx_fp32_opt"]
        bmax = max(r)
        out.append(f"Batching (ONNX FP32): throughput {r[1]['throughput_ips']} img/s at batch 1 -> "
                   f"{r[bmax]['throughput_ips']} img/s at batch {bmax}, while p50 latency rises "
                   f"{r[1]['p50_ms']} -> {r[bmax]['p50_ms']} ms")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="latest")
    ap.add_argument("--models", default=str(ROOT / "models" / "doc_classifier"))
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--images", type=int, default=256, help="approx images per (config, batch size)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--configs", nargs="+",
                    default=["pytorch_eager", "onnx_fp32_noopt", "onnx_fp32_opt", "onnx_int8"])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    models = Path(args.models)
    if args.version == "latest":
        from docserve.registry import _version_key
        args.version = max((p.name for p in models.iterdir() if p.is_dir()), key=_version_key)
    model_dir = models / args.version

    if args.child:
        print(json.dumps(run_config(args.child, model_dir, args.batch_sizes, args.images, args.threads)))
        return

    import onnxruntime as ort
    import torch

    configs = list(args.configs)
    if "CUDAExecutionProvider" in ort.get_available_providers():
        configs.append("onnx_fp32_opt_cuda")
    results = []
    for cfg in configs:
        if "int8" in cfg and not (model_dir / "model.int8.onnx").exists():
            continue
        print(f"running {cfg} ...", flush=True)
        cmd = [sys.executable, __file__, "--child", cfg, "--version", args.version, "--models", str(models),
               "--images", str(args.images), "--threads", str(args.threads), "--batch-sizes",
               *map(str, args.batch_sizes)]
        out = subprocess.run(cmd, capture_output=True, text=True, env=os.environ | {"PYTHONWARNINGS": "ignore"})
        if out.returncode != 0:
            print(out.stderr[-2000:])
            continue
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))

    report = {
        "model_version": args.version,
        "host": {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
                 "os": platform.platform(), "onnxruntime": ort.__version__, "torch": torch.__version__,
                 "providers": ort.get_available_providers()},
        "configs": results,
        "summary": summarize(results),
    }
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "benchmark_inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "benchmark_inference.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
