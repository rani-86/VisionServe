"""Versioned model registry backed by ONNX Runtime.

Layout on disk:
    models/<name>/<version>/
        metadata.json        labels, input size, normalisation, training metrics
        model.onnx           FP32 export
        model.int8.onnx      static INT8 quantisation (optional)

Every version present is loaded at start-up so requests can pin one with
`model_version=v1`; "latest" resolves to the highest version number.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .metrics import MODEL_INFERENCE, MODEL_INFO

log = logging.getLogger(__name__)

VARIANT_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx"}


class ModelNotFound(KeyError):
    pass


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)) or (0,)


def make_session(path: Path, device: str = "auto", intra_threads: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads:
        opts.intra_op_num_threads = intra_threads
    available = ort.get_available_providers()
    if device == "cuda" or (device == "auto" and "CUDAExecutionProvider" in available):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


@dataclass
class LoadedModel:
    name: str
    version: str
    variant: str
    metadata: dict
    session: ort.InferenceSession

    def __post_init__(self):
        self.labels: list[str] = self.metadata["labels"]
        self.input_size: int = int(self.metadata["input_size"])
        self.mean: list[float] = self.metadata["mean"]
        self.std: list[float] = self.metadata["std"]
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]  # session.run() is thread-safe

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Run a (N, 3, H, W) float32 batch, return (N, C) softmax probabilities."""
        t0 = time.perf_counter()
        logits = self.session.run(None, {self.input_name: batch.astype(np.float32, copy=False)})[0]
        MODEL_INFERENCE.labels(self.version, self.variant).observe(time.perf_counter() - t0)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "variant": self.variant,
            "provider": self.provider,
            "labels": self.labels,
            "input_size": self.input_size,
            "base_model": self.metadata.get("base_model"),
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics", {}),
        }


class ModelRegistry:
    def __init__(self, root: Path, name: str, variant: str = "fp32", device: str = "auto", intra_threads: int = 0):
        self.root, self.name, self.variant = Path(root), name, variant
        self.device, self.intra_threads = device, intra_threads
        self.models: dict[str, LoadedModel] = {}

    def load(self) -> "ModelRegistry":
        base = self.root / self.name
        if not base.exists():
            raise FileNotFoundError(f"no model directory at {base}; run scripts/train_classifier.py first")
        for vdir in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: _version_key(p.name)):
            meta_path = vdir / "metadata.json"
            onnx_path = vdir / VARIANT_FILES[self.variant]
            if not onnx_path.exists() and self.variant != "fp32":
                log.warning("variant missing, falling back to fp32", extra={"version": vdir.name, "variant": self.variant})
                onnx_path = vdir / VARIANT_FILES["fp32"]
            if not (meta_path.exists() and onnx_path.exists()):
                continue
            variant = "int8" if onnx_path.name == VARIANT_FILES["int8"] else "fp32"
            session = make_session(onnx_path, self.device, self.intra_threads)
            model = LoadedModel(self.name, vdir.name, variant, json.loads(meta_path.read_text()), session)
            self.models[vdir.name] = model
            MODEL_INFO.labels(self.name, vdir.name, variant, model.provider).set(1)
            log.info("model loaded", extra={"model": self.name, "version": vdir.name, "variant": variant,
                                            "provider": model.provider})
        if not self.models:
            raise FileNotFoundError(f"no loadable versions under {base}")
        return self

    @property
    def latest(self) -> str:
        return max(self.models, key=_version_key)

    def resolve(self, version: str | None) -> LoadedModel:
        v = self.latest if version in (None, "", "latest") else version
        if v not in self.models:
            raise ModelNotFound(f"model version {version!r} not found; available: {sorted(self.models)}")
        return self.models[v]

    def describe(self) -> list[dict]:
        return [m.describe() | {"default": m.version == self.latest} for m in self.models.values()]
