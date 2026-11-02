"""Test fixtures: a tiny ONNX classifier in a temp model registry and a deterministic fake OCR engine."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docserve import ocr as ocr_mod  # noqa: E402
from docserve.ocr import OCREngine, OCRLine  # noqa: E402
from docserve.synth import DOC_TYPES, DocumentGenerator  # noqa: E402

SIZE = 64


class FakeOCR(OCREngine):
    name = "fake"

    def recognize(self, gray, binary=None):
        return [OCRLine("TAX INVOICE", 0.99, (10, 10, 200, 30)),
                OCRLine("Invoice No: INV-12345", 0.97, (10, 50, 300, 70)),
                OCRLine("Invoice Date: 01/02/2025", 0.96, (10, 90, 300, 110)),
                OCRLine("Grand Total: 1,234.50", 0.95, (10, 130, 300, 150))]


def _write_tiny_model(vdir: Path, seed: int) -> None:
    """A real (random-weight) conv net exported to ONNX with a dynamic batch axis."""
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, stride=4), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(4, len(DOC_TYPES))).eval()
    vdir.mkdir(parents=True)
    torch.onnx.export(net, (torch.randn(1, 3, SIZE, SIZE),), str(vdir / "model.onnx"), dynamo=False,
                      input_names=["pixel_values"], output_names=["logits"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    (vdir / "metadata.json").write_text(json.dumps({
        "labels": DOC_TYPES, "input_size": SIZE, "mean": [0.5] * 3, "std": [0.5] * 3,
        "base_model": "tiny-test-net", "metrics": {"test": {"accuracy": 0.2}}}))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    _write_tiny_model(root / "doc_classifier" / "v1", 0)
    _write_tiny_model(root / "doc_classifier" / "v2", 1)
    return root


@pytest.fixture(autouse=True)
def fake_ocr(monkeypatch):
    monkeypatch.setitem(ocr_mod.ENGINES, "fake", FakeOCR)
    ocr_mod.get_engine.cache_clear()
    yield
    ocr_mod.get_engine.cache_clear()


@pytest.fixture(scope="session")
def doc_png() -> bytes:
    s = DocumentGenerator("test", seed=1).generate("invoice")
    return cv2.imencode(".png", s.image)[1].tobytes()


@pytest.fixture(scope="session")
def doc_pngs() -> list[bytes]:
    g = DocumentGenerator("test", seed=2)
    return [cv2.imencode(".jpg", g.generate(t).image)[1].tobytes() for t in DOC_TYPES]


def blank_png(w: int, h: int) -> bytes:
    return cv2.imencode(".png", np.full((h, w, 3), 255, np.uint8))[1].tobytes()
