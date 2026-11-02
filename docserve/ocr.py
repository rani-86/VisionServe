"""OCR engine adapters returning lines with boxes and confidences.

  paddle_onnx  PP-OCRv5 (det + English rec) on ONNX Runtime   <- default
  paddle       PP-OCRv5 on native Paddle Inference (PaddleOCR 3.x)
  tesseract    Tesseract 5 LSTM on the binarised image
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .synth import group_rows


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OCREngine:
    name = "base"

    def recognize(self, gray: np.ndarray, binary: np.ndarray | None = None) -> list[OCRLine]:
        raise NotImplementedError


class PaddleEngine(OCREngine):
    """PaddleOCR 3.x (PP-OCRv5 mobile det + English rec). Fed the enhanced grayscale image."""

    name = "paddle"

    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,  # geometry is handled by our own preprocessing
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            # The default (min side >= 64) runs detection at full resolution; capping the long
            # side bounds detector cost on large scans/photos.
            text_det_limit_side_len=int(os.getenv("DOCSERVE_PADDLE_DET_SIDE", "1280")),
            text_det_limit_type="max",
            text_recognition_batch_size=int(os.getenv("DOCSERVE_PADDLE_REC_BATCH", "16")),
            cpu_threads=int(os.getenv("DOCSERVE_PADDLE_CPU_THREADS", "4")),
            # Paddle 3.x oneDNN kernels fail on some PIR ops on CPU; opt in explicitly.
            enable_mkldnn=os.getenv("DOCSERVE_PADDLE_MKLDNN", "0") == "1",
        )
        self._lock = threading.Lock()  # the predictor is not thread-safe

    def recognize(self, gray: np.ndarray, binary: np.ndarray | None = None) -> list[OCRLine]:
        img = np.repeat(gray[..., None], 3, axis=2) if gray.ndim == 2 else gray
        lines = []
        with self._lock:
            results = list(self._ocr.predict(img))
        for res in results:
            data = res.json["res"] if hasattr(res, "json") else res
            for text, score, poly in zip(data["rec_texts"], data["rec_scores"], data["rec_polys"]):
                if str(text).strip():
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    x0, y0 = pts.min(axis=0)
                    x1, y1 = pts.max(axis=0)
                    lines.append(OCRLine(str(text).strip(), float(score), (int(x0), int(y0), int(x1), int(y1))))
        return lines


class PaddleONNXEngine(OCREngine):
    """The same PP-OCRv5 mobile det + English rec models, exported to ONNX and run on ONNX Runtime
    (via RapidOCR's DB post-processing and CTC decoding). On CPU this avoids Paddle Inference's
    reference kernels and is several times faster for identical models."""

    name = "paddle_onnx"

    def __init__(self):
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        common = {"engine_type": EngineType.ONNXRUNTIME, "ocr_version": OCRVersion.PPOCRV5,
                  "model_type": ModelType.MOBILE}
        params = {"Global.use_cls": False, "Global.log_level": "warning",
                  "Det.lang_type": LangDet.CH, "Rec.lang_type": LangRec.EN,  # PP-OCRv5 det is multilingual
                  "Det.limit_side_len": int(os.getenv("DOCSERVE_PADDLE_DET_SIDE", "1280")), "Det.limit_type": "max",
                  "Rec.rec_batch_num": int(os.getenv("DOCSERVE_PADDLE_REC_BATCH", "16"))}
        params |= {f"{stage}.{k}": v for stage in ("Det", "Rec") for k, v in common.items()}
        self._ocr = RapidOCR(params=params)

    def recognize(self, gray: np.ndarray, binary: np.ndarray | None = None) -> list[OCRLine]:
        img = np.repeat(gray[..., None], 3, axis=2) if gray.ndim == 2 else gray
        res = self._ocr(img, use_cls=False)
        if res.boxes is None:
            return []
        lines = []
        for box, text, score in zip(res.boxes, res.txts, res.scores):
            if str(text).strip():
                pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                lines.append(OCRLine(str(text).strip(), float(score), (int(x0), int(y0), int(x1), int(y1))))
        return lines


class TesseractEngine(OCREngine):
    """Tesseract 5 LSTM via pytesseract. Fed the binarised image; words are merged into lines."""

    name = "tesseract"

    def __init__(self):
        import pytesseract

        pytesseract.get_tesseract_version()  # raises if the binary is missing
        self._pt = pytesseract

    def recognize(self, gray: np.ndarray, binary: np.ndarray | None = None) -> list[OCRLine]:
        img = binary if binary is not None else gray
        d = self._pt.image_to_data(img, output_type=self._pt.Output.DICT, config="--oem 1 --psm 3")
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i, text in enumerate(d["text"]):
            if text.strip() and float(d["conf"][i]) >= 0:
                groups.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(i)
        lines = []
        for idx in groups.values():
            x0 = min(d["left"][i] for i in idx)
            y0 = min(d["top"][i] for i in idx)
            x1 = max(d["left"][i] + d["width"][i] for i in idx)
            y1 = max(d["top"][i] + d["height"][i] for i in idx)
            conf = float(np.mean([float(d["conf"][i]) for i in idx])) / 100
            lines.append(OCRLine(" ".join(d["text"][i] for i in idx), conf, (x0, y0, x1, y1)))
        return lines


ENGINES = {"paddle_onnx": PaddleONNXEngine, "paddle": PaddleEngine, "tesseract": TesseractEngine}


@lru_cache(maxsize=None)
def get_engine(name: str) -> OCREngine:
    if name not in ENGINES:
        raise ValueError(f"unknown OCR engine {name!r}; choose from {sorted(ENGINES)}")
    return ENGINES[name]()


def available_engines(names: list[str] | None = None) -> list[str]:
    """Engines (from `names`, default all) that load on this host."""
    out = []
    for name in names or ENGINES:
        try:
            get_engine(name)
            out.append(name)
        except Exception:
            pass
    return out


def lines_to_rows(lines: list[OCRLine]) -> list[str]:
    return group_rows([(ln.text, ln.bbox) for ln in lines])
