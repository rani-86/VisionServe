"""PyTorch -> ONNX export, ONNX Runtime parity check and static INT8 quantisation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class LogitsOnly:
    """Wrap a Hugging Face image classifier so forward() returns a plain logits tensor."""

    def __new__(cls, model):
        import torch

        class _W(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, pixel_values):
                return self.m(pixel_values=pixel_values).logits

        return _W(model).eval()


def export_onnx(model, path: Path, input_size: int, opset: int = 17) -> Path:
    import torch

    wrapped = LogitsOnly(model)
    dummy = torch.randn(2, 3, input_size, input_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["pixel_values"], output_names=["logits"], opset_version=opset)
    try:  # TorchScript-based exporter: dynamic batch axis via dynamic_axes
        torch.onnx.export(wrapped, (dummy,), str(path), dynamo=False,
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}}, **kwargs)
    except Exception as e:  # newer torch: dynamo exporter only
        log.warning("legacy exporter unavailable (%s); using dynamo exporter", e)
        batch = torch.export.Dim("batch", min=1, max=256)
        torch.onnx.export(wrapped, (dummy,), str(path), dynamo=True, external_data=False,
                          dynamic_shapes={"pixel_values": {0: batch}}, **kwargs)
    import onnx

    onnx.checker.check_model(str(path))
    return path


def parity(model, onnx_path: Path, batch: np.ndarray) -> float:
    """Max absolute difference between PyTorch and ONNX Runtime logits."""
    import onnxruntime as ort
    import torch

    with torch.inference_mode():
        ref = LogitsOnly(model)(torch.from_numpy(batch)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": batch})[0]
    return float(np.abs(ref - out).max())


def quantize_int8(fp32_path: Path, int8_path: Path, calibration: np.ndarray) -> Path:
    """Static QDQ INT8 quantisation, calibrated on representative training images."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray):
            self.it = iter([{"pixel_values": data[i:i + 1]} for i in range(len(data))])

        def get_next(self):
            return next(self.it, None)

    prep = int8_path.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32_path), str(prep), skip_symbolic_shape=True)
    quantize_static(str(prep), str(int8_path), Reader(calibration.astype(np.float32)),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax)
    prep.unlink(missing_ok=True)
    return int8_path


def onnx_predict(path: Path, x: np.ndarray, batch: int = 32) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"pixel_values": x[i:i + batch]})[0] for i in range(0, len(x), batch)])
