# Classifier inference benchmark (v2)

Host: Intel64 Family 6 Model 141 Stepping 1, GenuineIntel · 12 logical cores · ONNX Runtime 1.31.1 · torch 2.15.0+cpu

| Config | Batch | p50 (ms) | p95 (ms) | ms / image | Throughput (img/s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|
| pytorch_eager | 1 | 23.84 | 29.68 | 23.84 | 40.7 | 672.3 |
| pytorch_eager | 2 | 37.58 | 41.93 | 18.79 | 52.8 | 672.3 |
| pytorch_eager | 4 | 70.38 | 76.79 | 17.6 | 56.6 | 672.3 |
| pytorch_eager | 8 | 148.94 | 155.98 | 18.62 | 53.6 | 672.3 |
| pytorch_eager | 16 | 319.35 | 336.77 | 19.96 | 49.7 | 672.3 |
| pytorch_eager | 32 | 664.85 | 692.6 | 20.78 | 48.4 | 672.3 |
| onnx_fp32_noopt | 1 | 15.06 | 16.99 | 15.06 | 70.0 | 396.9 |
| onnx_fp32_noopt | 2 | 41.61 | 47.92 | 20.8 | 47.4 | 396.9 |
| onnx_fp32_noopt | 4 | 71.16 | 78.65 | 17.79 | 56.5 | 396.9 |
| onnx_fp32_noopt | 8 | 138.64 | 143.45 | 17.33 | 58.5 | 396.9 |
| onnx_fp32_noopt | 16 | 265.84 | 274.85 | 16.62 | 60.2 | 396.9 |
| onnx_fp32_noopt | 32 | 525.72 | 537.05 | 16.43 | 61.0 | 396.9 |
| onnx_fp32_opt | 1 | 10.13 | 11.99 | 10.13 | 105.2 | 396.8 |
| onnx_fp32_opt | 2 | 21.63 | 23.27 | 10.81 | 92.5 | 396.8 |
| onnx_fp32_opt | 4 | 42.86 | 44.84 | 10.71 | 94.0 | 396.8 |
| onnx_fp32_opt | 8 | 84.17 | 89.73 | 10.52 | 94.3 | 396.8 |
| onnx_fp32_opt | 16 | 164.76 | 174.18 | 10.3 | 96.0 | 396.8 |
| onnx_fp32_opt | 32 | 337.03 | 343.13 | 10.53 | 95.6 | 396.8 |
| onnx_int8 | 1 | 2.93 | 3.46 | 2.93 | 330.3 | 183.5 |
| onnx_int8 | 2 | 5.43 | 6.45 | 2.72 | 369.4 | 183.5 |
| onnx_int8 | 4 | 9.9 | 11.45 | 2.48 | 406.6 | 183.5 |
| onnx_int8 | 8 | 17.86 | 26.75 | 2.23 | 408.7 | 183.5 |
| onnx_int8 | 16 | 48.71 | 57.39 | 3.04 | 322.7 | 183.5 |
| onnx_int8 | 32 | 93.87 | 103.62 | 2.93 | 337.8 | 183.5 |

## Summary

- Batch 1: ONNX Runtime (optimised) p50 10.13 ms vs PyTorch 23.84 ms (2.35x faster)
- Batch 32: ONNX Runtime (optimised) p50 337.03 ms vs PyTorch 664.85 ms (1.97x faster)
- Graph optimisations at batch 1: 15.06 -> 10.13 ms
- INT8 vs FP32 at batch 32: 337.8 vs 95.6 img/s
- Batching (ONNX FP32): throughput 105.2 img/s at batch 1 -> 95.6 img/s at batch 32, while p50 latency rises 10.13 -> 337.03 ms
