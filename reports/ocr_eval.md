# OCR & extraction evaluation (150 documents, holdout split)

Field accuracy uses the true document type, so it measures OCR + extraction only.

| Engine | CER | WER | Field accuracy | OCR p50 (ms) | Preprocess p50 (ms) |
|---|---:|---:|---:|---:|---:|
| paddle_onnx_raw | 0.116 | 0.167 | 97.3% | 1817.6 | 1.0 |
| paddle_onnx+preprocess | 0.007 | 0.031 | 97.6% | 1554.7 | 28.6 |

## paddle_onnx_raw: by document type / capture

| Slice | Docs | CER | WER | Field accuracy |
|---|---:|---:|---:|---:|
| bank_statement | 30 | 0.194 | 0.272 | 98.9% |
| id_card | 30 | 0.017 | 0.030 | 100.0% |
| invoice | 30 | 0.162 | 0.231 | 98.9% |
| payslip | 30 | 0.158 | 0.193 | 97.8% |
| receipt | 30 | 0.047 | 0.110 | 91.1% |
| capture=scan | 96 | 0.122 | 0.167 | 99.3% |
| capture=photo | 54 | 0.105 | 0.167 | 93.8% |

## paddle_onnx+preprocess: by document type / capture

| Slice | Docs | CER | WER | Field accuracy |
|---|---:|---:|---:|---:|
| bank_statement | 30 | 0.005 | 0.046 | 96.7% |
| id_card | 30 | 0.010 | 0.022 | 100.0% |
| invoice | 30 | 0.005 | 0.018 | 98.9% |
| payslip | 30 | 0.001 | 0.013 | 100.0% |
| receipt | 30 | 0.017 | 0.057 | 92.2% |
| capture=scan | 96 | 0.005 | 0.028 | 98.6% |
| capture=photo | 54 | 0.011 | 0.037 | 95.7% |

Skipped (not installed on this host): tesseract
