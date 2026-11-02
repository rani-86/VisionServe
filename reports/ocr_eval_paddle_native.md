# OCR & extraction evaluation (20 documents, val split)

Field accuracy uses the true document type, so it measures OCR + extraction only.

| Engine | CER | WER | Field accuracy | OCR p50 (ms) | Preprocess p50 (ms) |
|---|---:|---:|---:|---:|---:|
| paddle+preprocess | 0.007 | 0.057 | 98.3% | 5407.8 | 56.0 |

## paddle+preprocess: by document type / capture

| Slice | Docs | CER | WER | Field accuracy |
|---|---:|---:|---:|---:|
| bank_statement | 4 | 0.003 | 0.030 | 100.0% |
| id_card | 4 | 0.004 | 0.067 | 100.0% |
| invoice | 4 | 0.008 | 0.053 | 100.0% |
| payslip | 4 | 0.000 | 0.000 | 100.0% |
| receipt | 4 | 0.019 | 0.135 | 91.7% |
| capture=scan | 15 | 0.008 | 0.068 | 97.8% |
| capture=photo | 5 | 0.002 | 0.025 | 100.0% |
