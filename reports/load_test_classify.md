# HTTP load test

| Scenario | Endpoint | Requests | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Success | Mean server batch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batching_off | classify | 1000 | 1 | 17.8 | 54.8 | 69.1 | 82.3 | 100.0% | 1.0 |
| batching_off | classify | 1000 | 8 | 37.2 | 211.3 | 278.8 | 312.3 | 100.0% | 1.0 |
| batching_off | classify | 1000 | 32 | 42.5 | 746.4 | 825.7 | 1013.9 | 100.0% | 1.0 |
| batching_off | classify | 1000 | 64 | 38.7 | 1649.0 | 1889.6 | 2264.7 | 100.0% | 1.0 |
| batching_on | classify | 1000 | 1 | 13.7 | 74.4 | 93.0 | 113.5 | 100.0% | 1.0 |
| batching_on | classify | 1000 | 8 | 32.9 | 233.2 | 347.1 | 399.0 | 100.0% | 1.05 |
| batching_on | classify | 1000 | 32 | 31.6 | 946.4 | 1345.5 | 1512.2 | 100.0% | 1.0 |
| batching_on | classify | 1000 | 64 | 26.9 | 2268.1 | 3276.5 | 3534.9 | 100.0% | 1.0 |
