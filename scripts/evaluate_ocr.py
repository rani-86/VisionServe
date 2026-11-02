"""OCR + extraction evaluation on a held-out split.

    python scripts/evaluate_ocr.py --split val --per-type 40 --engines paddle_onnx tesseract
    python scripts/evaluate_ocr.py --per-type 6 --engines paddle --modes preprocess --name ocr_eval_paddle_native

The default split is `val`: extraction rules were tuned by error analysis on `test`
pages, so `val` is the untouched set for reporting.

For every available engine, with and without OpenCV preprocessing:
  - page-level CER / WER against the ground-truth text lines
  - field-level exact-match accuracy (after normalisation), using the TRUE doc type so
    OCR quality is measured separately from the classifier
  - per document type and per capture mode (scan vs phone photo) breakdowns, OCR latency
Writes reports/<name>.{json,md} including every field miss for error analysis.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docserve.extraction import extract_fields, normalize_truth  # noqa: E402
from docserve.ocr import available_engines, get_engine, lines_to_rows  # noqa: E402
from docserve.preprocessing import preprocess  # noqa: E402
from docserve.text_metrics import cer, wer  # noqa: E402


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {}
    fields = [f for r in recs for f in r["field_hits"]]
    return {"documents": len(recs),
            "cer": round(float(np.mean([r["cer"] for r in recs])), 4),
            "wer": round(float(np.mean([r["wer"] for r in recs])), 4),
            "field_accuracy": round(float(np.mean(fields)), 4) if fields else None,
            "fields": len(fields),
            "ocr_ms_p50": round(float(np.percentile([r["ocr_ms"] for r in recs], 50)), 1),
            "preprocess_ms_p50": round(float(np.percentile([r["pre_ms"] for r in recs], 50)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--engines", nargs="+", default=["paddle_onnx", "paddle", "tesseract"])
    ap.add_argument("--split", default="val", help="val is untouched by extraction-rule tuning")
    ap.add_argument("--modes", nargs="+", choices=["raw", "preprocess"], default=["raw", "preprocess"])
    ap.add_argument("--out", default=str(ROOT / "reports"))
    ap.add_argument("--name", default="ocr_eval")
    args = ap.parse_args()

    data = Path(args.data)
    rows = [json.loads(l) for l in open(data / args.split / "labels.jsonl", encoding="utf-8")]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["doc_type"]].append(r)
    rng = random.Random(0)
    sample = [r for t in sorted(by_type) for r in rng.sample(by_type[t], min(args.per_type, len(by_type[t])))]

    usable = available_engines(args.engines)
    skipped = sorted(set(args.engines) - set(usable))
    if skipped:
        print(f"skipping unavailable engines: {skipped}")
    results, worst, misses = {}, {}, {}
    for engine_name in usable:
        engine = get_engine(engine_name)
        for pre_on in [m == "preprocess" for m in args.modes]:
            key = f"{engine_name}{'+preprocess' if pre_on else '_raw'}"
            recs = []
            for i, r in enumerate(sample):
                img = cv2.imread(str(data / args.split / r["path"]))
                t0 = time.perf_counter()
                pre = preprocess(img, enabled=pre_on)
                t1 = time.perf_counter()
                lines = engine.recognize(pre.gray, pre.binary)
                t2 = time.perf_counter()
                hyp_rows = lines_to_rows(lines)
                ref, hyp = "\n".join(r["lines"]), "\n".join(hyp_rows)
                pred = extract_fields(r["doc_type"], hyp_rows)
                hits = [pred.get(k) == normalize_truth(r["doc_type"], k, v) for k, v in r["fields"].items()]
                recs.append({"path": r["path"], "doc_type": r["doc_type"], "capture": r["capture"],
                             "cer": cer(ref, hyp), "wer": wer(ref, hyp), "field_hits": hits,
                             "pre_ms": (t1 - t0) * 1000, "ocr_ms": (t2 - t1) * 1000,
                             "misses": {k: {"expected": normalize_truth(r["doc_type"], k, v), "got": pred.get(k)}
                                        for (k, v), h in zip(r["fields"].items(), hits) if not h}})
                if (i + 1) % 20 == 0:
                    print(f"{key}: {i + 1}/{len(sample)}", flush=True)
            results[key] = {
                "overall": summarize(recs),
                "by_doc_type": {t: summarize([x for x in recs if x["doc_type"] == t]) for t in sorted(by_type)},
                "by_capture": {c: summarize([x for x in recs if x["capture"] == c]) for c in ("scan", "photo")},
            }
            worst[key] = sorted(recs, key=lambda x: -x["cer"])[:5]
            misses[key] = [{"path": x["path"], "capture": x["capture"], **{"field": k, **v}}
                           for x in recs for k, v in x["misses"].items()]
            print(key, json.dumps(results[key]["overall"]), flush=True)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    report = {"split": args.split, "documents": len(sample), "engines_skipped": skipped, "results": results,
              "worst_pages": {k: [{kk: v[kk] for kk in ("path", "doc_type", "capture", "cer", "misses")} for v in vs]
                              for k, vs in worst.items()},
              "field_misses": misses}
    (out / f"{args.name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [f"# OCR & extraction evaluation ({len(sample)} documents, {args.split} split)", "",
          "Field accuracy uses the true document type, so it measures OCR + extraction only.", "",
          "| Engine | CER | WER | Field accuracy | OCR p50 (ms) | Preprocess p50 (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in results.items():
        o = v["overall"]
        md.append(f"| {k} | {o['cer']:.3f} | {o['wer']:.3f} | {o['field_accuracy'] * 100:.1f}% | {o['ocr_ms_p50']} | "
                  f"{o['preprocess_ms_p50']} |")
    for k, v in results.items():
        md += ["", f"## {k}: by document type / capture", "", "| Slice | Docs | CER | WER | Field accuracy |",
               "|---|---:|---:|---:|---:|"]
        for name, s in list(v["by_doc_type"].items()) + [(f"capture={c}", s) for c, s in v["by_capture"].items()]:
            if s:
                md.append(f"| {name} | {s['documents']} | {s['cer']:.3f} | {s['wer']:.3f} | "
                          f"{s['field_accuracy'] * 100:.1f}% |")
    if skipped:
        md += ["", f"Skipped (not installed on this host): {', '.join(skipped)}"]
    (out / f"{args.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
