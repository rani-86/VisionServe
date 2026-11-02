"""Generate the synthetic train/val/test document sets and a leakage report.

    python scripts/generate_data.py --train 160 --val 40 --test 40
    python scripts/generate_data.py --only holdout --holdout 30

Writes data/<split>/<doc_type>/<id>.jpg plus data/<split>/labels.jsonl
(doc type, capture mode, key-value fields, ground-truth text lines).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docserve.synth import DOC_TYPES, DocumentGenerator  # noqa: E402

PERSON_FIELDS = {"name"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=160, help="documents per type")
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=0, help="extra untouched split for final OCR reporting")
    ap.add_argument("--only", nargs="+", help="generate just these splits (e.g. --only holdout)")
    args = ap.parse_args()
    out = Path(args.out)

    hashes: dict[str, set[str]] = {}
    entities: dict[str, set[str]] = {}
    ids: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    plan = [("train", args.train), ("val", args.val), ("test", args.test), ("holdout", args.holdout)]
    plan = [(s, n) for s, n in plan if n and (not args.only or s in args.only)]
    for split, n in plan:
        gen = DocumentGenerator(split)
        hashes[split], entities[split], ids[split] = set(), set(), set()
        counts[split] = {}
        rows = []
        for doc_type in DOC_TYPES:
            d = out / split / doc_type
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                s = gen.generate(doc_type)
                name = f"{doc_type}_{i:04d}.jpg"
                ok, buf = cv2.imencode(".jpg", s.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                data = buf.tobytes()
                (d / name).write_bytes(data)
                hashes[split].add(hashlib.sha256(data).hexdigest())
                for k, v in s.fields.items():
                    (entities if k in PERSON_FIELDS else ids)[split].add(v)
                rows.append({"path": f"{doc_type}/{name}", "doc_type": doc_type, "capture": s.capture,
                             "fields": s.fields, "lines": s.lines})
            counts[split][doc_type] = n
        with open(out / split / "labels.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} documents")

    # Re-read splits generated in earlier runs so every pair is checked.
    for split in ("train", "val", "test", "holdout"):
        if split in hashes or not (out / split / "labels.jsonl").exists():
            continue
        hashes[split] = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / split).glob("*/*.jpg")}
        entities[split], ids[split] = set(), set()
        for line in open(out / split / "labels.jsonl", encoding="utf-8"):
            for k, v in json.loads(line)["fields"].items():
                (entities if k in PERSON_FIELDS else ids)[split].add(v)
    pairs = [(a, b) for i, a in enumerate(sorted(hashes)) for b in sorted(hashes)[i + 1:]]

    def overlap(d: dict[str, set[str]], a: str, b: str) -> int:
        return len(d[a] & d[b])

    report = {
        "counts": counts,
        "duplicate_images_across_splits": {f"{a}-{b}": overlap(hashes, a, b) for a, b in pairs},
        "shared_person_names": {f"{a}-{b}": overlap(entities, a, b) for a, b in pairs},
        "shared_field_values": {f"{a}-{b}": overlap(ids, a, b) for a, b in pairs},
        "note": "Person names come from disjoint syllable pools per split. Field values (amounts, dates, ids) "
                "are random; a few short values such as dates can collide by chance.",
    }
    prev = out / "split_report.json"
    if args.only and prev.exists():
        report["counts"] = json.loads(prev.read_text())["counts"] | counts
    prev.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if any(report["duplicate_images_across_splits"].values()) or any(report["shared_person_names"].values()):
        raise SystemExit("leakage detected between splits")


if __name__ == "__main__":
    main()
