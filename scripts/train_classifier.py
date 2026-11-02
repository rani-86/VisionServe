"""Fine-tune a Hugging Face image classifier on document images and register a new model version.

    python scripts/train_classifier.py --version v1 --mode linear_probe --epochs 3
    python scripts/train_classifier.py --version v2 --mode finetune --epochs 5

Images go through the same OpenCV preprocessing the service uses (no train/serve
skew); letterboxed 224x224 inputs are cached once, then augmented on the fly.
Writes models/doc_classifier/<version>/ with the HF checkpoint, FP32 + INT8
ONNX exports and metadata.json (labels, normalisation, val/test metrics, parity).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from docserve.export import export_onnx, onnx_predict, parity, quantize_int8  # noqa: E402
from docserve.preprocessing import classifier_tensor, preprocess  # noqa: E402
from docserve.synth import DOC_TYPES  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_split(data_dir: Path, split: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess every image in a split to a letterboxed uint8 canvas (cached as .npz)."""
    cache = data_dir / f".cache_{split}_{size}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"]
    rows = [json.loads(l) for l in open(data_dir / split / "labels.jsonl", encoding="utf-8")]

    def one(r):
        gray = preprocess(cv2.imread(str(data_dir / split / r["path"]))).gray
        t = classifier_tensor(gray, size, [0, 0, 0], [1, 1, 1])[0]  # unnormalised [0,1]
        return (t * 255).astype(np.uint8)

    with ThreadPoolExecutor(8) as ex:
        x = np.stack(list(ex.map(one, rows)))
    y = np.array([DOC_TYPES.index(r["doc_type"]) for r in rows])
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def normalize(x_u8: np.ndarray) -> np.ndarray:
    x = np.repeat((x_u8.astype(np.float32) / 255.0)[:, None], 3, axis=1)
    return (x - np.array(MEAN, np.float32)[None, :, None, None]) / np.array(STD, np.float32)[None, :, None, None]


def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small geometric + photometric jitter on top of the already-degraded scans."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), rng.uniform(0.92, 1.05))
    M[:, 2] += [rng.uniform(-8, 8), rng.uniform(-8, 8)]
    out = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.1) + rng.uniform(-15, 15), 0, 255)
    return out.astype(np.uint8)


def evaluate(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(DOC_TYPES)))).tolist()}


def torch_logits(model, x: np.ndarray, bs: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), bs):
            out.append(model(pixel_values=torch.from_numpy(x[i:i + bs])).logits.numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--models", default=str(ROOT / "models" / "doc_classifier"))
    ap.add_argument("--version", required=True)
    ap.add_argument("--base-model", default="microsoft/resnet-18")
    ap.add_argument("--mode", choices=["linear_probe", "finetune"], default="finetune")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForImageClassification

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    data = Path(args.data)
    t0 = time.time()
    xtr, ytr = load_split(data, "train", args.size)
    xva, yva = load_split(data, "val", args.size)
    xte, yte = load_split(data, "test", args.size)
    print(f"loaded train={len(xtr)} val={len(xva)} test={len(xte)} in {time.time() - t0:.0f}s")

    model = AutoModelForImageClassification.from_pretrained(
        args.base_model, num_labels=len(DOC_TYPES), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(DOC_TYPES)), label2id={l: i for i, l in enumerate(DOC_TYPES)})
    if args.mode == "linear_probe":  # baseline: frozen ImageNet backbone, train only the head
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr if args.mode == "finetune" else args.lr * 10, weight_decay=1e-4)
    steps = args.epochs * ((len(xtr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = random.Random(args.seed)
    xva_n, xte_n = normalize(xva), normalize(xte)
    history, best = [], (-1.0, None)
    for epoch in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(xtr))
        tl, te = 0.0, time.time()
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            xb = torch.from_numpy(normalize(np.stack([augment(xtr[j], rng) for j in b])))
            loss = loss_fn(model(pixel_values=xb).logits, torch.from_numpy(ytr[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
        val = evaluate(torch_logits(model, xva_n), yva)
        history.append({"epoch": epoch + 1, "train_loss": round(tl / len(xtr), 4), "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "seconds": round(time.time() - te, 1)})
        print(history[-1])
        if val["macro_f1"] > best[0]:
            best = (val["macro_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])  # keep the best-on-validation epoch; test is only touched once

    out = Path(args.models) / args.version
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "hf")
    val_m = evaluate(torch_logits(model, xva_n), yva)
    test_m = evaluate(torch_logits(model, xte_n), yte)

    fp32 = export_onnx(model, out / "model.onnx", args.size)
    max_diff = parity(model, fp32, xte_n[:16])
    onnx_test = evaluate(onnx_predict(fp32, xte_n), yte)
    metrics = {"val": val_m, "test": test_m, "test_onnx_fp32": onnx_test, "onnx_max_abs_logit_diff": max_diff}
    if not args.no_int8:
        calib = normalize(xtr[np.random.RandomState(0).choice(len(xtr), 64, replace=False)])
        int8 = quantize_int8(fp32, out / "model.int8.onnx", calib)
        metrics["test_onnx_int8"] = evaluate(onnx_predict(int8, xte_n), yte)
        metrics["size_mb"] = {"fp32": round(fp32.stat().st_size / 1e6, 2), "int8": round(int8.stat().st_size / 1e6, 2)}

    meta = {
        "name": "doc_classifier", "version": args.version, "labels": DOC_TYPES, "input_size": args.size,
        "mean": MEAN, "std": STD, "base_model": args.base_model, "training_mode": args.mode,
        "epochs": args.epochs, "seed": args.seed, "train_size": int(len(xtr)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history, "metrics": metrics,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in metrics if "confusion" not in k}, indent=2))
    print(f"saved {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
