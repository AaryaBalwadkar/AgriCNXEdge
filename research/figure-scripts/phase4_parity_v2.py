"""Phase 4 parity v2 (CPU-only): laptop reference for bench_v2 accuracy set.

Runs bench_v2/manifest_v2.csv through adc_student_full.onnx (correct decode
@0.3) once per image. Output = expected phone values after the sigmoid fix.
"""
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Proposed Model" / "student" / "adc_student_full.onnx"
V2_DIR = ROOT / "AgriCNXEdge" / "app" / "src" / "main" / "assets" / "bench_v2"
OUT_DIR = ROOT / "parity_results"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 512
NMS_IOU = 0.45


def sig(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def softmax(a):
    a = a.astype(np.float64)
    a -= a.max()
    e = np.exp(a)
    return (e / e.sum()).astype(np.float32)


def iou1m(box, others):
    ax1, ay1 = box[0] - box[2] / 2, box[1] - box[3] / 2
    ax2, ay2 = box[0] + box[2] / 2, box[1] + box[3] / 2
    bx1, by1 = others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2
    bx2, by2 = others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2
    inter = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1)) * \
        np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    union = box[2] * box[3] + others[:, 2] * others[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def decode(raw, thr=0.3):
    det = raw[0].reshape(6, -1).transpose(1, 0)
    bs, sc = sig(det[:, :4]), sig(det[:, 4]) * sig(det[:, 5])
    ms = float(sc.max())
    m = sc > thr
    if not np.any(m):
        return 0, ms
    boxes, s = bs[m], sc[m]
    order = np.argsort(-s)
    boxes, s = boxes[order], s[order]
    kept = []
    for i in range(len(boxes)):
        if kept and np.any(iou1m(boxes[i], np.stack(kept)) > NMS_IOU):
            continue
        kept.append(boxes[i])
    return len(kept), ms


def pre(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None, ...].astype(np.float32)


def main():
    import onnxruntime as ort
    assert V2_DIR.exists(), f"missing {V2_DIR}"
    specs = list(csv.DictReader(open(V2_DIR / "manifest_v2.csv", encoding="utf-8")))
    print(f"v2 specs: {len(specs)}")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    db = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    di = np.array([0], dtype=np.int64)
    rows, t0 = [], time.perf_counter()
    for s in specs:
        img = Image.open(V2_DIR / s["image"])
        leaf, pest, raw = sess.run(
            ["leaf_logits", "pest_logits", "raw_det"],
            {"image": pre(img), "boxes": db, "box_idx": di})
        li, pi = int(np.argmax(softmax(leaf.ravel()))), int(np.argmax(softmax(pest.ravel())))
        n, ms = decode(raw)
        rows.append({"image": s["image"], "task": s["task"], "gt_count": s["gt_count"],
                     "gt_class": s["gt_class"], "exp_leaf": li, "exp_pest": pi,
                     "exp_pred": n, "max_score": round(ms, 4)})
        print(f"{s['image']} {s['task']} gt={s['gt_count']}/{s['gt_class']} -> pred={n} leaf={li} pest={pi} max={ms:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "parity_v2_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    y = [r for r in rows if r["task"] == "yield"]
    mae = sum(abs(r["exp_pred"] - int(r["gt_count"])) for r in y) / max(1, len(y))
    with open(OUT_DIR / "parity_v2_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"images={len(rows)} yield={len(y)} yield_MAE@0.3={mae:.3f} "
                 f"total_s={(time.perf_counter()-t0):.0f}\n")
    print(f"Wrote parity_v2_table.csv (yield MAE {mae:.3f})")


if __name__ == "__main__":
    main()
