"""Green-apple overlays + detector confidence maps (CPU-only, laptop).

For bench_v2 yield-val images: runs adc_student_full.onnx, decodes boxes
(correct sigmoid, conf 0.3, NMS 0.45), draws GT (green solid) vs pred
(orange dashed) on the 512 model-input image, and saves the 32x32
detector-confidence grid (max obj x cls per cell).

Outputs paper_figures/green_ov/<stem>.png + <stem>_conf.png
+ green_overlay_data.json
"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Proposed Model" / "student" / "adc_student_full.onnx"
V2 = ROOT / "AgriCNXEdge" / "app" / "src" / "main" / "assets" / "bench_v2"
YOLO_IMG = ROOT / "YOLO26s apple object detection model 1" / "data" / "yield" / "images" / "val"
YOLO_LBL = ROOT / "YOLO26s apple object detection model 1" / "data" / "yield" / "labels" / "val"
OUT = ROOT / "paper_figures" / "green_ov"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 512


def sig(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def decode(raw, thr=0.3, iou=0.45):
    det = raw[0].reshape(6, -1).transpose(1, 0)
    bs, sc = sig(det[:, :4]), sig(det[:, 4]) * sig(det[:, 5])
    grid = sc.reshape(32, 32)
    m = sc > thr
    if not np.any(m):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), grid
    boxes, s = bs[m], sc[m]
    order = np.argsort(-s)
    boxes, s = boxes[order], s[order]
    kept_b, kept_s = [], []
    for i in range(len(boxes)):
        b = boxes[i]
        drop = False
        for k in kept_b:
            ax1, ay1 = b[0] - b[2] / 2, b[1] - b[3] / 2
            ax2, ay2 = b[0] + b[2] / 2, b[1] + b[3] / 2
            bx1, by1 = k[0] - k[2] / 2, k[1] - k[3] / 2
            bx2, by2 = k[0] + k[2] / 2, k[1] + k[3] / 2
            inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            union = b[2] * b[3] + k[2] * k[3] - inter
            if union > 0 and inter / union > iou:
                drop = True
                break
        if not drop:
            kept_b.append(b)
            kept_s.append(s[i])
    if not kept_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), grid
    return np.stack(kept_b), np.array(kept_s, np.float32), grid


def pre(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)[None, ...].astype(np.float32)


def gt_boxes(stem):
    boxes = []
    p = YOLO_LBL / (stem + ".txt")
    if not p.exists():
        return boxes
    for line in open(p, encoding="utf-8", errors="ignore"):
        parts = line.split()
        if len(parts) >= 5:
            boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def main():
    import onnxruntime as ort
    specs = [r for r in csv.DictReader(open(V2 / "manifest_v2.csv", encoding="utf-8")) if r["task"] == "yield"]
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    db = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    di = np.array([0], dtype=np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in specs:
        name = s["image"]
        stem = Path(name).stem
        src = YOLO_IMG / name
        if not src.exists():
            print(f"SKIP (no image): {name}")
            continue
        img = Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        leaf, pest, raw = sess.run(
            ["leaf_logits", "pest_logits", "raw_det"],
            {"image": pre(Image.open(src)), "boxes": db, "box_idx": di})
        boxes, scores, grid = decode(raw)
        d = ImageDraw.Draw(img)
        for (cx, cy, w, h) in gt_boxes(stem):
            x1, y1, x2, y2 = (cx - w / 2) * SIZE, (cy - h / 2) * SIZE, (cx + w / 2) * SIZE, (cy + h / 2) * SIZE
            d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        for (cx, cy, w, h) in boxes:
            x1, y1, x2, y2 = (cx - w / 2) * SIZE, (cy - h / 2) * SIZE, (cx + w / 2) * SIZE, (cy + h / 2) * SIZE
            d.rectangle([x1, y1, x2, y2], outline=(230, 110, 0), width=2)
            # dashed effect: overlay white gaps
        img.save(OUT / f"{stem}.png")
        g = (grid / max(1e-9, grid.max()) * 255).astype(np.uint8)
        Image.fromarray(g).resize((SIZE, SIZE), Image.NEAREST).save(OUT / f"{stem}_conf.png")
        flat = raw[0].reshape(6, -1).transpose(1, 0)
        ms = round(float((sig(flat[:, 4]) * sig(flat[:, 5])).max()), 4)
        recs.append({"image": name, "gt": len(gt_boxes(stem)), "pred": len(boxes),
                     "max_score": ms})
        print(f"{name} gt={recs[-1]['gt']} pred={recs[-1]['pred']}")
    json.dump(recs, open(OUT / "green_overlay_data.json", "w"), indent=1)
    print(f"wrote {len(recs)} overlays + conf maps")


if __name__ == "__main__":
    main()
