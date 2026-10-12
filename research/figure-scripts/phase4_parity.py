"""Phase 4 laptop parity check (CPU-only, no GPU, no phone).

Compares the exact 10 bench images through adc_student_full.onnx on this
laptop (onnxruntime CPU) against the A17 phone CSVs.

Outputs parity_results/parity_table.csv + parity_summary.txt
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "Proposed Model" / "student" / "adc_student_full.onnx"
FALLBACK_MODEL = ROOT / "AgriCNXEdge" / "app" / "src" / "main" / "assets" / "adc_student_full.onnx"
BENCH_DIR = ROOT / "AgriCNXEdge" / "app" / "src" / "main" / "assets" / "bench"
A17_DIR = ROOT / "A17"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZE = 512
NMS_IOU = 0.45


def sigmoid_correct(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)


def sigmoid_kotlin_mirror(x):
    # Mirrors Agrimodelrunner.kt:307  1/(1+exp(+x))  -- suspected inversion
    return 1.0 / (1.0 + np.exp(x.astype(np.float64))).astype(np.float32)


def softmax(a):
    a = a.astype(np.float64)
    a = a - a.max()
    e = np.exp(a)
    return (e / e.sum()).astype(np.float32)


def box_iou_one_vs_many(box, others):
    # box [4] cxcywh, others [N,4] cxcywh -> [N] IoU
    ax1, ay1 = box[0] - box[2] / 2, box[1] - box[3] / 2
    ax2, ay2 = box[0] + box[2] / 2, box[1] + box[3] / 2
    bx1, by1 = others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2
    bx2, by2 = others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = box[2] * box[3] + others[:, 2] * others[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def decode_numpy(raw_det, conf_thresh, sig_fn, nms_iou=NMS_IOU):
    # raw_det [1,6,32,32] -> boxes [K,4] cxcywh, scores [K]
    det = raw_det[0]  # [6,32,32]
    flat = det.reshape(6, -1).transpose(1, 0)  # [1024,6]
    box_sig = sig_fn(flat[:, :4])
    obj = sig_fn(flat[:, 4])
    cls = sig_fn(flat[:, 5])
    scores = obj * cls
    max_score = float(scores.max()) if scores.size else 0.0
    mask = scores > conf_thresh
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), max_score
    boxes = box_sig[mask]
    sc = scores[mask]
    order = np.argsort(-sc)
    boxes, sc = boxes[order], sc[order]
    kept_b, kept_s = [], []
    for i in range(len(boxes)):
        b = boxes[i]
        if kept_b:
            ious = box_iou_one_vs_many(b, np.stack(kept_b))
            if np.any(ious > nms_iou):
                continue
        kept_b.append(b)
        kept_s.append(sc[i])
    if not kept_b:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), max_score
    return np.stack(kept_b), np.array(kept_s, dtype=np.float32), max_score


def preprocess(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # [1,3,512,512]


def load_phone_map():
    phone = {}
    for f in sorted(A17_DIR.glob("bench_*.csv")):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                # both modes give identical preds; keep first seen per image
                if row["image"] not in phone:
                    phone[row["image"]] = {
                        "leaf": int(row["leaf"]),
                        "pest": int(row["pest"]),
                        "pred": int(float(row["pred_count"])),
                        "gt": int(float(row["gt_count"])),
                        "src": f.name,
                    }
    return phone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(ROOT / "parity_results"))
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and FALLBACK_MODEL.exists():
        model_path = FALLBACK_MODEL
    assert model_path.exists(), f"Model not found: {model_path}"
    assert BENCH_DIR.exists(), f"Bench dir missing: {BENCH_DIR}"

    import onnxruntime as ort

    manifest = {}
    with open(BENCH_DIR / "manifest.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            manifest[row["image"].strip()] = int(row["gt_count"])
    assert len(manifest) == 10, f"Expected 10 bench images, got {len(manifest)}"

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Model: {model_path} ({model_path.stat().st_size/1048576:.1f} MB)")
    print(f"Inputs: {in_names}  Outputs: {out_names}")
    assert in_names == ["image", "boxes", "box_idx"], f"Unexpected inputs: {in_names}"

    dummy_boxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
    dummy_idx = np.array([0], dtype=np.int64)
    phone = load_phone_map()
    print(f"Phone rows loaded for {len(phone)} images from A17/")

    thresholds = [0.3, 0.1, 0.05, 0.01]
    rows = []
    t_total = 0.0
    for img_name, gt in sorted(manifest.items()):
        img = Image.open(BENCH_DIR / img_name)
        inp = preprocess(img)
        t0 = time.perf_counter()
        leaf, pest, raw = sess.run(
            ["leaf_logits", "pest_logits", "raw_det"],
            {"image": inp, "boxes": dummy_boxes, "box_idx": dummy_idx},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        t_total += dt
        assert raw.shape == (1, 6, 32, 32), f"Unexpected raw_det shape {raw.shape}"
        lp, pp = softmax(leaf.ravel()), softmax(pest.ravel())
        li, pi = int(np.argmax(lp)), int(np.argmax(pp))
        counts = {}
        max_s = 0.0
        for th in thresholds:
            _, sc, ms = decode_numpy(raw, th, sigmoid_correct)
            counts[th] = len(sc)
            if th == 0.3:
                max_s = ms
        _, sc_inv, _ = decode_numpy(raw, 0.3, sigmoid_kotlin_mirror)
        ph = phone.get(img_name, {})
        rows.append({
            "image": img_name, "gt": gt,
            "phone_leaf": ph.get("leaf", -1), "phone_pest": ph.get("pest", -1),
            "phone_pred": ph.get("pred", -1),
            "laptop_leaf": li, "laptop_pest": pi,
            "leaf_match": int(ph.get("leaf", -99) == li),
            "pest_match": int(ph.get("pest", -99) == pi),
            "max_score": round(max_s, 4),
            "pred_0.3": counts[0.3], "pred_0.1": counts[0.1],
            "pred_0.05": counts[0.05], "pred_0.01": counts[0.01],
            "kotlin_inv_0.3": len(sc_inv),
            "laptop_ms": round(dt, 1),
        })
        print(f"{img_name} gt={gt} phone_pred={ph.get('pred')} lap0.3={counts[0.3]} "
              f"0.1={counts[0.1]} 0.05={counts[0.05]} 0.01={counts[0.01]} "
              f"max={max_s:.4f} kotlin_inv={len(sc_inv)} leaf {ph.get('leaf')}->{li} "
              f"pest {ph.get('pest')}->{pi} ({dt:.0f}ms)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "parity_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mae(key):
        return sum(abs(r[key] - r["gt"]) for r in rows) / len(rows)

    with open(out_dir / "parity_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"model={model_path} size_mb={model_path.stat().st_size/1048576:.2f}\n")
        fh.write(f"images=10 laptop_total_ms={t_total:.0f} laptop_mean_ms={t_total/len(rows):.0f}\n")
        fh.write(f"phone_MAE(all-zero)={sum(r['gt'] for r in rows)/len(rows):.3f}\n")
        for th, k in [(0.3, "pred_0.3"), (0.1, "pred_0.1"), (0.05, "pred_0.05"), (0.01, "pred_0.01")]:
            fh.write(f"laptop_MAE@{th}={mae(k):.3f}\n")
        fh.write(f"leaf_match={sum(r['leaf_match'] for r in rows)}/10 "
                 f"pest_match={sum(r['pest_match'] for r in rows)}/10\n")
        fh.write("max_score per image = max(obj*cls) with CORRECT sigmoid; "
                 "kotlin_inv_0.3 = count with inverted 1/(1+exp(+x)) at 0.3\n")
    print(f"\nWrote {out_dir/'parity_table.csv'} and {out_dir/'parity_summary.txt'}")


if __name__ == "__main__":
    main()
