"""
run_yield_teacher.py
====================
Trains YOLO26s teacher for apple detection (yield estimation task).

Pipeline:
  1. Writes yield_dataset.yaml dynamically (correct absolute paths)
  2. Optional: adds COCO background negatives to fix false positives
  3. Fine-tunes YOLO26s on yield dataset (100 epochs, imgsz=512)
  4. Generates bounding-box soft labels for ALL training images → CSV
     (student reads this CSV during distillation)

OUTPUTS in outputs/yield/:
  yield_teacher/weights/best.pt    ← best YOLO checkpoint
  yield_soft_labels.csv            ← teacher predictions per training image
  yield_teacher_summary.json       ← mAP metrics + paths

USAGE:
  pip install ultralytics
  python run_yield_teacher.py

DATA STRUCTURE EXPECTED:
  data/yield/
    images/train/   (1093 images)
    labels/train/   (YOLO .txt annotations)
    images/val/     (264 images)
    labels/val/
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path

import numpy as np
import argparse
import random
import time
from datetime import datetime
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed.")
    print("        Run: pip install ultralytics")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_ROOT     = BASE_DIR / "data"  / "yield"
OUTPUT_DIR    = BASE_DIR / "outputs" / "yield"
YAML_PATH     = OUTPUT_DIR / "yield_dataset.yaml"
SOFT_CSV      = OUTPUT_DIR / "yield_soft_labels.csv"
SUMMARY_JSON  = OUTPUT_DIR / "yield_teacher_summary.json"

# ── Hyper-parameters (Updated for Camouflage & Tiny Targets) ───────────
MODEL_NAME    = "yolo26s.pt"   
EPOCHS        = 1000           
IMGSZ         = 640            # INCREASED: 640x640 retains tiny red dot features
BATCH         = 16             
PATIENCE      = 100            
LR0           = 0.001
LRF           = 0.01           
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 5              
CONF_THRESH   = 0.20           # LOWERED: Captures subtle green camouflaged soft labels
IOU_THRESH    = 0.45           
NC            = 1              
CLASS_NAMES   = ["apple"]

# ===========================================================================
#  AGRICULTURAL PRE-FILTER: LAB Color Decoupling & Spherical Contrast
# ===========================================================================
import cv2

# CLAHE / enhancement defaults (exposed via CLI)
CLAHE_COLOR_CLIP = 3.5
CLAHE_LIGHT_CLIP = 1.5
CLAHE_TILE_GRID = (8, 8)

def apply_agricultural_lab_filter(img_path: Path, output_path: Path) -> None:
    """
    Applies CIELAB a* channel CLAHE enhancement to break green fruit camouflage 
    and highlight tiny red dot targets. Saves enhanced image to output_path.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return

    # Convert to CIELAB space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # CLAHE on a* channel (Red-Green chromatic axis)
    clahe_color = cv2.createCLAHE(clipLimit=CLAHE_COLOR_CLIP, tileGridSize=CLAHE_TILE_GRID)
    a_enhanced = clahe_color.apply(a)

    # Mild CLAHE on L channel (Shadow/Lighting normalization)
    clahe_light = cv2.createCLAHE(clipLimit=CLAHE_LIGHT_CLIP, tileGridSize=CLAHE_TILE_GRID)
    L_enhanced = clahe_light.apply(L)

    # Recombine and convert back to BGR image space
    lab_enhanced = cv2.merge([L_enhanced, a_enhanced, b])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(output_path), bgr_enhanced)


def apply_edge_texture_filter(img_path: Path, output_path: Path, mode: str = "edge") -> None:
    """
    Compute an edge/texture enhancement and inject it into the image's L channel
    (CIELAB) so shape/edge cues are emphasized while preserving color.

    mode: 'edge' = edge-only enhancement
          'lab_edge' = CLAHE on a*/L plus edge injection
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Sobel magnitude as texture/edge map
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = cv2.GaussianBlur(mag, (3, 3), 0)
    mag_uint8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert to LAB and optionally apply CLAHE like the LAB filter
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    if mode == "lab_edge":
        # apply CLAHE on a* and L (mild)
        clahe_color = cv2.createCLAHE(clipLimit=CLAHE_COLOR_CLIP, tileGridSize=CLAHE_TILE_GRID)
        a = clahe_color.apply(a)
        clahe_light = cv2.createCLAHE(clipLimit=CLAHE_LIGHT_CLIP, tileGridSize=CLAHE_TILE_GRID)
        L = clahe_light.apply(L)

    # Inject edges into L channel (weighted add)
    edge_weight = 0.6
    L_f = L.astype(np.float32) + (mag_uint8.astype(np.float32) * edge_weight)
    L_new = np.clip(L_f, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([L_new, a, b])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(output_path), bgr_enhanced)


def preprocess_dataset_images(enhancement_mode: str = "lab"):
    """
    Preprocesses train and val splits to apply LAB chromatic enhancement.
    Creates an enhanced dataset folder used for teacher training.
    """
    print("\n[INFO] Preprocessing dataset with LAB Chromatic Enhancement...")
    enhanced_root = DATA_ROOT.parent / "yield_enhanced"

    accepted_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in ["train", "val"]:
        src_img_dir = DATA_ROOT / "images" / split
        dst_img_dir = enhanced_root / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        # Prepare label dirs (we will copy labels that correspond to processed images)
        src_lbl_dir = DATA_ROOT / "labels" / split
        dst_lbl_dir = enhanced_root / "labels" / split
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Process each image we can find (case-insensitive suffix match)
        for img_path in sorted(src_img_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in accepted_exts:
                continue

            out_path = dst_img_dir / img_path.name
            try:
                if enhancement_mode == "lab":
                    apply_agricultural_lab_filter(img_path, out_path)
                elif enhancement_mode == "edge":
                    apply_edge_texture_filter(img_path, out_path, mode="edge")
                elif enhancement_mode == "lab_edge":
                    apply_edge_texture_filter(img_path, out_path, mode="lab_edge")
                else:
                    # default to lab
                    apply_agricultural_lab_filter(img_path, out_path)
            except Exception as e:
                print(f"[WARN] Failed to enhance {img_path.name}: {e}")
                continue

            # If enhancement succeeded (file created), copy corresponding label if present
            if out_path.exists():
                src_label = src_lbl_dir / (img_path.stem + ".txt")
                if src_label.exists():
                    shutil.copy(src_label, dst_lbl_dir / src_label.name)
            else:
                print(f"[WARN] Enhanced image not written for {img_path.name}; skipping its label copy.")

    print(f"[INFO] ✓ Enhanced dataset prepared at: {enhanced_root}  (mode={enhancement_mode})")
    return enhanced_root


def verify_dataset_consistency(data_root: Path) -> None:
    """Verify images ↔ labels correspondence and report mismatches.

    Prints counts and up to 10 mismatched filenames per split.
    """
    for split in ["train", "val"]:
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

        imgs = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbls = {p.stem for p in lbl_dir.iterdir() if p.is_file()}

        imgs_without_lbl = sorted(list(imgs - lbls))[:10]
        lbls_without_img = sorted(list(lbls - imgs))[:10]

        print(f"\n[VERIFY] {split}: images={len(imgs)}, labels={len(lbls)}")
        if imgs_without_lbl:
            print(f"  [WARN] {len(imgs_without_lbl)} images without label (showing up to 10): {imgs_without_lbl}")
        if lbls_without_img:
            print(f"  [WARN] {len(lbls_without_img)} labels without image (showing up to 10): {lbls_without_img}")
        if not imgs_without_lbl and not lbls_without_img:
            print("  [OK] images and labels appear consistent for this split.")


def create_mixed_dataset(enhanced_root: Path) -> Path:
    """Create a mixed dataset that contains both original and enhanced images.

    Enhanced images are added with suffix `_enh` to their stem so labels are
    duplicated with matching stems. Returns the mixed dataset root.
    """
    mixed_root = DATA_ROOT.parent / "yield_mixed"
    for split in ["train", "val"]:
        img_src = DATA_ROOT / "images" / split
        lbl_src = DATA_ROOT / "labels" / split
        img_enh = enhanced_root / "images" / split

        img_dst = mixed_root / "images" / split
        lbl_dst = mixed_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        # Copy originals
        for p in sorted(img_src.iterdir()):
            if not p.is_file():
                continue
            shutil.copy(p, img_dst / p.name)
            lbl = lbl_src / (p.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_dst / lbl.name)

        # Copy enhanced with new name suffix
        for p in sorted(img_enh.iterdir()):
            if not p.is_file():
                continue
            new_name = f"{p.stem}_enh{p.suffix}"
            shutil.copy(p, img_dst / new_name)
            # copy label (same annotations)
            lbl = enhanced_root / "labels" / split / (p.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_dst / (p.stem + "_enh.txt"))

    print(f"[INFO] Mixed dataset prepared at: {mixed_root}")
    return mixed_root


def save_visual_samples(original_root: Path, enhanced_root: Path, out_dir: Path, n: int = 20) -> None:
    """Save side-by-side visuals of original vs enhanced images (train split).

    Saves up to `n` images to out_dir/visuals.
    """
    vis_dir = out_dir / "visuals"
    vis_dir.mkdir(parents=True, exist_ok=True)
    src_dir = original_root / "images" / "train"
    enh_dir = enhanced_root / "images" / "train"

    pairs = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        enh_p = enh_dir / p.name
        if enh_p.exists():
            pairs.append((p, enh_p))
        if len(pairs) >= n:
            break

    for i, (orig, enh) in enumerate(pairs):
        try:
            oimg = Image.open(orig).convert("RGB")
            eimg = Image.open(enh).convert("RGB")
            # resize to same height
            if oimg.size[1] != eimg.size[1]:
                eimg = eimg.resize((int(eimg.size[0] * oimg.size[1] / eimg.size[1]), oimg.size[1]))
            total_w = oimg.size[0] + eimg.size[0]
            new = Image.new("RGB", (total_w, oimg.size[1]))
            new.paste(oimg, (0, 0))
            new.paste(eimg, (oimg.size[0], 0))
            new.save(vis_dir / f"sample_{i:03d}.jpg")
        except Exception as e:
            print(f"[WARN] Could not create visual for {orig.name}: {e}")


def save_experiment_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[INFO] Experiment manifest saved: {out_dir / 'manifest.json'}")

# ===========================================================================
#  STEP 0 — Pre-flight checks
# ===========================================================================

def preflight():
    print("\n" + "=" * 70)
    print("  YIELD DETECTION TEACHER — YOLO26s Fine-tuning")
    print(f"  Data root  : {DATA_ROOT}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Image size : {IMGSZ}×{IMGSZ}")
    print(f"  Epochs     : {EPOCHS}")
    print("=" * 70)

    missing = []
    for split in ["train", "val"]:
        img_dir = DATA_ROOT / "images" / split
        lbl_dir = DATA_ROOT / "labels" / split
        if not img_dir.exists():
            missing.append(str(img_dir))
        if not lbl_dir.exists():
            missing.append(str(lbl_dir))

    if missing:
        print("\n[ERROR] Missing directories:")
        for m in missing:
            print(f"  {m}")
        print("\nExpected structure: data/yield/train/images & labels, etc.")
        sys.exit(1)

    # Count images
    for split in ["train", "val"]:
        n = len(list((DATA_ROOT / "images" / split).glob("*")))
        print(f"  {split:<8}: {n} images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[INFO] Pre-flight OK.")


def scan_yolo_label_classes(split: str, data_root: Path) -> tuple[set[int], int, int, int]:
    """Scan split labels under `data_root` and return (classes, files, labeled_lines, empty_files)."""
    label_dir = data_root / "labels" / split
    classes = set()
    files = 0
    labeled_lines = 0
    empty_files = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        files += 1
        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                classes.add(int(parts[0]))
                labeled_lines += 1
            except ValueError:
                classes.add(-1)

    return classes, files, labeled_lines, empty_files


def normalize_yolo_labels(data_root: Path) -> None:
    """Validate YOLO labels under `data_root` and convert 1-based apple labels to 0-based when needed.

    Creates a `.bak` copy of any label file that is rewritten.
    """
    print(f"\n[INFO] Checking YOLO labels for class-index consistency under: {data_root}")
    all_classes = set()
    for split in ["train", "val"]:
        classes, files, labeled_lines, empty_files = scan_yolo_label_classes(split, data_root)
        print(f"  {split:<5}: {files} label files, {labeled_lines} object lines, {empty_files} empty label files")
        print(f"         classes present: {sorted(classes)}")
        all_classes.update(classes)

        if not classes:
            print(f"[WARN] No object classes found in {split} labels.")
            continue

        if -1 in classes:
            raise ValueError(
                f"Invalid label contents found in {split} label files. "
                "Check for non-numeric class ids or malformed lines."
            )

    if all_classes == {1} and NC == 1:
        print("[INFO] Detected single-class labels encoded as '1'. Rewriting labels to '0' for NC=1.")
        for split in ["train", "val"]:
            label_dir = data_root / "labels" / split
            for label_path in sorted(label_dir.glob("*.txt")):
                orig_text = label_path.read_text()
                lines = orig_text.splitlines()
                rewritten = []
                changed = False
                for line in lines:
                    if not line.strip():
                        rewritten.append(line)
                        continue
                    parts = line.split()
                    if not parts:
                        rewritten.append(line)
                        continue
                    try:
                        cls = int(parts[0])
                    except ValueError:
                        rewritten.append(line)
                        continue
                    if cls == 1:
                        parts[0] = "0"
                        changed = True
                    rewritten.append(" ".join(parts))
                if changed:
                    # Backup original once
                    bak = label_path.with_suffix(label_path.suffix + ".bak")
                    if not bak.exists():
                        bak.write_text(orig_text)
                    label_path.write_text("\n".join(rewritten).rstrip() + "\n")
        print("[INFO] Label normalization complete. All apple labels are now 0-based.")
        return

    invalid = [cls for cls in all_classes if isinstance(cls, int) and cls >= NC]
    if invalid:
        raise ValueError(
            f"Label class {max(invalid)} exceeds dataset class count {NC}. "
            f"Possible class labels are 0-{NC-1}."
        )

    if NC == 1 and 0 not in all_classes and all_classes:
        raise ValueError(
            "Dataset is configured as single-class (NC=1) but no class 0 labels were found. "
            "Please remap labels to 0-based indices."
        )

    print("[INFO] YOLO label classes are consistent with NC.")


# ===========================================================================
#  STEP 1 — Write YAML (dynamic absolute paths, no broken relative paths)
# ===========================================================================

def write_yaml(data_path: Path):
    content = f"""# YOLO Dataset Config — Apple Yield Detection (LAB Enhanced)
# Auto-generated by run_yield_teacher.py — do not edit manually

path: {data_path}
train: images/train
val:   images/val

nc: {NC}
names: {CLASS_NAMES}
"""
    YAML_PATH.write_text(content)
    print(f"[INFO] Dataset YAML written for: {data_path}")


def write_yaml_to(data_path: Path, yaml_out: Path):
    """Write a YOLO dataset YAML for `data_path` to `yaml_out`."""
    content = f"""# YOLO Dataset Config — Apple Yield Detection (auto)
# Auto-generated by run_yield_teacher.py — do not edit manually

path: {data_path}
train: images/train
val:   images/val

nc: {NC}
names: {CLASS_NAMES}
"""
    yaml_out.write_text(content)
    print(f"[INFO] Dataset YAML written: {yaml_out}")


# ===========================================================================
#  STEP 2 — Optional COCO background negatives (false positive fix)
# ===========================================================================

# COCO download functionality removed — assume any negatives are pre-downloaded by user.


# ===========================================================================
#  STEP 3 — Train YOLO26s teacher
# ===========================================================================

def train_teacher() -> Path:
    """
    Fine-tunes YOLO26s on the yield dataset.
    Returns path to best.pt checkpoint.
    """
    raise RuntimeError("train_teacher must be called with explicit yaml_path and project_dir. Use train_teacher_with_config().")


def set_deterministic(seed: int):
    """Set seeds for reproducibility where possible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def train_teacher_with_config(yaml_path: Path, project_dir: Path) -> Path:
    """Train YOLO with an explicit YAML path and project dir. Returns best checkpoint path."""
    print("\n[INFO] Loading YOLO26s pretrained weights...")
    print(f"[INFO] If '{MODEL_NAME}' is not found, Ultralytics will auto-download it.")

    model = YOLO(MODEL_NAME)

    print(f"\n[INFO] Starting training — {EPOCHS} epochs, imgsz={IMGSZ}, batch={BATCH}")
    print("[INFO] This will take approximately 1-2 hours on RTX A4000.")

    model.train(
        data          = str(yaml_path),
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH,
        workers       = 4,
        patience      = PATIENCE,
        optimizer     = "AdamW",
        lr0           = LR0,
        lrf           = LRF,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        close_mosaic  = 10,

        # Small target & dense cluster enhancements
        max_det       = 300,
        mosaic        = 1.0,
        copy_paste    = 0.30,
        mixup         = 0.10,
        degrees       = 0.0,
        fliplr        = 0.5,
        flipud        = 0.0,
        scale         = 0.5,
        hsv_h         = 0.025,
        hsv_s         = 0.8,
        hsv_v         = 0.4,
        translate     = 0.1,

        project       = str(project_dir),
        name          = "yield_teacher",
        exist_ok      = True,
        save          = True,
        val           = True,
        plots         = True,
        verbose       = True,
        device        = 0,
    )

    best_ckpt = project_dir / "yield_teacher" / "weights" / "best.pt"
    if not best_ckpt.exists():
        candidates = list((project_dir / "yield_teacher" / "weights").glob("*.pt"))
        if candidates:
            best_ckpt = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"\n[INFO] ✓ Training complete.")
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    return best_ckpt


# ===========================================================================
#  STEP 4 — Evaluate on test set
# ===========================================================================

def evaluate_teacher(ckpt_path: Path, yaml_path: Path) -> dict:
    """
    Runs validation on test split. Returns mAP metrics dict.
    """
    print("\n[INFO] Evaluating teacher on test set...")
    # Extended evaluation: standard ultralytics val + per-size mAP and confusion matrix
    model = YOLO(str(ckpt_path))
    metrics = model.val(
        data    = str(yaml_path),
        split   = "val",
        imgsz   = IMGSZ,
        conf    = 0.001,
        iou     = 0.6,
        verbose = True,
    )

    results = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }

    print(f"\n  mAP@0.5      : {results['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {results['mAP50_95']:.4f}")
    print(f"  Precision    : {results['precision']:.4f}")
    print(f"  Recall       : {results['recall']:.4f}")

    return results


def _load_ground_truths(data_root: Path, split: str = "val"):
    """Load ground-truth boxes per image as dict: image_path -> list of xyxy (pixels).
    Assumes YOLO label format (cls cx cy w h) normalized.
    """
    gt = {}
    img_dir = data_root / "images" / split
    lbl_dir = data_root / "labels" / split
    for p in sorted(img_dir.iterdir()):
        if not p.is_file():
            continue
        w, h = Image.open(p).size
        lbl_path = lbl_dir / (p.stem + ".txt")
        boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                except Exception:
                    continue
                x1 = (cx - bw/2.0) * w
                y1 = (cy - bh/2.0) * h
                x2 = (cx + bw/2.0) * w
                y2 = (cy + bh/2.0) * h
                area = (x2 - x1) * (y2 - y1)
                boxes.append({"xyxy": [x1, y1, x2, y2], "area": area})
        gt[str(p)] = {"shape": (h, w), "boxes": boxes}
    return gt


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def _ap_from_pr(prec, rec):
    # integrate using trapezoidal rule over recall axis
    if len(prec) == 0:
        return 0.0
    # ensure monotonic
    prec = np.array(prec)
    rec = np.array(rec)
    # sort by recall
    order = np.argsort(rec)
    rec_s = rec[order]
    prec_s = prec[order]
    # integrate
    ap = 0.0
    for i in range(1, len(rec_s)):
        ap += (rec_s[i] - rec_s[i-1]) * prec_s[i]
    return float(ap)


def evaluate_per_size_and_confusion(ckpt_path: Path, data_root: Path, iou_thresh: float = 0.5):
    """Run inference on val set, compute per-size AP@0.5 and confusion counts.

    Size buckets (relative area): small <=0.01, medium <=0.05, large >0.05
    Returns dict with per-size AP, and overall TP/FP/FN.
    """
    model = YOLO(str(ckpt_path))
    gt = _load_ground_truths(data_root, split="val")
    # gather predictions
    preds_all = []  # list of {image, conf, xyxy}
    total_gts_per_size = {"small": 0, "medium": 0, "large": 0}

    for img_path, info in gt.items():
        h, w = info["shape"]
        img_p = Path(img_path)
        try:
            res = model.predict(source=str(img_p), conf=0.001, iou=IOU_THRESH, imgsz=IMGSZ, verbose=False)
        except Exception as e:
            print(f"[WARN] Prediction failed for {img_p.name}: {e}")
            continue
        boxes = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for j in range(len(xy)):
                preds_all.append({
                    "image": img_path,
                    "conf": float(confs[j]),
                    "xyxy": xy[j].tolist()
                })
        # count gt sizes
        for g in info["boxes"]:
            area = g["area"]
            rel = area / (w * h) if (w*h) > 0 else 0
            if rel <= 0.01:
                total_gts_per_size["small"] += 1
            elif rel <= 0.05:
                total_gts_per_size["medium"] += 1
            else:
                total_gts_per_size["large"] += 1

    # Sort predictions by conf desc
    preds_all = sorted(preds_all, key=lambda x: -x["conf"])

    # Prepare matches per size
    tp_flags = {"small": [], "medium": [], "large": []}
    fp_flags = {"small": [], "medium": [], "large": []}

    # For efficient matching, create mutable copy of GT boxes per image
    gt_mut = {k: [dict(x) for x in v["boxes"]] for k, v in gt.items()}

    for pred in preds_all:
        img = pred["image"]
        pbox = pred["xyxy"]
        best_iou = 0.0
        best_idx = -1
        if img in gt_mut:
            for idx, g in enumerate(gt_mut[img]):
                iou = _compute_iou(pbox, g["xyxy"] if isinstance(g["xyxy"], list) else g["xyxy"]) if isinstance(g["xyxy"], list) else _compute_iou(pbox, g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

        # determine size bucket for this match/pred by comparing to image area
        h, w = gt[img]["shape"]
        # assign predicted box area
        pa = max(0.0, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
        relp = pa / (w * h) if (w*h) > 0 else 0
        if relp <= 0.01:
            bucket = "small"
        elif relp <= 0.05:
            bucket = "medium"
        else:
            bucket = "large"

        if best_iou >= iou_thresh and best_idx >= 0:
            # true positive: remove GT so it's not matched again
            tp_flags[bucket].append(1)
            fp_flags[bucket].append(0)
            # remove matched GT
            del gt_mut[img][best_idx]
        else:
            tp_flags[bucket].append(0)
            fp_flags[bucket].append(1)

    # remaining GTs are false negatives
    fn_counts = {"small": 0, "medium": 0, "large": 0}
    for img, info in gt.items():
        for g in info["boxes"]:
            area = g["area"]
            rel = area / (info["shape"][1] * info["shape"][0]) if (info["shape"][1] * info["shape"][0]) > 0 else 0
            if rel <= 0.01:
                fn_counts["small"] += 1
            elif rel <= 0.05:
                fn_counts["medium"] += 1
            else:
                fn_counts["large"] += 1
    # subtract matched (TP) counts
    for b in ["small", "medium", "large"]:
        fn_counts[b] = max(0, fn_counts[b] - sum(tp_flags[b]))

    # compute AP per bucket using PR curve
    per_size_ap = {}
    for b in ["small", "medium", "large"]:
        tps = np.array(tp_flags[b], dtype=int)
        fps = np.array(fp_flags[b], dtype=int)
        if len(tps) == 0:
            per_size_ap[b] = 0.0
            continue
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        prec = tp_cum / (tp_cum + fp_cum + 1e-12)
        rec = tp_cum / (total_gts_per_size[b] + 1e-12)
        ap = _ap_from_pr(prec.tolist(), rec.tolist())
        per_size_ap[b] = float(ap)

    conf = {
        "per_size_ap": per_size_ap,
        "tp": {b: int(sum(tp_flags[b])) for b in ["small", "medium", "large"]},
        "fp": {b: int(sum(fp_flags[b])) for b in ["small", "medium", "large"]},
        "fn": fn_counts,
        "total_gts": total_gts_per_size,
    }
    return conf


# ===========================================================================
#  STEP 5 — Generate bounding-box soft labels for student distillation
# ===========================================================================

def generate_soft_labels(ckpt_path: Path, data_root: Path, soft_csv_out: Path):
    """
    Runs the trained teacher on ALL training images and saves predictions.

    CSV format (one row per image):
      image_path, json_boxes

    json_boxes is a JSON list of dicts:
      [{"xyxy": [x1,y1,x2,y2], "conf": 0.87}, ...]
    where xyxy coordinates are NORMALISED to [0, 1] image space.

    Student code (AgriculturalDataset.__init__) reads this CSV and converts
    xyxy → YOLO (cx, cy, w, h) format internally.

    Confidence threshold: {CONF_THRESH} (lower than inference to capture
    more candidate boxes — student selects further by its own objectness head).
    """
    print(f"\n[INFO] Generating soft labels (conf≥{CONF_THRESH}) "
          f"for training images...")

    model     = YOLO(str(ckpt_path))
    train_dir = data_root / "images" / "train"
    img_paths = sorted([
        p for p in train_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    print(f"[INFO] Processing {len(img_paths)} training images...")

    with open(soft_csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "json_boxes"])

        for i, img_path in enumerate(img_paths):
            try:
                results = model.predict(
                    source  = str(img_path),
                    conf    = CONF_THRESH,
                    iou     = IOU_THRESH,
                    verbose = False,
                    imgsz   = IMGSZ,
                )

                boxes_list = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        h, w    = r.orig_shape          # original image H, W
                        norm    = np.array([w, h, w, h], dtype=np.float32)
                        xyxy_px = r.boxes.xyxy.cpu().numpy()   # [N, 4] pixels
                        confs   = r.boxes.conf.cpu().numpy()   # [N]
                        for j in range(len(xyxy_px)):
                            xyxy_norm = (xyxy_px[j] / norm).tolist()
                            boxes_list.append({
                                "xyxy": xyxy_norm,
                                "conf": float(confs[j]),
                            })

                writer.writerow([str(img_path), json.dumps(boxes_list)])

            except Exception as e:
                # Write empty boxes for failed images
                writer.writerow([str(img_path), json.dumps([])])
                print(f"[WARN] Failed on {img_path.name}: {e}")

            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(img_paths)} images...")

    print(f"\n[INFO] ✓ Soft labels saved: {soft_csv_out}")
    print(f"       Rows    : {len(img_paths)} (one per training image)")
    print(f"       Columns : image_path, json_boxes")
    print(f"       Format  : normalised XYXY coordinates in [0,1]")
    print(f"       Method  : conf≥{CONF_THRESH}, NMS IoU≤{IOU_THRESH}")

    # Count stats
    total_boxes = 0
    with open(soft_csv_out) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_boxes += len(json.loads(row["json_boxes"]))

    avg_boxes = total_boxes / max(len(img_paths), 1)
    print(f"       Total boxes captured : {total_boxes}")
    print(f"       Mean boxes per image : {avg_boxes:.1f}")
    return str(soft_csv_out)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    # Ensure CLAHE globals are declared before any use in this function
    global CLAHE_COLOR_CLIP, CLAHE_LIGHT_CLIP, CLAHE_TILE_GRID
    parser = argparse.ArgumentParser(description="Run yield teacher experiments")
    parser.add_argument("--mode", choices=["original", "enhanced", "mixed"], default="enhanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--visual-samples", type=int, default=20)
    parser.add_argument("--clahe-color-clip", type=float, default=CLAHE_COLOR_CLIP)
    parser.add_argument("--clahe-light-clip", type=float, default=CLAHE_LIGHT_CLIP)
    parser.add_argument("--tile-grid", type=int, default=CLAHE_TILE_GRID[0])
    parser.add_argument("--enhancement", choices=["lab", "edge", "lab_edge"], default="lab")
    parser.add_argument("--auto", action="store_true", help="Run default automated sweep (no prompts)")
    parser.add_argument("--auto-seeds", type=str, default="42,7,123",
                        help="Comma-separated seeds for automated sweep")
    parser.add_argument("--auto-modes", type=str, default="original,enhanced,mixed",
                        help="Comma-separated modes for automated sweep")
    parser.add_argument("--auto-enhancements", type=str, default="lab,edge,lab_edge",
                        help="Comma-separated enhancement modes for automated sweep")
    parser.add_argument("--auto-clahe-color", type=str, default=str(CLAHE_COLOR_CLIP),
                        help="Comma-separated CLAHE color clip values for sweep")
    parser.add_argument("--auto-clahe-light", type=str, default=str(CLAHE_LIGHT_CLIP),
                        help="Comma-separated CLAHE light clip values for sweep")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip saving visual samples to save time in batch runs")
    parser.add_argument("--on-the-fly", action="store_true", help="Create randomized augmented copies on disk before training (simulates on-the-fly CLAHE)")
    parser.add_argument("--on-the-fly-copies", type=int, default=0, help="Number of randomized CLAHE copies per image when using --on-the-fly")
    args = parser.parse_args()

    preflight()

    # Set CLAHE params from CLI
    CLAHE_COLOR_CLIP = args.clahe_color_clip
    CLAHE_LIGHT_CLIP = args.clahe_light_clip
    CLAHE_TILE_GRID = (args.tile_grid, args.tile_grid)

    # If auto mode requested, run parameter sweep without interactive prompts
    if args.auto:
        print("\n[INFO] Running automated sweep (no prompts). This may take a long time.")
        seeds = [int(s) for s in args.auto_seeds.split(",") if s.strip()]
        modes = [m.strip() for m in args.auto_modes.split(",") if m.strip()]
        enhancements = [e.strip() for e in args.auto_enhancements.split(",") if e.strip()]
        clahe_color_vals = [float(x) for x in args.auto_clahe_color.split(",") if x.strip()]
        clahe_light_vals = [float(x) for x in args.auto_clahe_light.split(",") if x.strip()]

        batch_summary = []
        enhanced_cache = {}

        ts_batch = datetime.now().strftime("%Y%m%d-%H%M%S")
        batch_dir = OUTPUT_DIR / f"batch_{ts_batch}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            for mode in modes:
                for enhancement in enhancements:
                    # skip meaningless combinations: original + enhancement
                    if mode == "original" and enhancement != "lab":
                        # treat as original only once
                        enhancement_to_use = "lab"
                    else:
                        enhancement_to_use = enhancement

                    for cc in clahe_color_vals:
                        for cl in clahe_light_vals:
                            # set CLAHE params
                            CLAHE_COLOR_CLIP = cc
                            CLAHE_LIGHT_CLIP = cl
                            CLAHE_TILE_GRID = (args.tile_grid, args.tile_grid)

                            exp_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                            exp_stamp = f"{exp_ts}_{mode}_{enhancement_to_use}_s{seed}_cc{cc}_cl{cl}"
                            if args.exp_name:
                                exp_stamp = f"{args.exp_name}_{exp_stamp}"
                            exp_dir = batch_dir / exp_stamp
                            exp_dir.mkdir(parents=True, exist_ok=True)

                            # Reproducibility
                            set_deterministic(seed)

                            # Preprocess / dataset selection (cache enhanced roots)
                            if mode == "original":
                                data_root = DATA_ROOT
                            else:
                                if enhancement_to_use in enhanced_cache:
                                    enhanced_root = enhanced_cache[enhancement_to_use]
                                else:
                                    enhanced_root = preprocess_dataset_images(enhancement_mode=enhancement_to_use)
                                    enhanced_cache[enhancement_to_use] = enhanced_root

                                verify_dataset_consistency(enhanced_root)
                                normalize_yolo_labels(enhanced_root)
                                verify_dataset_consistency(enhanced_root)

                                if mode == "enhanced":
                                    data_root = enhanced_root
                                elif mode == "mixed":
                                    data_root = create_mixed_dataset(enhanced_root)

                            # Write dataset YAML into experiment dir
                            yaml_out = exp_dir / "yield_dataset.yaml"
                            write_yaml_to(data_root, yaml_out)

                            # Save visuals and manifest (skip if requested)
                            if not args.skip_visuals:
                                save_visual_samples(DATA_ROOT, data_root if data_root != DATA_ROOT else data_root, exp_dir, n=args.visual_samples)
                            manifest = {
                                "mode": mode,
                                "seed": seed,
                                "clahe": {
                                    "color_clip": CLAHE_COLOR_CLIP,
                                    "light_clip": CLAHE_LIGHT_CLIP,
                                    "tile_grid": CLAHE_TILE_GRID,
                                },
                                "enhancement": enhancement_to_use,
                                "data_root": str(data_root),
                                "yaml": str(yaml_out),
                                "timestamp": exp_ts,
                            }
                            save_experiment_manifest(exp_dir, manifest)

                            # (COCO download removed; assume negatives are pre-provisioned by user)

                            # Train
                            # Optionally create on-the-fly augmented copies (disk cache)
                            temp_aug_root = None
                            if args.on_the_fly and args.on_the_fly_copies > 0 and mode != "original":
                                from tempfile import mkdtemp
                                tmp = Path(mkdtemp(prefix="on_the_fly_", dir=str(exp_dir)))
                                temp_aug_root = tmp
                                print(f"[INFO] Creating {args.on_the_fly_copies} randomized augmented copies per image at: {tmp}")
                                # generate randomized augmented copies using lab/edge modes
                                def _rand_clip(base, scale=0.5):
                                    return max(0.1, float(base) + np.random.randn() * scale)

                                for split2 in ["train", "val"]:
                                    src_dir = data_root / "images" / split2
                                    dst_dir = tmp / "images" / split2
                                    dst_dir.mkdir(parents=True, exist_ok=True)
                                    lbl_dst = tmp / "labels" / split2
                                    lbl_dst.mkdir(parents=True, exist_ok=True)
                                    for p in sorted(src_dir.iterdir()):
                                        if not p.is_file():
                                            continue
                                        # copy original once
                                        shutil.copy(p, dst_dir / p.name)
                                        lbl = (data_root / "labels" / split2 / (p.stem + ".txt"))
                                        if lbl.exists():
                                            shutil.copy(lbl, lbl_dst / lbl.name)
                                        # create randomized copies
                                        for k in range(args.on_the_fly_copies):
                                            rnd_color = _rand_clip(CLAHE_COLOR_CLIP, scale=1.0)
                                            rnd_light = _rand_clip(CLAHE_LIGHT_CLIP, scale=0.5)
                                            out_name = f"{p.stem}_aug{k}{p.suffix}"
                                            out_path = dst_dir / out_name
                                            # apply chosen enhancement deterministic-ish
                                            # temporarily override globals for this call
                                            old_cc = CLAHE_COLOR_CLIP
                                            old_cl = CLAHE_LIGHT_CLIP
                                            CLAHE_COLOR_CLIP = rnd_color
                                            CLAHE_LIGHT_CLIP = rnd_light
                                            try:
                                                if enhancement_to_use == "lab":
                                                    apply_agricultural_lab_filter(p, out_path)
                                                elif enhancement_to_use == "edge":
                                                    apply_edge_texture_filter(p, out_path, mode="edge")
                                                else:
                                                    apply_edge_texture_filter(p, out_path, mode="lab_edge")
                                            except Exception as _e:
                                                print(f"[WARN] on-the-fly augment failed for {p.name}: {_e}")
                                            CLAHE_COLOR_CLIP = old_cc
                                            CLAHE_LIGHT_CLIP = old_cl
                                            # copy label
                                            if lbl.exists():
                                                shutil.copy(lbl, lbl_dst / (out_name.replace(p.suffix, ".txt")))

                                # point yaml/data_root to tmp for training
                                data_root_for_train = tmp
                                write_yaml_to(data_root_for_train, yaml_out)
                                train_data_root_used = data_root_for_train
                            else:
                                best_ckpt = train_teacher_with_config(yaml_out, exp_dir)
                                train_data_root_used = data_root

                            # If we created temp_aug_root, run training using that dataset
                            if args.on_the_fly and args.on_the_fly_copies > 0 and mode != "original":
                                best_ckpt = train_teacher_with_config(yaml_out, exp_dir)

                            # Evaluate (standard metrics)
                            metrics = evaluate_teacher(best_ckpt, yaml_out)
                            # Extended per-size evaluation and confusion counts (on val set)
                            per_size = evaluate_per_size_and_confusion(best_ckpt, (train_data_root_used if 'train_data_root_used' in locals() else data_root), iou_thresh=0.5)

                            # Soft labels
                            soft_csv_out = exp_dir / "yield_soft_labels.csv"
                            csv_path = generate_soft_labels(best_ckpt, train_data_root_used if 'train_data_root_used' in locals() else data_root, soft_csv_out)

                            # Save summary
                            summary = {
                                "model":           MODEL_NAME,
                                "best_checkpoint": str(best_ckpt),
                                "soft_labels_csv": csv_path,
                                "test_metrics":    metrics,
                                "per_size":        per_size,
                                "imgsz":           IMGSZ,
                                "conf_threshold":  CONF_THRESH,
                                "iou_threshold":   IOU_THRESH,
                                "enhancement":     enhancement_to_use,
                                "experiment_dir":  str(exp_dir),
                            }
                            (exp_dir / "yield_teacher_summary.json").write_text(json.dumps(summary, indent=2))

                            # cleanup temp aug dir if any
                            if args.on_the_fly and args.on_the_fly_copies > 0 and temp_aug_root is not None:
                                try:
                                    shutil.rmtree(temp_aug_root)
                                    print(f"[INFO] Removed temporary on-the-fly folder: {temp_aug_root}")
                                except Exception:
                                    pass

                            # Append to batch summary
                            batch_summary.append(summary)

        # Save batch summary
        (batch_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2))
        print(f"\n[INFO] Automated sweep complete. Batch results: {batch_dir}")
        return


if __name__ == "__main__":
    main()