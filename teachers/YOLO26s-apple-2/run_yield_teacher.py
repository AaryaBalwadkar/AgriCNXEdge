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
import urllib.request
from pathlib import Path

import numpy as np

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
    clahe_color = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    a_enhanced = clahe_color.apply(a)

    # Mild CLAHE on L channel (Shadow/Lighting normalization)
    clahe_light = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    L_enhanced = clahe_light.apply(L)

    # Recombine and convert back to BGR image space
    lab_enhanced = cv2.merge([L_enhanced, a_enhanced, b])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    cv2.imwrite(str(output_path), bgr_enhanced)


def preprocess_dataset_images():
    """
    Preprocesses train and val splits to apply LAB chromatic enhancement.
    Creates an enhanced dataset folder used for teacher training.
    """
    print("\n[INFO] Preprocessing dataset with LAB Chromatic Enhancement...")
    enhanced_root = DATA_ROOT.parent / "yield_enhanced"

    for split in ["train", "val"]:
        src_img_dir = DATA_ROOT / "images" / split
        dst_img_dir = enhanced_root / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels to new root
        src_lbl_dir = DATA_ROOT / "labels" / split
        dst_lbl_dir = enhanced_root / "labels" / split
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for lbl in src_lbl_dir.glob("*.txt"):
            shutil.copy(lbl, dst_lbl_dir / lbl.name)

        # Apply LAB filter to images
        for img_path in src_img_dir.glob("*.[jJ][pP][gG]"):
            apply_agricultural_lab_filter(img_path, dst_img_dir / img_path.name)
        for img_path in src_img_dir.glob("*.[pP][nN][gG]"):
            apply_agricultural_lab_filter(img_path, dst_img_dir / img_path.name)

    print(f"[INFO] ✓ LAB Enhanced dataset prepared at: {enhanced_root}")
    return enhanced_root

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


def scan_yolo_label_classes(split: str) -> tuple[set[int], int, int, int]:
    """Scan split labels and return (classes, files, labeled_lines, empty_files)."""
    label_dir = DATA_ROOT / "labels" / split
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


def normalize_yolo_labels() -> None:
    """Validate YOLO labels and convert 1-based apple labels to 0-based when needed."""
    print("\n[INFO] Checking YOLO labels for class-index consistency...")
    all_classes = set()
    for split in ["train", "val"]:
        classes, files, labeled_lines, empty_files = scan_yolo_label_classes(split)
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
            label_dir = DATA_ROOT / "labels" / split
            for label_path in sorted(label_dir.glob("*.txt")):
                lines = label_path.read_text().splitlines()
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


# ===========================================================================
#  STEP 2 — Optional COCO background negatives (false positive fix)
# ===========================================================================

def add_coco_negatives(n_images: int = 500):
    """
    Downloads n COCO 2017-val images that contain NO apple annotation
    and adds them to train/images + empty train/labels files.
    This directly teaches YOLO that floors/walls/random scenes = no apple.

    Skip this step if network access is blocked — training still works,
    but false positives on non-orchard images will be higher.
    """
    try:
        import urllib.request, zipfile, tempfile
        print(f"\n[INFO] Downloading COCO 2017 annotation file (~241 MB)...")

        ann_url  = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        img_url  = "http://images.cocodataset.org/zips/val2017.zip"

        with tempfile.TemporaryDirectory() as tmp:
            # Download annotations
            ann_zip = os.path.join(tmp, "ann.zip")
            urllib.request.urlretrieve(ann_url, ann_zip)
            with zipfile.ZipFile(ann_zip) as z:
                z.extract("annotations/instances_val2017.json", tmp)

            with open(os.path.join(tmp, "annotations/instances_val2017.json")) as f:
                coco = json.load(f)

            # Find images with APPLE annotation (cat_id=53)
            apple_img_ids = set(
                ann["image_id"] for ann in coco["annotations"]
                if ann["category_id"] == 53
            )
            # Non-apple images
            neg_images = [
                img for img in coco["images"]
                if img["id"] not in apple_img_ids
            ][:n_images]

            print(f"[INFO] Found {len(neg_images)} non-apple COCO images. Downloading...")

            # Download val images ZIP
            img_zip_path = os.path.join(tmp, "val2017.zip")
            urllib.request.urlretrieve(img_url, img_zip_path)

            needed = {img["file_name"] for img in neg_images}
            train_img = DATA_ROOT / "images" / "train"
            train_lbl = DATA_ROOT / "labels" / "train"

            with zipfile.ZipFile(img_zip_path) as z:
                for name in z.namelist():
                    fname = Path(name).name
                    if fname in needed:
                        # Extract + copy to train/images
                        data = z.read(name)
                        dst  = train_img / f"coco_neg_{fname}"
                        dst.write_bytes(data)
                        # Empty label file = no objects in this image
                        (train_lbl / f"coco_neg_{Path(fname).stem}.txt").write_text("")

        print(f"[INFO] ✓ Added {n_images} COCO background negatives to training set.")

    except Exception as e:
        print(f"[WARN] Could not download COCO negatives: {e}")
        print("[WARN] Skipping negatives. Train will still work — expect "
              "more false positives on non-orchard images.")


# ===========================================================================
#  STEP 3 — Train YOLO26s teacher
# ===========================================================================

def train_teacher() -> Path:
    """
    Fine-tunes YOLO26s on the yield dataset.
    Returns path to best.pt checkpoint.
    """
    print("\n[INFO] Loading YOLO26s pretrained weights...")
    print(f"[INFO] If '{MODEL_NAME}' is not found, Ultralytics will auto-download it.")

    model = YOLO(MODEL_NAME)

    print(f"\n[INFO] Starting training — {EPOCHS} epochs, imgsz={IMGSZ}, batch={BATCH}")
    print("[INFO] This will take approximately 1-2 hours on RTX A4000.")

    model.train(
        data          = str(YAML_PATH),
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
        max_det       = 300,    # Allows up to 300 detections per canopy image
        mosaic        = 1.0,    
        copy_paste    = 0.30,   # Increased: forces network to learn occluded green boundaries
        mixup         = 0.10,
        degrees       = 0.0,    
        fliplr        = 0.5,
        flipud        = 0.0,
        scale         = 0.5,    # Multi-scale training forces small-object feature retention
        hsv_h         = 0.025,  # Increased hue jitter to improve color invariance
        hsv_s         = 0.8,    # Higher saturation jitter for foliage separation
        hsv_v         = 0.4,
        translate     = 0.1,

        project       = str(OUTPUT_DIR),
        name          = "yield_teacher",
        exist_ok      = True,
        save          = True,
        val           = True,
        plots         = True,
        verbose       = True,
        device        = 0,
    )

    best_ckpt = OUTPUT_DIR / "yield_teacher" / "weights" / "best.pt"
    if not best_ckpt.exists():
        # Fallback — Ultralytics sometimes names differently
        candidates = list((OUTPUT_DIR / "yield_teacher" / "weights").glob("*.pt"))
        if candidates:
            best_ckpt = max(candidates, key=lambda p: p.stat().st_mtime)

    print(f"\n[INFO] ✓ Training complete.")
    print(f"[INFO] Best checkpoint: {best_ckpt}")
    return best_ckpt


# ===========================================================================
#  STEP 4 — Evaluate on test set
# ===========================================================================

def evaluate_teacher(ckpt_path: Path) -> dict:
    """
    Runs validation on test split. Returns mAP metrics dict.
    """
    print("\n[INFO] Evaluating teacher on test set...")
    model   = YOLO(str(ckpt_path))
    metrics = model.val(
        data    = str(YAML_PATH),
        split   = "val",
        imgsz   = IMGSZ,
        conf    = 0.001,    # low conf for mAP computation (standard)
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


# ===========================================================================
#  STEP 5 — Generate bounding-box soft labels for student distillation
# ===========================================================================

def generate_soft_labels(ckpt_path: Path):
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
    train_dir = DATA_ROOT / "images" / "train"
    img_paths = sorted([
        p for p in train_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

    print(f"[INFO] Processing {len(img_paths)} training images...")

    with open(SOFT_CSV, "w", newline="") as f:
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

    print(f"\n[INFO] ✓ Soft labels saved: {SOFT_CSV}")
    print(f"       Rows    : {len(img_paths)} (one per training image)")
    print(f"       Columns : image_path, json_boxes")
    print(f"       Format  : normalised XYXY coordinates in [0,1]")
    print(f"       Method  : conf≥{CONF_THRESH}, NMS IoU≤{IOU_THRESH}")

    # Count stats
    total_boxes = 0
    with open(SOFT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_boxes += len(json.loads(row["json_boxes"]))

    avg_boxes = total_boxes / max(len(img_paths), 1)
    print(f"       Total boxes captured : {total_boxes}")
    print(f"       Mean boxes per image : {avg_boxes:.1f}")
    return str(SOFT_CSV)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    preflight()

    # Step 0.1 — Preprocess dataset with LAB Chromatic Enhancement
    # (Separates green apples from leaves & sharpens tiny red dots)
    enhanced_data_root = preprocess_dataset_images()

    # Step 1 — Validate labels & write YAML pointing to ENHANCED data
    normalize_yolo_labels()
    write_yaml(data_path=enhanced_data_root)

    # Step 2 — Optional background negatives
    print("\n[Q] Add COCO background negatives to fix false positives?")
    print("    (Downloads ~1.3GB. Skip if network is slow/blocked.)")
    ans = input("    Add negatives? (y/n): ").strip().lower()
    if ans == "y":
        add_coco_negatives(n_images=500)
    else:
        print("[INFO] Skipping negatives. You can add them later and retrain.")

    # Step 3 — Train Teacher on LAB-Enhanced Dataset
    best_ckpt = train_teacher()

    # Step 4 — Evaluate
    metrics = evaluate_teacher(best_ckpt)

    # Step 5 — Soft labels generation
    csv_path = generate_soft_labels(best_ckpt)

    # Save summary
    summary = {
        "model":           MODEL_NAME,
        "best_checkpoint": str(best_ckpt),
        "soft_labels_csv": csv_path,
        "test_metrics":    metrics,
        "imgsz":           IMGSZ,
        "conf_threshold":  CONF_THRESH,
        "iou_threshold":   IOU_THRESH,
        "enhancement":     "CIELAB a* CLAHE Chromatic Decoupling",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("  YIELD TEACHER (LAB ENHANCED) — COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()