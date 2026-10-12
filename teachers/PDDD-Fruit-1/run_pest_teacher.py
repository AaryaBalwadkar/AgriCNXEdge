"""
run_pest_teacher.py
===================
Entry point for training the Pest classification teacher.

USAGE:
  python run_pest_teacher.py

BEFORE RUNNING — fill in the three paths below:
  1. DATA_ROOT       : folder containing pest class subfolders
  2. PDDD_WEIGHTS    : path to convnext_small-model-99.pth
  3. OUTPUT_DIR      : where checkpoints, logs, soft labels will be saved

To train Leaf teacher  → copy this file as run_leaf_teacher.py  + change TASK
To train Fruit teacher → copy this file as run_fruit_teacher.py + change TASK
"""

import os
import sys
from pathlib import Path

import torch

# ============================================================
#  ← FILL THESE IN BEFORE RUNNING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_str: str) -> str:
    expanded = os.path.expanduser(path_str)
    path = Path(expanded)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())


# Root folder that contains: Xylotrechus/, aphids/, whitefly/, etc.
# If you have train/val/test subfolders instead, set use_existing_split=True below
DATA_ROOT = resolve_path("data/pest")

# Path to the downloaded PDDD ConvNeXt-Small checkpoint
# Download: https://zenodo.org/records/7890438/files/convnext_small-model-99.pth?download=1
PDDD_WEIGHTS = resolve_path("weights/convnext_small-model-99.pth")

# Where to save: best_teacher.pth, training_log.json, soft labels will be saved
OUTPUT_DIR = resolve_path("outputs/pest")

# ============================================================
#  TASK SELECTION — change to LEAF_TASK or FRUIT_TASK for other teachers
# ============================================================
from teacher_config import PEST_TASK as TASK

# ============================================================
#  CONFIG — tune these if needed, defaults are good for A4000
# ============================================================
from teacher_config import DataConfig, ModelConfig, TrainConfig

data_cfg = DataConfig(
    data_root          = DATA_ROOT,
    use_existing_split = False,     # Set True if you have train/val/test subfolders
    image_size         = 224,
    min_image_dim      = 64,
    min_pixel_std      = 5.0,
    phash_threshold    = 8,
    val_split          = 0.15,
    test_split         = 0.15,
    seed               = 42,
    batch_size         = 32,        # Safe for RTX A4000 with ConvNeXt-Small at 224px
    num_workers        = 4,
    pin_memory         = True,
)

model_cfg = ModelConfig(
    pddd_weights_path = PDDD_WEIGHTS,
    pddd_num_classes  = 120,        # The checkpoint head size is auto-detected by teacher_model.py
    backbone_name     = "convnext_small",
    feature_dim       = 768,
    dropout_rate      = 0.5,
)

train_cfg = TrainConfig(
    output_dir   = OUTPUT_DIR,
    task_name    = TASK["task_name"],
    num_classes  = TASK["num_classes"],
    epochs       = 60,
    batch_size   = data_cfg.batch_size,
    num_workers  = data_cfg.num_workers,
    lr_backbone  = 2e-6,            # Smaller LR for backbone fine-tuning
    lr_head      = 4e-4,            # Moderate LR for the new head
    weight_decay = 1e-4,
    eta_min      = 1e-8,
    focal_gamma  = 2.0,
    patience     = 8,
    min_delta    = 1e-4,
    quality_gate_f1 = 0.70,
    max_grad_norm   = 1.0,
)

# ============================================================
#  MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print(f"  PEST TEACHER TRAINING")
    print(f"  Task         : {TASK['task_name']}")
    print(f"  Classes ({TASK['num_classes']})  : {TASK['class_names']}")
    print(f"  Data root    : {DATA_ROOT}")
    print(f"  PDDD weights : {PDDD_WEIGHTS}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("=" * 70)

    # Sanity checks
    if not os.path.exists(DATA_ROOT):
        print(f"\n[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        print("Please update the DATA_ROOT path at the top of this file.")
        sys.exit(1)

    if not os.path.exists(PDDD_WEIGHTS):
        print(f"\n[ERROR] PDDD weights not found: {PDDD_WEIGHTS}")
        print("Download from: https://zenodo.org/records/7890438/files/convnext_small-model-99.pth?download=1")
        sys.exit(1)

    # GPU check
    if not torch.cuda.is_available():
        print("\n[WARN] CUDA not available. Training on CPU will be very slow.")
        print("Expected hardware: RTX A4000 (16GB). Proceeding anyway...")
    else:
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU detected: {gpu} ({vram:.1f} GB VRAM)")

    # --------------------------------------------------------
    # PHASE 1: TRAIN THE TEACHER
    # --------------------------------------------------------
    from teacher_train import TeacherTrainer

    trainer = TeacherTrainer(
        data_config  = data_cfg,
        model_config = model_cfg,
        train_config = train_cfg,
        class_names  = TASK["class_names"],
    )

    test_macro_f1, gate_result = trainer.train()

    print(f"\n[RESULT] Test Macro F1: {test_macro_f1:.4f}")

    # Check if teacher is approved for all classes
    unsafe_classes = [cls for cls, info in gate_result.items() if info["status"] == "UNSAFE"]
    if unsafe_classes:
        print(f"\n[WARNING] Teacher FAILED quality gate for: {unsafe_classes}")
        print("  These classes will produce unreliable soft labels.")
        print("  Recommendation: collect more images for these classes before distillation.")
        proceed = input("\n  Proceed with soft label generation anyway? (y/n): ").strip().lower()
        if proceed != "y":
            print("[INFO] Soft label generation skipped.")
            return
    else:
        print("\n[INFO] Teacher passed all quality gates. Proceeding to soft label generation.")

    # --------------------------------------------------------
    # PHASE 2: GENERATE SOFT LABELS FOR ALL IMAGES
    # --------------------------------------------------------
    # Rebuild the full image map (train + val + test combined)
    # so we generate soft labels for every single image
    from teacher_data import discover_images
    from collections import defaultdict

    raw_map = discover_images(
        data_root          = DATA_ROOT,
        class_names        = TASK["class_names"],
        use_existing_split = data_cfg.use_existing_split,
    )

    # Flatten all splits into one map for soft label generation
    all_images: dict = defaultdict(list)
    for cls in TASK["class_names"]:
        for split in ["all", "train", "val", "test"]:
            all_images[cls].extend(raw_map[cls].get(split, []))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from teacher_train import generate_soft_labels
    csv_path = generate_soft_labels(
        model        = trainer.model,
        image_dir_map= dict(all_images),
        class_names  = TASK["class_names"],
        image_size   = data_cfg.image_size,
        task_name    = TASK["task_name"],
        output_dir   = OUTPUT_DIR,
        device       = device,
        batch_size   = 64,          # Larger batch is fine for inference
        num_workers  = data_cfg.num_workers,
    )

    print("\n" + "=" * 70)
    print("  PEST TEACHER TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best checkpoint : {trainer.best_ckpt_path}")
    print(f"  Soft labels CSV : {csv_path}")
    print(f"  Training log    : {os.path.join(OUTPUT_DIR, 'pest_training_log.json')}")
    print(f"  Cleaning report : {os.path.join(OUTPUT_DIR, 'pest_cleaning_report.json')}")
    print(f"\n  Test Macro F1   : {test_macro_f1:.4f}")
    print(f"\n  Quality gate summary:")
    for cls, info in gate_result.items():
        marker = "✓" if info["status"] == "SAFE" else "✗"
        print(f"    {marker} {cls:<30} F1={info['f1']:.4f}  [{info['status']}]")
    print("=" * 70)

    print("\n  NEXT STEPS:")
    print("  1. Review the quality gate results above.")
    print("  2. If any class is UNSAFE, collect more images before distillation.")
    print("  3. The soft labels CSV is ready for use in the student distillation pipeline.")
    print("  4. To train the LEAF teacher: copy this file as run_leaf_teacher.py,")
    print("     change TASK to LEAF_TASK, update DATA_ROOT and OUTPUT_DIR.")
    print("  5. To train the FRUIT teacher: same, use FRUIT_TASK.")


if __name__ == "__main__":
    main()
