"""
run_fruit_teacher.py
====================
Entry point for training the Fruit Disease classification teacher.

Classes: Anthracnose, Black Rot, Blotch, Healthy, Scab
Data:    data/fruit/ subfolder of project root

USAGE:
  python run_fruit_teacher.py

IMPORTANT — verify your actual fruit folder names before running:
  ls data/fruit/
  Expected: Anthracnose  "Black Rot"  Blotch  Healthy  Scab

NOTE on "Black Rot": the folder has a SPACE in its name.
The class_name in FRUIT_TASK must match the folder name exactly.
If your folder is named "Black_Rot" (underscore), change it in teacher_config.py.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import torch

# ── Resolve paths relative to this script ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def rp(s: str) -> str:
    p = Path(os.path.expanduser(s))
    return str((BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve())

DATA_ROOT    = rp("data/fruit")          # must contain Anthracnose/, Black Rot/, etc.
PDDD_WEIGHTS = rp("weights/convnext_small-model-99.pth")
OUTPUT_DIR   = rp("outputs/fruit")

# ── Task ─────────────────────────────────────────────────────────────────────
from old.oldfruit2.teacher_config import FRUIT_TASK as TASK
from old.oldfruit2.teacher_config import DataConfig, ModelConfig, TrainConfig

data_cfg = DataConfig(
    data_root          = DATA_ROOT,
    use_existing_split = False,
    image_size         = 224,
    min_image_dim      = 64,
    min_pixel_std      = 5.0,
    phash_threshold    = 8,
    val_split          = 0.10,
    test_split         = 0.15,
    seed               = 42,
    batch_size         = 32,
    num_workers        = 4,
    pin_memory         = True,
    sampler_power      = 0.85,
    rare_class_threshold = 400,
    oversample_factor  = 1,
    rare_oversample_factor = 2,
)

model_cfg = ModelConfig(
    pddd_weights_path = PDDD_WEIGHTS,
    # backbone_name and pddd_num_classes are AUTO-DETECTED from checkpoint
    # these values are only fallbacks and are effectively ignored by teacher_model.py v4
    pddd_num_classes  = 120,
    backbone_name     = "convnext_small",
    feature_dim       = 768,
    dropout_rate      = 0.5,
    drop_path_rate    = 0.15,
)

train_cfg = TrainConfig(
    output_dir   = OUTPUT_DIR,
    task_name    = TASK["task_name"],
    num_classes  = TASK["num_classes"],
    epochs       = 80,
    batch_size   = data_cfg.batch_size,
    num_workers  = data_cfg.num_workers,
    lr_backbone  = 5e-6,
    lr_head      = 5e-4,
    weight_decay = 1e-4,
    eta_min      = 1e-7,
    focal_gamma  = 2.0,
    label_smoothing = 0.05,
    patience     = 20,
    min_delta    = 1e-4,
    # NOTE: Blotch (24 images) and Scab (23 images) will very likely fail
    # the quality gate (F1 < 0.70). This is expected. The training will
    # complete, report them as UNSAFE, and ask whether to proceed.
    quality_gate_f1 = 0.70,
    max_grad_norm   = 1.0,
    use_mixup       = True,
    mixup_alpha     = 0.1,
    cutmix_alpha    = 0.8,
    mixup_prob      = 0.4,
    use_tta         = True,
    calibrate_temperature = True,
)


def main():
    print("\n" + "=" * 70)
    print("  FRUIT DISEASE TEACHER TRAINING")
    print(f"  Task    : {TASK['task_name']}")
    print(f"  Classes : {TASK['class_names']}")
    print(f"  Data    : {DATA_ROOT}")
    print(f"  Weights : {PDDD_WEIGHTS}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 70)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not os.path.exists(DATA_ROOT):
        print(f"\n[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        sys.exit(1)

    if not os.path.exists(PDDD_WEIGHTS):
        print(f"\n[ERROR] PDDD weights not found: {PDDD_WEIGHTS}")
        print("Download: https://zenodo.org/records/7890438/files/"
              "convnext_small-model-99.pth?download=1")
        sys.exit(1)

    # Verify all class folders exist
    print("\n[INFO] Verifying class folders...")
    all_ok = True
    for cls in TASK["class_names"]:
        cls_path = Path(DATA_ROOT) / cls
        if cls_path.exists():
            n = sum(1 for f in cls_path.rglob("*")
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            flag = "  ← WARNING: very few images" if n < 30 else ""
            print(f"  ✓ {cls:<20} {n} images{flag}")
        else:
            print(f"  ✗ {cls:<20} FOLDER NOT FOUND: {cls_path}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] One or more class folders missing. "
              "Check FRUIT_TASK class_names in teacher_config.py match "
              "your actual folder names exactly (Linux is case-sensitive, "
              "spaces in names matter).")
        print(f"\nActual contents of {DATA_ROOT}:")
        os.system(f"ls '{DATA_ROOT}'")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\n[WARN] CUDA not available — training on CPU (slow).")
    else:
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU: {gpu} ({vram:.1f} GB VRAM)")

    # ── Phase 1: Train ───────────────────────────────────────────────────────
    from old.oldfruit2.teacher_train import TeacherTrainer
    trainer = TeacherTrainer(
        data_config  = data_cfg,
        model_config = model_cfg,
        train_config = train_cfg,
        class_names  = TASK["class_names"],
    )
    test_macro_f1, gate_result = trainer.train()

    print(f"\n[RESULT] Test Macro F1: {test_macro_f1:.4f}")

    unsafe = [c for c, i in gate_result.items() if i["status"] == "UNSAFE"]
    if unsafe:
        print(f"\n[WARNING] Classes FAILED quality gate: {unsafe}")
        print("  Expected for Blotch (24 images) and Scab (23 images).")
        print("  These classes have too few training samples for a reliable teacher.")
        proceed = input("\n  Proceed with soft label generation anyway? (y/n): ").strip().lower()
        if proceed != "y":
            print("[INFO] Soft label generation skipped.")
            return
    else:
        print("\n[INFO] All classes passed quality gate. Generating soft labels...")

    # ── Phase 2: Generate soft labels ────────────────────────────────────────
    from old.oldfruit2.teacher_data import discover_images
    from old.oldfruit2.teacher_train import generate_soft_labels

    raw_map   = discover_images(DATA_ROOT, TASK["class_names"],
                                use_existing_split=False)
    all_imgs  = defaultdict(list)
    for cls in TASK["class_names"]:
        for split in ["all", "train", "val", "test"]:
            all_imgs[cls].extend(raw_map[cls].get(split, []))

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = generate_soft_labels(
        model         = trainer.model,
        image_dir_map = dict(all_imgs),
        class_names   = TASK["class_names"],
        image_size    = data_cfg.image_size,
        task_name     = TASK["task_name"],
        output_dir    = OUTPUT_DIR,
        device        = device,
        batch_size    = 64,
        num_workers   = data_cfg.num_workers,
        temperature   = getattr(trainer, "calibrated_temperature", 1.0),
        use_tta       = getattr(train_cfg, "use_tta", False),
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FRUIT TEACHER TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best checkpoint : {trainer.best_ckpt_path}")
    print(f"  Soft labels CSV : {csv_path}")
    print(f"  Test Macro F1   : {test_macro_f1:.4f}")
    print(f"\n  Quality gate:")
    for cls, info in gate_result.items():
        m = "✓" if info["status"] == "SAFE" else "✗"
        print(f"    {m} {cls:<25} F1={info['f1']:.4f}  [{info['status']}]")
    print("=" * 70)
    print("\n  NEXT STEPS:")
    print("  1. Check quality gate. Blotch/Scab UNSAFE is expected — document in paper.")
    print("  2. Soft labels CSV → used in student distillation pipeline.")
    print("  3. To run the Leaf teacher: python run_leaf_teacher.py")


if __name__ == "__main__":
    main()