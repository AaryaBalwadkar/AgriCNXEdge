"""
run_pest_teacher.py  (v2 — Ensemble: 3 seeds × 8 TTA views)
=============================================================
Trains 3 independent pest classifiers (seeds 42, 123, 456).
Same two-stage + warmup + label smoothing pipeline as fruit teacher.

CLASSES (4 — melaybug/whitefly folders absent):
  xylotrechus, aphids, leafhoppers, spider_mite

USAGE:
  python run_pest_teacher.py
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent

def rp(s: str) -> str:
    p = Path(os.path.expanduser(s))
    return str((BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve())

DATA_ROOT    = rp("data/pest")
PDDD_WEIGHTS = rp("weights/convnext_small-model-99.pth")
OUTPUT_DIR   = rp("outputs/pest")

from old.oldYOLO26s3.teacher_config import PEST_TASK as TASK, ENSEMBLE_SEEDS
from old.oldYOLO26s3.teacher_config import DataConfig, ModelConfig, TrainConfig

data_cfg = DataConfig(
    data_root          = DATA_ROOT,
    use_existing_split = False,
    image_size         = 224,
    min_image_dim      = 64,
    min_pixel_std      = 5.0,
    phash_threshold    = 8,
    val_split          = 0.15,
    test_split         = 0.15,
    seed               = 42,
    batch_size         = 32,
    num_workers        = 4,
    pin_memory         = True,
)

model_cfg = ModelConfig(
    pddd_weights_path = PDDD_WEIGHTS,
    pddd_num_classes  = 120,
    backbone_name     = "convnext_small",
    feature_dim       = 768,
    dropout_rate      = 0.5,
)


def make_train_cfg(seed_output_dir: str) -> TrainConfig:
    return TrainConfig(
        output_dir             = seed_output_dir,
        task_name              = TASK["task_name"],
        num_classes            = TASK["num_classes"],
        epochs                 = 100,
        freeze_epochs          = 8,
        warmup_epochs          = 5,
        lr_head                = 1e-3,
        lr_head_phase2         = 1e-4,
        lr_backbone            = 1e-5,
        weight_decay           = 1e-4,
        eta_min                = 1e-7,
        focal_gamma            = 2.0,
        label_smoothing        = 0.1,
        minority_aug_threshold = 400,
        patience               = 20,
        min_delta              = 1e-4,
        quality_gate_f1        = 0.70,
        max_grad_norm          = 1.0,
        batch_size             = data_cfg.batch_size,
        num_workers            = data_cfg.num_workers,
        use_tta                = True,
        tta_n                  = 8,
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def preflight():
    print("\n" + "=" * 70)
    print("  PEST TEACHER — ENSEMBLE TRAINING")
    print(f"  Classes : {TASK['class_names']}")
    print(f"  Seeds   : {ENSEMBLE_SEEDS}")
    print(f"  Data    : {DATA_ROOT}")
    print("=" * 70)

    if not os.path.exists(DATA_ROOT):
        print(f"\n[ERROR] DATA_ROOT not found: {DATA_ROOT}"); sys.exit(1)
    if not os.path.exists(PDDD_WEIGHTS):
        print(f"\n[ERROR] PDDD weights not found: {PDDD_WEIGHTS}"); sys.exit(1)

    print("\n[INFO] Verifying class folders...")
    all_ok = True
    for cls in TASK["class_names"]:
        p = Path(DATA_ROOT) / cls
        if p.exists():
            n = sum(1 for f in p.rglob("*")
                    if f.suffix.lower() in {".jpg",".jpeg",".png"})
            print(f"  ✓ {cls:<20} {n} images")
        else:
            print(f"  ✗ {cls:<20} NOT FOUND"); all_ok = False

    if not all_ok:
        print(f"\n[ERROR] Missing folders. Actual contents:")
        os.system(f"ls '{DATA_ROOT}'"); sys.exit(1)

    if torch.cuda.is_available():
        print(f"\n[INFO] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("\n[WARN] CUDA not available — CPU training will be slow.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    preflight()

    from old.oldYOLO26s3.teacher_train import TeacherTrainer, generate_ensemble_soft_labels
    from old.oldYOLO26s3.teacher_data  import discover_images, get_tta_transforms

    trained_models, all_f1s, all_gates = [], [], []

    for run_idx, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n\n{'='*70}")
        print(f"  ENSEMBLE RUN {run_idx+1}/{len(ENSEMBLE_SEEDS)}  —  SEED {seed}")
        print(f"{'='*70}")
        set_seed(seed)

        sdir = rp(f"outputs/pest/seed_{seed}")
        os.makedirs(sdir, exist_ok=True)

        trainer = TeacherTrainer(
            data_config=data_cfg, model_config=model_cfg,
            train_config=make_train_cfg(sdir),
            class_names=TASK["class_names"],
        )
        f1, gate = trainer.train()
        trained_models.append(trainer.model)
        all_f1s.append(f1)
        all_gates.append(gate)
        print(f"\n[SEED {seed}] Test Macro F1 = {f1:.4f}")

    mean_f1 = sum(all_f1s) / len(all_f1s)
    print(f"\n  Mean ensemble F1: {mean_f1:.4f}")

    # Soft label generation
    raw_map  = discover_images(DATA_ROOT, TASK["class_names"], False)
    all_imgs = defaultdict(list)
    for cls in TASK["class_names"]:
        for sp in ["all","train","val","test"]:
            all_imgs[cls].extend(raw_map[cls].get(sp, []))

    tta_n  = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = generate_ensemble_soft_labels(
        models         = trained_models,
        image_dir_map  = dict(all_imgs),
        class_names    = TASK["class_names"],
        image_size     = data_cfg.image_size,
        task_name      = TASK["task_name"],
        output_dir     = OUTPUT_DIR,
        device         = device,
        tta_transforms = get_tta_transforms(data_cfg.image_size, n=tta_n),
    )

    summary = {
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "class_names": TASK["class_names"],
        "per_seed_test_f1": dict(zip(ENSEMBLE_SEEDS, all_f1s)),
        "mean_test_f1": mean_f1,
        "soft_labels_csv": csv_path,
    }
    sp = os.path.join(OUTPUT_DIR, "ensemble_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Soft labels: {csv_path}")
    print(f"  Summary:     {sp}")
    print("\n  NEXT STEPS: python run_fruit_teacher.py | python run_leaf_teacher.py")


if __name__ == "__main__":
    main()