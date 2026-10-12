"""
run_fruit_teacher.py  (v2 — Ensemble: 3 seeds × 8 TTA views = 24 passes/image)
================================================================================
Trains 3 independent fruit disease teachers (seeds 42, 123, 456).
Each uses two-stage fine-tuning + warmup + label smoothing + minority augmentation.
After all 3 complete, generates ensemble + TTA soft labels for every image.

USAGE:
  python run_fruit_teacher.py

EXPECTED RUNTIME on RTX A4000:
  ~3 hours per seed × 3 seeds = ~9 hours total
  Soft label generation: ~5-10 minutes additional

OUTPUTS (in outputs/fruit/):
  seed_42/fruit_best_teacher.pth
  seed_123/fruit_best_teacher.pth
  seed_456/fruit_best_teacher.pth
  fruit_soft_labels.csv          ← ensemble + TTA averaged probabilities
  ensemble_summary.json          ← per-seed F1 scores + gate results
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ── Path resolution ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

def rp(s: str) -> str:
    p = Path(os.path.expanduser(s))
    return str((BASE_DIR / p).resolve() if not p.is_absolute() else p.resolve())

DATA_ROOT    = rp("data/fruit")
PDDD_WEIGHTS = rp("weights/convnext_small-model-99.pth")
OUTPUT_DIR   = rp("outputs/fruit")

# ── Task + config ─────────────────────────────────────────────────────────────
from old.oldYOLO26s3.teacher_config import FRUIT_TASK as TASK, ENSEMBLE_SEEDS
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
    seed               = 42,           # data split seed (fixed, same for all runs)
    batch_size         = 32,
    num_workers        = 4,
    pin_memory         = True,
)

model_cfg = ModelConfig(
    pddd_weights_path = PDDD_WEIGHTS,
    pddd_num_classes  = 120,           # auto-detected from checkpoint
    backbone_name     = "convnext_small",
    feature_dim       = 768,
    dropout_rate      = 0.5,
)


def make_train_cfg(seed_output_dir: str) -> TrainConfig:
    """Creates a TrainConfig for one ensemble run."""
    return TrainConfig(
        output_dir             = seed_output_dir,
        task_name              = TASK["task_name"],
        num_classes            = TASK["num_classes"],

        # ── Duration ──────────────────────────────────────────────────────
        epochs                 = 100,
        freeze_epochs          = 8,         # Phase 1: head only
        warmup_epochs          = 5,         # Phase 2: linear warmup

        # ── LR ────────────────────────────────────────────────────────────
        lr_head                = 1e-3,      # Phase 1 head LR
        lr_head_phase2         = 1e-4,      # Phase 2 head LR (lower — head pre-warmed)
        lr_backbone            = 1e-5,      # Phase 2 backbone LR
        weight_decay           = 1e-4,
        eta_min                = 1e-7,

        # ── Loss ──────────────────────────────────────────────────────────
        focal_gamma            = 3.0,
        label_smoothing        = 0.1,       # prevents Anthracnose overconfidence

        # ── Augmentation ──────────────────────────────────────────────────
        minority_aug_threshold = 400,       # Anthracnose (193) gets heavy aug

        # ── Stopping ──────────────────────────────────────────────────────
        patience               = 20,
        min_delta              = 1e-4,
        quality_gate_f1        = 0.70,

        # ── Gradient ──────────────────────────────────────────────────────
        max_grad_norm          = 1.0,

        # ── Loader ────────────────────────────────────────────────────────
        batch_size             = data_cfg.batch_size,
        num_workers            = data_cfg.num_workers,

        # ── TTA ───────────────────────────────────────────────────────────
        use_tta                = True,
        tta_n                  = 8,
    )


# ===========================================================================
#  SEED HELPER
# ===========================================================================

def set_seed(seed: int):
    """Fix all random sources for reproducibility of one ensemble run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ===========================================================================
#  PRE-FLIGHT CHECKS
# ===========================================================================

def preflight():
    print("\n" + "=" * 70)
    print("  FRUIT DISEASE TEACHER — ENSEMBLE TRAINING")
    print(f"  Task    : {TASK['task_name']}")
    print(f"  Classes : {TASK['class_names']}")
    print(f"  Seeds   : {ENSEMBLE_SEEDS}  (one full training run per seed)")
    print(f"  Data    : {DATA_ROOT}")
    print(f"  Weights : {PDDD_WEIGHTS}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 70)

    if not os.path.exists(DATA_ROOT):
        print(f"\n[ERROR] DATA_ROOT not found: {DATA_ROOT}")
        sys.exit(1)

    if not os.path.exists(PDDD_WEIGHTS):
        print(f"\n[ERROR] PDDD weights not found: {PDDD_WEIGHTS}")
        print("Download: https://zenodo.org/records/7890438/files/"
              "convnext_small-model-99.pth?download=1")
        sys.exit(1)

    print("\n[INFO] Verifying class folders...")
    all_ok = True
    for cls in TASK["class_names"]:
        cls_path = Path(DATA_ROOT) / cls
        if cls_path.exists():
            n = sum(1 for f in cls_path.rglob("*")
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            flag = "  ← WARNING: very few images" if n < 50 else ""
            print(f"  ✓ {cls:<20} {n} images{flag}")
        else:
            print(f"  ✗ {cls:<20} FOLDER NOT FOUND")
            all_ok = False

    if not all_ok:
        print(f"\n[ERROR] Class folders missing. Actual contents of {DATA_ROOT}:")
        os.system(f"ls '{DATA_ROOT}'")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("\n[WARN] CUDA not available — training on CPU. Estimated time: 3× longer.")
    else:
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU: {gpu} ({vram:.1f} GB VRAM)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    preflight()

    from old.oldYOLO26s3.teacher_train import TeacherTrainer, generate_ensemble_soft_labels
    from old.oldYOLO26s3.teacher_data  import discover_images, get_tta_transforms

    trained_models  = []
    all_f1s         = []
    all_gate_results = []

    # ── Train one model per seed ──────────────────────────────────────────────
    for run_idx, seed in enumerate(ENSEMBLE_SEEDS):

        print("\n\n" + "=" * 70)
        print(f"  ENSEMBLE RUN {run_idx + 1}/{len(ENSEMBLE_SEEDS)}  —  SEED {seed}")
        print("=" * 70)

        set_seed(seed)

        seed_output_dir = rp(f"outputs/fruit/seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        train_cfg = make_train_cfg(seed_output_dir)

        trainer = TeacherTrainer(
            data_config  = data_cfg,
            model_config = model_cfg,
            train_config = train_cfg,
            class_names  = TASK["class_names"],
        )

        test_f1, gate = trainer.train()

        trained_models.append(trainer.model)
        all_f1s.append(test_f1)
        all_gate_results.append(gate)

        print(f"\n[SEED {seed}] Test Macro F1 = {test_f1:.4f}")
        for cls, info in gate.items():
            m = "✓" if info["status"] == "SAFE" else "✗"
            print(f"  {m} {cls:<25} F1={info['f1']:.4f} [{info['status']}]")

    # ── Ensemble summary ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  ENSEMBLE TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    for seed, f1 in zip(ENSEMBLE_SEEDS, all_f1s):
        print(f"  Seed {seed:>3d}  →  Test Macro F1 = {f1:.4f}")
    mean_f1 = sum(all_f1s) / len(all_f1s)
    print(f"  Mean F1 across seeds: {mean_f1:.4f}")

    # ── Check that at least one model passed quality gate ─────────────────────
    any_safe = any(
        all(info["status"] == "SAFE" for info in gate.values())
        for gate in all_gate_results
    )
    if not any_safe:
        print("\n[WARNING] No single model passed the quality gate for ALL classes.")
        print("  Generating ensemble soft labels anyway — averaged probabilities")
        print("  from multiple models are more robust than any single failed model.")
        proceed = input("  Proceed with ensemble soft label generation? (y/n): ").strip().lower()
        if proceed != "y":
            print("[INFO] Soft label generation skipped.")
            _save_ensemble_summary(all_f1s, all_gate_results, None)
            return
    else:
        print("\n[INFO] At least one model passed all quality gates. ✓")
        print("[INFO] Generating ensemble + TTA soft labels...")

    # ── Build full image map for soft label generation ────────────────────────
    raw_map  = discover_images(DATA_ROOT, TASK["class_names"],
                               use_existing_split=False)
    all_imgs = defaultdict(list)
    for cls in TASK["class_names"]:
        for split in ["all", "train", "val", "test"]:
            all_imgs[cls].extend(raw_map[cls].get(split, []))

    # ── TTA transforms ────────────────────────────────────────────────────────
    tta_n          = make_train_cfg(OUTPUT_DIR).tta_n
    tta_transforms = get_tta_transforms(data_cfg.image_size, n=tta_n)
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Generate ensemble + TTA soft labels ───────────────────────────────────
    csv_path = generate_ensemble_soft_labels(
        models         = trained_models,
        image_dir_map  = dict(all_imgs),
        class_names    = TASK["class_names"],
        image_size     = data_cfg.image_size,
        task_name      = TASK["task_name"],
        output_dir     = OUTPUT_DIR,
        device         = device,
        tta_transforms = tta_transforms,
    )

    # ── Save ensemble summary JSON ────────────────────────────────────────────
    _save_ensemble_summary(all_f1s, all_gate_results, csv_path)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FRUIT TEACHER ENSEMBLE — DONE")
    print("=" * 70)
    for seed, f1, gate in zip(ENSEMBLE_SEEDS, all_f1s, all_gate_results):
        print(f"  Seed {seed}:")
        print(f"    Test Macro F1: {f1:.4f}")
        for cls, info in gate.items():
            m = "✓" if info["status"] == "SAFE" else "✗"
            print(f"    {m} {cls:<25} F1={info['f1']:.4f}")
    print(f"\n  Mean Ensemble F1    : {mean_f1:.4f}")
    print(f"  Soft labels CSV     : {csv_path}")
    print(f"  Method              : {len(ENSEMBLE_SEEDS)} models × {tta_n} TTA = "
          f"{len(ENSEMBLE_SEEDS)*tta_n} passes/image")
    print("=" * 70)
    print("\n  NEXT STEPS:")
    print("  1. Soft labels CSV is ready for the student distillation pipeline.")
    print("     Note: CSV stores averaged PROBABILITIES (prob_0..C), not logits.")
    print("     Student should use these directly at T=1 (no temperature scaling).")
    print("  2. To train the LEAF teacher:  python run_leaf_teacher.py")
    print("  3. To train the PEST teacher:  python run_pest_teacher.py")


def _save_ensemble_summary(all_f1s, all_gate_results, csv_path):
    summary = {
        "ensemble_seeds":      ENSEMBLE_SEEDS,
        "class_names":         TASK["class_names"],
        "per_seed_test_f1":    dict(zip(ENSEMBLE_SEEDS, all_f1s)),
        "mean_test_f1":        sum(all_f1s) / len(all_f1s) if all_f1s else 0,
        "per_seed_gate":       {
            str(seed): gate
            for seed, gate in zip(ENSEMBLE_SEEDS, all_gate_results)
        },
        "soft_labels_csv":     csv_path,
        "soft_label_format":   "prob_0..C (averaged probabilities, use at T=1)",
    }
    sp = os.path.join(OUTPUT_DIR, "ensemble_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Ensemble summary saved to: {sp}")


if __name__ == "__main__":
    main()
