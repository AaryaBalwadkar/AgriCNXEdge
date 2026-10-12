"""
teacher_config.py  (v5 — Two-Stage + Warmup + Label Smoothing + TTA + Ensemble)
================================================================================
New parameters added:
  TrainConfig:
    freeze_epochs      : epochs to train head only (Phase 1)
    warmup_epochs      : linear warmup epochs at start of Phase 2
    lr_head_phase2     : head LR in Phase 2 (lower than Phase 1 since head is pre-warmed)
    label_smoothing    : prevents overconfidence on training examples
    patience           : increased to 20 (warm restarts need more room)
    epochs             : increased to 100
    use_tta            : apply test-time augmentation at evaluation + soft-label gen
    tta_n              : number of TTA augmentations (8 recommended)
    minority_aug_threshold : classes with fewer training samples get heavier augmentation
  ensemble_seeds       : module-level list of seeds for ensemble training
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    data_root: str = ""
    use_existing_split: bool = False
    task_name: str = "pest"
    image_size: int = 224
    min_image_dim: int = 64
    min_pixel_std: float = 5.0
    phash_threshold: int = 8
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    pddd_weights_path: str = ""
    pddd_num_classes: int = 120       # Auto-detected from checkpoint; kept for docs
    backbone_name: str = "convnext_small"
    feature_dim: int = 768
    dropout_rate: float = 0.5


@dataclass
class TrainConfig:
    output_dir: str = ""
    task_name: str = "pest"
    num_classes: int = 4

    # ── Duration ──────────────────────────────────────────────────────────────
    epochs: int = 100               # Increased from 60

    # ── Two-stage training ────────────────────────────────────────────────────
    freeze_epochs: int = 8          # Phase 1: freeze backbone, train head only
    warmup_epochs: int = 5          # Phase 2: linear warmup before cosine restarts
    lr_head_phase2: float = 1e-4    # Head LR in Phase 2 (lower, head is pre-warmed)

    # ── Learning rates ────────────────────────────────────────────────────────
    lr_backbone: float = 1e-5       # Phase 2 backbone LR
    lr_head: float = 1e-3           # Phase 1 head LR (higher — head is random init)
    weight_decay: float = 1e-4
    eta_min: float = 1e-7

    # ── Gradient ──────────────────────────────────────────────────────────────
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # ── Loss ──────────────────────────────────────────────────────────────────
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1    # Prevents overconfidence; helps Anthracnose precision

    # ── Augmentation ──────────────────────────────────────────────────────────
    minority_aug_threshold: int = 400  # Classes with < N training samples → heavy aug

    # ── Batch ─────────────────────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4

    # ── Early stopping ────────────────────────────────────────────────────────
    patience: int = 20              # Increased from 15 — warm restarts need more room
    min_delta: float = 1e-4

    # ── Quality gate ──────────────────────────────────────────────────────────
    quality_gate_f1: float = 0.70

    # ── TTA (Test-Time Augmentation) ──────────────────────────────────────────
    use_tta: bool = True
    tta_n: int = 8                  # 8 deterministic augmentation views per image

    # ── Soft labels ───────────────────────────────────────────────────────────
    save_top_k: int = 1
    soft_label_temperature: float = 1.0


# ── Ensemble seeds ────────────────────────────────────────────────────────────
# Three independent training runs with different random seeds.
# Their TTA-averaged softmax outputs are averaged to produce the final soft labels.
ENSEMBLE_SEEDS = [42, 123, 456]


# ===========================================================================
#  TASK CONFIGS
# ===========================================================================

PEST_TASK = {
    "task_name": "pest",
    "num_classes": 4,
    "class_names": [
        "xylotrechus",
        "aphids",
        "leafhoppers",
        "spider_mite",
    ],
}

LEAF_TASK = {
    "task_name": "leaf",
    "num_classes": 4,
    "class_names": [
        "Apple_Mosaic",
        "Apple___Black_rot",
        "Alternaria",
        "Healthy",
    ],
}

FRUIT_TASK = {
    "task_name": "fruit",
    "num_classes": 3,
    "class_names": [
        "Anthracnose",
        "Black Rot",
        "Healthy",
    ],
}


# ===========================================================================
#  CLIENT-SIDE MAPPINGS
# ===========================================================================

PEST_TO_DISEASE_MAP = {
    "xylotrechus": ["Stem/wood damage", "Structural weakening of apple branches"],
    "aphids":      ["Apple Mosaic Virus (vector)", "Powdery mildew (stress trigger)",
                    "Sooty mold (secondary from honeydew)"],
    "leafhoppers": ["Phytoplasma diseases", "Apple proliferation", "Leaf burn"],
    "spider_mite": ["Leaf bronzing", "Premature defoliation", "Reduced photosynthesis"],
}

FRUIT_DISEASE_INFO = {
    "Anthracnose": "Fungal (Colletotrichum). Dark, sunken lesions. Remove infected fruit.",
    "Black Rot":   "Fungal (Botryosphaeria). Brown rot spreading from core outward.",
    "Healthy":     "No disease detected.",
}
