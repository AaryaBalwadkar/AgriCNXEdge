"""
teacher_config.py  (FIXED — v4)
================================
Changes from v3:
  - FRUIT_TASK class_names updated to match actual folder names
    "Black Rot" uses a SPACE (matching the original folder name)
  - ModelConfig backbone_name and pddd_num_classes are now labelled
    as IGNORED — teacher_model.py v4 auto-detects these from checkpoint
"""

from dataclasses import dataclass
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

    # ── NEW: sampler softening ────────────────────────────────────────────
    # WeightedRandomSampler weight = count^(-sampler_power).
    #   power=1.0 -> old behaviour (full inverse frequency, exact class balance
    #                per batch). For small minority classes this means the
    #                same handful of images get replayed many times per
    #                epoch, which drives memorisation/overfitting.
    #   power=0.5 -> softened (sqrt inverse frequency). Minority classes are
    #                still boosted but not blown out 4-5x. Recommended default.
    #   power=0.0 -> plain uniform sampling (no reweighting at all).
    sampler_power: float = 0.5
    rare_class_threshold: int = 400
    oversample_factor: int = 1
    rare_oversample_factor: int = 2


@dataclass
class ModelConfig:
    pddd_weights_path: str = ""
    # NOTE: backbone_name and pddd_num_classes below are AUTO-DETECTED
    # by teacher_model.py v4 from the checkpoint itself.
    # These values are kept only as documentation/fallback.
    pddd_num_classes: int = 120       # PDDD dataset has 120 classes
    backbone_name: str = "convnext_small"
    feature_dim: int = 768
    dropout_rate: float = 0.5
    # ── NEW: stochastic depth on the ConvNeXt backbone ────────────────────
    # Randomly drops whole residual blocks during training. Standard
    # regulariser for transfer-learned ConvNeXt on small datasets; was
    # previously unset (defaulted to 0, i.e. off).
    drop_path_rate: float = 0.1


@dataclass
class TrainConfig:
    output_dir: str = ""
    task_name: str = "pest"
    num_classes: int = 4

    epochs: int = 60
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    batch_size: int = 32
    num_workers: int = 4

    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    eta_min: float = 1e-7
    focal_gamma: float = 2.0
    # ── CHANGED: was 0.0. Small amount of label smoothing softens the
    # hard-label targets, which fights the overconfidence/overfitting seen
    # in the val-loss curve (val loss bottomed at epoch ~14 then climbed
    # back up while train loss kept falling).
    label_smoothing: float = 0.1
    patience: int = 15
    min_delta: float = 1e-4
    quality_gate_f1: float = 0.70
    save_top_k: int = 1
    soft_label_temperature: float = 1.0

    # ── NEW: Mixup / CutMix ────────────────────────────────────────────────
    # Applied per-batch (with probability mixup_prob). Directly targets the
    # overfitting caused by the WeightedRandomSampler replaying the same
    # minority-class images many times per epoch -- mixed samples are never
    # bit-identical twice even if the same source images get drawn again.
    use_mixup: bool = True
    mixup_alpha: float = 0.2      # mixup interpolation strength
    cutmix_alpha: float = 1.0     # cutmix box strength
    mixup_prob: float = 0.5       # chance a given batch gets mixed at all

    # ── NEW: Test-time augmentation (hflip) ─────────────────────────────────
    # Used for final test-set evaluation and soft-label generation. Cheap,
    # typically +0.5-1.5 F1 points, and smooths the logits used for KD.
    use_tta: bool = True

    # ── NEW: Post-hoc temperature scaling ───────────────────────────────────
    # Fits a single scalar T on the val set (minimising NLL) after training,
    # then divides all exported soft-label logits by T. Calibration matters
    # a lot for a distillation *teacher* -- KD trains the student against
    # softmax(logits/T_kd), so a miscalibrated teacher passes on miscalibrated
    # "dark knowledge" even if its top-1 accuracy looks fine.
    calibrate_temperature: bool = True


# ===========================================================================
#  TASK CONFIGS
# ===========================================================================

# ── PEST (4 classes — melaybug and whitefly folders absent) ─────────────────
# Folder names on Linux must match EXACTLY (case-sensitive, no spaces)
PEST_TASK = {
    "task_name": "pest",
    "num_classes": 4,
    "class_names": [
        "xylotrechus",    # data/pest/xylotrechus/  — 296 clean images
        "aphids",         # data/pest/aphids/        — 271 clean images
        "leafhoppers",    # data/pest/leafhoppers/   — 567 clean images
        "spider_mite",    # data/pest/spider_mite/   — 271 clean images
    ],
}

# ── LEAF (5 classes) ─────────────────────────────────────────────────────────
# Verify with: ls data/leaf/
LEAF_TASK = {
    "task_name": "leaf",
    "num_classes": 5,
    "class_names": [
        "Apple_Mosaic",
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Alternaria",
        "Healthy",
    ],
}

# ── FRUIT (3 classes) ────────────────────────────────────────────────────────
# IMPORTANT: "Black Rot" has a SPACE — verify with: ls data/fruit/
# If your folder is "Black_Rot" (underscore), change it here.
# Blotch and Scab are removed to keep the teacher focused on more reliable classes.
FRUIT_TASK = {
    "task_name": "fruit",
    "num_classes": 3,
    "class_names": [
        "Anthracnose",  # data/fruit/Anthracnose/ — 295 images
        "Black Rot",    # data/fruit/Black Rot/   — 542 images (SPACE in name)
        "Healthy",      # data/fruit/Healthy/     — 1424 images
    ],
}

# ===========================================================================
#  CLIENT-SIDE MAPPINGS (shipped with app, not encoded in model weights)
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