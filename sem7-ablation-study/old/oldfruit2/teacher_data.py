"""
teacher_data.py
===============
Universal dataset pipeline for all three classification teachers.
Handles:
  - Data cleaning  (resolution, blank, near-duplicate via perceptual hash)
  - Stratified splitting (raw class folders OR existing train/val/test)
  - Class-imbalance aware sampling (WeightedRandomSampler)
  - Task-specific augmentation (heavier for rare-class tasks)
  - Dataloader construction
"""

import os
import json
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- imagehash is needed for deduplication --------------------------------
# Install if missing:  pip install imagehash
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("[WARN] imagehash not installed. Perceptual-hash deduplication SKIPPED.")
    print("       Run: pip install imagehash")
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===========================================================================
#  STEP 1: DATA CLEANING
# ===========================================================================

def clean_images(
    image_paths: List[str],
    min_dim: int = 64,
    min_pixel_std: float = 5.0,
    phash_threshold: int = 8,
    verbose: bool = True,
) -> Tuple[List[str], Dict]:
    """
    Runs three cleaning passes on a list of image paths:
      Pass 1 — Minimum resolution (reject < min_dim in any dimension)
      Pass 2 — Blank/near-blank (reject if pixel std < min_pixel_std)
      Pass 3 — Near-duplicate (perceptual hash, Hamming dist <= phash_threshold)

    Returns:
      cleaned_paths : list of paths that passed all checks
      report        : dict with removed paths per reason
    """
    report = {
        "too_small":    [],
        "blank":        [],
        "duplicate":    [],
        "unreadable":   [],
        "original_count": len(image_paths),
    }

    seen_hashes: Dict[str, str] = {}   # hash_str -> first path seen
    cleaned: List[str] = []

    for path in image_paths:
        try:
            img = Image.open(path)
            img.load()   # Force decode (catches truncated files PIL.verify() misses)
            w, h = img.size

            # --- Pass 1: Minimum resolution ---
            if w < min_dim or h < min_dim:
                report["too_small"].append(path)
                continue

            # --- Pass 2: Blank image ---
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            if arr.std() < min_pixel_std:
                report["blank"].append(path)
                continue

            # --- Pass 3: Perceptual hash deduplication ---
            if IMAGEHASH_AVAILABLE:
                phash = imagehash.phash(img)
                for seen_hash_str, seen_path in seen_hashes.items():
                    seen_hash = imagehash.hex_to_hash(seen_hash_str)
                    if abs(phash - seen_hash) <= phash_threshold:
                        # Store (removed_path, matched_kept_path, hamming_dist)
                        # so the pair can be spot-checked later -- with
                        # phash_threshold=8 it's worth confirming these are
                        # true near-duplicates and not just visually similar
                        # but distinct photos, especially for small classes
                        # where every image counts.
                        report["duplicate"].append(
                            (path, seen_path, int(abs(phash - seen_hash)))
                        )
                        break
                else:
                    seen_hashes[str(phash)] = path
                    cleaned.append(path)
            else:
                cleaned.append(path)

        except (UnidentifiedImageError, OSError, Exception) as e:
            report["unreadable"].append(path)

    report["cleaned_count"] = len(cleaned)
    report["removed_count"] = len(image_paths) - len(cleaned)

    if verbose:
        print("\n" + "=" * 60)
        print("  DATA CLEANING REPORT")
        print("=" * 60)
        print(f"  Original   : {report['original_count']} images")
        print(f"  Too small  : {len(report['too_small'])} removed  (< {min_dim}px)")
        print(f"  Blank      : {len(report['blank'])} removed  (pixel std < {min_pixel_std})")
        print(f"  Duplicates : {len(report['duplicate'])} removed  (phash ≤ {phash_threshold})")
        print(f"  Unreadable : {len(report['unreadable'])} removed")
        print(f"  Retained   : {report['cleaned_count']} images")
        print("=" * 60 + "\n")

    return cleaned, report


# ===========================================================================
#  STEP 2: DISCOVER IMAGES (handles both raw class folders and split folders)
# ===========================================================================

def discover_images(
    data_root: str,
    class_names: List[str],
    use_existing_split: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
      If use_existing_split=False:
        { class_name: {"all": [paths...]} }
      If use_existing_split=True:
        { class_name: {"train": [paths...], "val": [paths...], "test": [paths...]} }
    """
    root = Path(data_root)
    result = defaultdict(lambda: defaultdict(list))

    if not use_existing_split:
        # Raw class folders: data_root/class_name/image.jpg
        for cls in class_names:
            cls_dir = root / cls
            if not cls_dir.exists():
                print(f"[WARN] Class folder not found: {cls_dir}")
                continue
            for f in cls_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result[cls]["all"].append(str(f))
    else:
        # Pre-split folders: data_root/train/class_name/image.jpg
        for split in ["train", "val", "test"]:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for cls in class_names:
                cls_dir = split_dir / cls
                if not cls_dir.exists():
                    continue
                for f in cls_dir.rglob("*"):
                    if f.suffix.lower() in SUPPORTED_EXTS:
                        result[cls][split].append(str(f))

    return result


# ===========================================================================
#  STEP 3: STRATIFIED SPLIT
# ===========================================================================

def stratified_split(
    class_image_map: Dict[str, List[str]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    """
    Splits raw class image lists into train/val/test dicts.
    Each dict maps class_name -> [paths].
    Handles tiny classes: if a class has < 3 images, all go to train.
    """
    rng = random.Random(seed)
    train_map, val_map, test_map = {}, {}, {}

    if verbose:
        print("\n" + "=" * 60)
        print("  STRATIFIED SPLIT SUMMARY")
        print("=" * 60)
        print(f"  {'Class':<30} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")

    for cls, paths in class_image_map.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)

        if n < 3:
            # All go to train; val and test empty — flag warning
            train_map[cls] = paths
            val_map[cls]   = []
            test_map[cls]  = []
            if verbose:
                print(f"  {cls:<30} {n:>6} {n:>6} {'0':>5} {'0':>5}  ← CRITICAL: only {n} images")
        else:
            n_test  = max(1, int(n * test_split))
            n_val   = max(1, int(n * val_split))
            n_train = n - n_val - n_test

            train_map[cls] = paths[:n_train]
            val_map[cls]   = paths[n_train : n_train + n_val]
            test_map[cls]  = paths[n_train + n_val:]

            if verbose:
                flag = "  ← WARNING: < 20 images" if n < 20 else ""
                print(f"  {cls:<30} {n:>6} {n_train:>6} {n_val:>5} {n_test:>5}{flag}")

    if verbose:
        tot   = sum(len(p) for p in train_map.values()) + \
                sum(len(p) for p in val_map.values())   + \
                sum(len(p) for p in test_map.values())
        tr    = sum(len(p) for p in train_map.values())
        va    = sum(len(p) for p in val_map.values())
        te    = sum(len(p) for p in test_map.values())
        print(f"  {'TOTAL':<30} {tot:>6} {tr:>6} {va:>5} {te:>5}")
        print("=" * 60 + "\n")

    return train_map, val_map, test_map


# ===========================================================================
#  STEP 4: PYTORCH DATASET
# ===========================================================================

class TeacherDataset(Dataset):
    """
    Universal classification dataset for PDDD teacher fine-tuning.
    Accepts a dict { class_name -> [image_paths] } and a class_names list
    that defines the canonical label ordering (index = class index).
    """

    def __init__(
        self,
        class_image_map: Dict[str, List[str]],
        class_names: List[str],
        transform: Optional[A.Compose] = None,
    ):
        self.class_names  = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform    = transform

        self.samples: List[Tuple[str, int]] = []
        for cls, paths in class_image_map.items():
            if cls not in self.class_to_idx:
                print(f"[WARN] Class '{cls}' not in class_names — skipping {len(paths)} images")
                continue
            idx = self.class_to_idx[cls]
            for p in paths:
                self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        path, label = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
            img_np = np.array(img)
        except Exception:
            # Return a black image on read failure — rare after cleaning
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path":  path,
        }

    def get_labels(self) -> List[int]:
        """Returns all labels as a list (needed for WeightedRandomSampler)."""
        return [s[1] for s in self.samples]


# ===========================================================================
#  STEP 5: AUGMENTATION TRANSFORMS
# ===========================================================================

def get_train_transform(image_size: int, has_rare_classes: bool = False) -> A.Compose:
    """
    Training augmentation.
    has_rare_classes=True adds CoarseDropout to help regularise low-count classes.
    """
    ops = [
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.55, 1.0),
            ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.15,
            rotate_limit=25, border_mode=0, p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.7),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(p=0.3),
        ], p=0.4),
        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            A.ChannelShuffle(p=0.2),
        ], p=0.3),
    ]
    if has_rare_classes:
        # CoarseDropout forces the model to classify from partial views,
        # which helps when the number of training images is limited.
        ops.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
                p=0.4,
            )
        )
    ops += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


def get_val_transform(image_size: int) -> A.Compose:
    """Validation / Test / Soft-label-generation transform — no augmentation."""
    return A.Compose([
        A.Resize(int(image_size * 256 / 224), int(image_size * 256 / 224)),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ===========================================================================
#  STEP 6: WEIGHTED SAMPLER (handles class imbalance during training)
# ===========================================================================

def build_weighted_sampler(
    labels: List[int],
    num_classes: int,
    power: float = 0.5,
    oversample_factor: int = 1,
) -> WeightedRandomSampler:
    """
    Gives each sample a weight of count^(-power).

    power=1.0 (old behaviour) fully equalises class frequency per epoch. For
    a minority class with e.g. 193 images vs. a majority class with 935, that
    means the same 193 images get replayed ~5x more often than the majority
    class's images, every single epoch -- with `replacement=True` this is a
    direct memorisation risk on exactly the classes that most need to
    generalise well (confirmed here: Anthracnose had high recall / low
    precision, the classic "memorised training images, doesn't generalise
    the decision boundary" signature).

    power=0.5 (new default) still strongly boosts minority classes but the
    boost grows with sqrt(imbalance) instead of linear(imbalance), so a 5x
    imbalance becomes a ~2.2x oversampling instead of a full 5x.
    power=0.0 disables reweighting entirely (plain uniform sampling).
    """
    counts = Counter(labels)
    class_weights = {
        cls: max(count, 1) ** (-power)
        for cls, count in counts.items()
    }
    # Fill in any missing classes with high weight (so they're never starved)
    for cls in range(num_classes):
        if cls not in class_weights:
            class_weights[cls] = 1.0

    sample_weights = [class_weights[l] for l in labels]

    if power > 0:
        eff_ratio = {
            cls: round((counts[cls] * class_weights[cls]), 2)
            for cls in counts
        }
        print(f"[INFO] Sampler power={power}. Relative per-class draw weight "
              f"(unnormalised, count*weight): {eff_ratio}")

    num_samples = len(sample_weights) * max(1, oversample_factor)
    if oversample_factor > 1:
        print(f"[INFO] Oversampling training batches by factor={oversample_factor}. "
              f"Epoch size will be {num_samples} samples.")

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=num_samples,
        replacement=True,
    )


# ===========================================================================
#  MAIN PIPELINE FUNCTION — called by teacher_train.py
# ===========================================================================

def build_dataloaders(
    data_config,
    class_names: List[str],
    num_classes: int,
    task_name: str,
    output_dir: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Full pipeline from raw folder → clean images → split → dataloaders.

    Returns (train_loader, val_loader, test_loader, metadata_dict)
    metadata_dict contains cleaning report and split summary for logging.
    """
    # -----------------------------------------------------------------------
    # 1. Discover all images
    # -----------------------------------------------------------------------
    print(f"\n[INFO] Discovering images in: {data_config.data_root}")
    raw_map = discover_images(
        data_config.data_root,
        class_names,
        use_existing_split=data_config.use_existing_split,
    )

    # -----------------------------------------------------------------------
    # 2. Clean each class's image list
    # -----------------------------------------------------------------------
    cleaning_reports = {}
    cleaned_all: Dict[str, List[str]] = {}

    if not data_config.use_existing_split:
        for cls in class_names:
            raw_paths = raw_map[cls].get("all", [])
            if not raw_paths:
                print(f"[WARN] No images found for class '{cls}'")
                cleaned_all[cls] = []
                continue
            cleaned, report = clean_images(
                raw_paths,
                min_dim=data_config.min_image_dim,
                min_pixel_std=data_config.min_pixel_std,
                phash_threshold=data_config.phash_threshold,
                verbose=True,
            )
            cleaned_all[cls] = cleaned
            cleaning_reports[cls] = report
            print(f"  [{cls}] {len(raw_paths)} raw → {len(cleaned)} clean")
    else:
        # For pre-split data, flatten each split into one list for reporting
        for cls in class_names:
            all_for_cls = []
            for split in ["train", "val", "test"]:
                all_for_cls.extend(raw_map[cls].get(split, []))
            cleaned_all[cls] = all_for_cls   # cleaning still recommended
            cleaning_reports[cls] = {"skipped": "pre-split mode"}

    # -----------------------------------------------------------------------
    # 3. Save cleaning report
    # -----------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{task_name}_cleaning_report.json")
    with open(report_path, "w") as f:
        # Convert to serializable format
        serializable = {
            k: {k2: len(v2) if isinstance(v2, list) else v2
                for k2, v2 in v.items()}
            for k, v in cleaning_reports.items()
        }
        json.dump(serializable, f, indent=2)
    print(f"[INFO] Cleaning report saved to: {report_path}")

    # ── NEW: full duplicate-pair dump for manual spot-checking ─────────────
    # phash_threshold=8 can occasionally flag distinct-but-similar photos as
    # duplicates. This writes every (removed, kept, hamming_distance) triple
    # so it's easy to eyeball whether real data is being thrown away,
    # especially for the smaller classes.
    dup_review_path = os.path.join(output_dir, f"{task_name}_duplicate_review.json")
    dup_review = {
        cls: cleaning_reports[cls].get("duplicate", [])
        for cls in cleaning_reports
        if isinstance(cleaning_reports[cls].get("duplicate"), list)
        and cleaning_reports[cls]["duplicate"]
        and isinstance(cleaning_reports[cls]["duplicate"][0], tuple)
    }
    if dup_review:
        with open(dup_review_path, "w") as f:
            json.dump(dup_review, f, indent=2)
        n_dups = sum(len(v) for v in dup_review.values())
        print(f"[INFO] {n_dups} duplicate pairs written for review to: {dup_review_path}")

    # -----------------------------------------------------------------------
    # 4. Split (or re-use existing split)
    # -----------------------------------------------------------------------
    if not data_config.use_existing_split:
        train_map, val_map, test_map = stratified_split(
            cleaned_all,
            val_split=data_config.val_split,
            test_split=data_config.test_split,
            seed=data_config.seed,
            verbose=True,
        )
    else:
        train_map, val_map, test_map = {}, {}, {}
        for cls in class_names:
            train_map[cls] = raw_map[cls].get("train", [])
            val_map[cls]   = raw_map[cls].get("val",   [])
            test_map[cls]  = raw_map[cls].get("test",  [])

    # -----------------------------------------------------------------------
    # 5. Detect if task has rare classes and enable stronger augmentation
    # -----------------------------------------------------------------------
    rare_class_threshold = getattr(data_config, "rare_class_threshold", 30)
    has_rare = any(len(v) < rare_class_threshold for v in train_map.values())
    if has_rare:
        rare = [c for c, v in train_map.items() if len(v) < rare_class_threshold]
        print(f"[INFO] Rare classes detected (threshold={rare_class_threshold}): {rare}")
        print(f"[INFO] Enabling CoarseDropout augmentation for all classes.")

    # -----------------------------------------------------------------------
    # 6. Build datasets
    # -----------------------------------------------------------------------
    train_ds = TeacherDataset(
        train_map, class_names,
        transform=get_train_transform(data_config.image_size, has_rare_classes=has_rare),
    )
    val_ds = TeacherDataset(
        val_map, class_names,
        transform=get_val_transform(data_config.image_size),
    )
    test_ds = TeacherDataset(
        test_map, class_names,
        transform=get_val_transform(data_config.image_size),
    )

    print(f"\n[INFO] Dataset sizes:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Val   : {len(val_ds)}")
    print(f"  Test  : {len(test_ds)}")

    # -----------------------------------------------------------------------
    # 7. Build weighted sampler for training
    # -----------------------------------------------------------------------
    sampler_power = getattr(data_config, "sampler_power", 0.5)
    oversample_factor = getattr(data_config, "oversample_factor", 1)
    if has_rare:
        oversample_factor = getattr(data_config, "rare_oversample_factor", 2)

    sampler = build_weighted_sampler(
        train_ds.get_labels(), num_classes,
        power=sampler_power,
        oversample_factor=oversample_factor,
    )

    # -----------------------------------------------------------------------
    # 8. Build DataLoaders
    # -----------------------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=data_config.batch_size,
        sampler=sampler,            # WeightedRandomSampler replaces shuffle=True
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_config.batch_size * 2,  # Can use larger batch for eval
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_config.batch_size * 2,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    # Metadata for logging
    metadata = {
        "train_size": len(train_ds),
        "val_size":   len(val_ds),
        "test_size":  len(test_ds),
        "has_rare_classes": has_rare,
        "class_train_counts": {c: len(v) for c, v in train_map.items()},
        "class_val_counts":   {c: len(v) for c, v in val_map.items()},
        "class_test_counts":  {c: len(v) for c, v in test_map.items()},
    }

    return train_loader, val_loader, test_loader, metadata
