"""
teacher_data.py  (v2 — Minority Augmentation + TTA Transforms)
==============================================================
New additions:
  get_minority_train_transform() : heavier augmentation for minority classes
  get_tta_transforms()           : 8 deterministic TTA pipelines
  TeacherDataset                 : now supports per-label transform routing
  build_dataloaders              : auto-detects minority classes, applies heavy aug
"""

import os
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("[WARN] imagehash not installed. Run: pip install imagehash")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]


# ===========================================================================
#  DATA CLEANING  (unchanged from v1)
# ===========================================================================

def clean_images(image_paths, min_dim=64, min_pixel_std=5.0,
                 phash_threshold=8, verbose=True):
    report = {"too_small": [], "blank": [], "duplicate": [],
              "unreadable": [], "original_count": len(image_paths)}
    seen_hashes: Dict[str, str] = {}
    cleaned: List[str] = []

    for path in image_paths:
        try:
            img = Image.open(path)
            img.load()
            w, h = img.size
            if w < min_dim or h < min_dim:
                report["too_small"].append(path); continue
            arr = np.array(img.convert("RGB"), dtype=np.float32)
            if arr.std() < min_pixel_std:
                report["blank"].append(path); continue
            if IMAGEHASH_AVAILABLE:
                phash = imagehash.phash(img)
                dup = False
                for hs in seen_hashes:
                    if abs(phash - imagehash.hex_to_hash(hs)) <= phash_threshold:
                        report["duplicate"].append(path); dup = True; break
                if dup: continue
                seen_hashes[str(phash)] = path
            cleaned.append(path)
        except Exception:
            report["unreadable"].append(path)

    report["cleaned_count"] = len(cleaned)
    report["removed_count"]  = len(image_paths) - len(cleaned)

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
#  IMAGE DISCOVERY  (unchanged from v1)
# ===========================================================================

def discover_images(data_root, class_names, use_existing_split=False):
    root   = Path(data_root)
    result = defaultdict(lambda: defaultdict(list))

    if not use_existing_split:
        for cls in class_names:
            cls_dir = root / cls
            if not cls_dir.exists():
                print(f"[WARN] Class folder not found: {cls_dir}")
                continue
            for f in cls_dir.rglob("*"):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result[cls]["all"].append(str(f))
    else:
        for split in ["train", "val", "test"]:
            split_dir = root / split
            if not split_dir.exists(): continue
            for cls in class_names:
                cls_dir = split_dir / cls
                if not cls_dir.exists(): continue
                for f in cls_dir.rglob("*"):
                    if f.suffix.lower() in SUPPORTED_EXTS:
                        result[cls][split].append(str(f))

    return result


# ===========================================================================
#  STRATIFIED SPLIT  (unchanged from v1)
# ===========================================================================

def stratified_split(class_image_map, val_split=0.15, test_split=0.15,
                     seed=42, verbose=True):
    rng = random.Random(seed)
    train_map, val_map, test_map = {}, {}, {}

    if verbose:
        print("\n" + "=" * 60)
        print("  STRATIFIED SPLIT SUMMARY")
        print("=" * 60)
        print(f"  {'Class':<30} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
        print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")

    for cls, paths in class_image_map.items():
        paths = list(paths); rng.shuffle(paths); n = len(paths)
        if n < 3:
            train_map[cls] = paths; val_map[cls] = []; test_map[cls] = []
            if verbose:
                print(f"  {cls:<30} {n:>6} {n:>6} {'0':>5} {'0':>5}  ← CRITICAL: only {n} images")
        else:
            nt = max(1, int(n * test_split))
            nv = max(1, int(n * val_split))
            nr = n - nv - nt
            train_map[cls] = paths[:nr]
            val_map[cls]   = paths[nr:nr+nv]
            test_map[cls]  = paths[nr+nv:]
            if verbose:
                flag = "  ← WARNING: < 20 images" if n < 20 else ""
                print(f"  {cls:<30} {n:>6} {nr:>6} {nv:>5} {nt:>5}{flag}")

    if verbose:
        tr = sum(len(v) for v in train_map.values())
        va = sum(len(v) for v in val_map.values())
        te = sum(len(v) for v in test_map.values())
        print(f"  {'TOTAL':<30} {tr+va+te:>6} {tr:>6} {va:>5} {te:>5}")
        print("=" * 60 + "\n")

    return train_map, val_map, test_map


# ===========================================================================
#  AUGMENTATION TRANSFORMS
# ===========================================================================

def get_train_transform(image_size: int, has_rare_classes: bool = False) -> A.Compose:
    """Standard training augmentation."""
    ops = [
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.65, 1.0),
                            ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
    ]
    if has_rare_classes:
        ops.append(A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=0, p=0.4
        ))
    ops += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


def get_minority_train_transform(image_size: int) -> A.Compose:
    """
    Heavier augmentation for minority classes (< minority_aug_threshold training samples).
    More aggressive than standard: wider crop scale, stronger color, GridDistortion,
    heavier CoarseDropout — effectively multiplies minority class visual diversity.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.50, 1.0),
                            ratio=(0.65, 1.50)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, p=0.6),
        A.ColorJitter(brightness=0.4, contrast=0.4,
                      saturation=0.4, hue=0.15, p=0.7),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.35, p=1.0),
            A.OpticalDistortion(distort_limit=0.4, p=1.0),
        ], p=0.4),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(16, 64),
            hole_width_range=(16, 64),
            fill=0, p=0.55
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int) -> A.Compose:
    """Validation / test — no augmentation."""
    resize_to = int(image_size * 256 / 224)
    return A.Compose([
        A.Resize(resize_to, resize_to),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int, n: int = 8) -> List[A.Compose]:
    """
    Returns n deterministic TTA transform pipelines.
    Each pipeline applies a fixed (non-random) augmentation then normalises.
    Results from all n pipelines are averaged in probability space.

    8 views:
      1. Original (just resize + crop + normalise)
      2. Horizontal flip
      3. Vertical flip
      4. Horizontal + Vertical flip
      5. +15° rotation
      6. -15° rotation
      7. +30° rotation
      8. -30° rotation
    """
    rs = int(image_size * 256 / 224)
    base  = [A.Resize(rs, rs), A.CenterCrop(image_size, image_size)]
    norm  = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    all_t = [
        A.Compose([*base, *norm]),                                              # 1 original
        A.Compose([A.HorizontalFlip(p=1.0), *base, *norm]),                    # 2 h-flip
        A.Compose([A.VerticalFlip(p=1.0), *base, *norm]),                      # 3 v-flip
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), *base, *norm]),  # 4 both flips
        A.Compose([A.Rotate(limit=(15, 15), p=1.0), *base, *norm]),            # 5 +15°
        A.Compose([A.Rotate(limit=(-15, -15), p=1.0), *base, *norm]),          # 6 -15°
        A.Compose([A.Rotate(limit=(30, 30), p=1.0), *base, *norm]),            # 7 +30°
        A.Compose([A.Rotate(limit=(-30, -30), p=1.0), *base, *norm]),          # 8 -30°
    ]
    return all_t[:n]


# ===========================================================================
#  DATASET
# ===========================================================================

class TeacherDataset(Dataset):
    """
    Universal classification dataset.
    Supports per-label transform routing: minority classes can receive a
    different (heavier) augmentation than majority classes.
    """

    def __init__(
        self,
        class_image_map: Dict[str, List[str]],
        class_names: List[str],
        transform: Optional[A.Compose] = None,
        minority_transform: Optional[A.Compose] = None,
        minority_label_set: Optional[Set[int]] = None,
    ):
        self.class_names       = class_names
        self.class_to_idx      = {c: i for i, c in enumerate(class_names)}
        self.transform         = transform
        self.minority_transform = minority_transform
        self.minority_label_set = minority_label_set or set()

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
            img_np = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)

        # Route to heavy augmentation for minority classes
        if (self.minority_transform is not None
                and label in self.minority_label_set):
            t = self.minority_transform
        else:
            t = self.transform

        if t:
            transformed = t(image=img_np)
            img_tensor = transformed["image"]
        else:
            img_tensor = torch.from_numpy(img_np).float() / 255.0

        # Ensure image tensor is always CHW and float
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)

        if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

        img_tensor = img_tensor.float()

        return {"image": img_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "path":  path,
                "image_np": img_np}

    def get_labels(self) -> List[int]:
        return [s[1] for s in self.samples]


# ===========================================================================
#  WEIGHTED SAMPLER  (unchanged from v1)
# ===========================================================================

def build_weighted_sampler(labels, num_classes):
    counts = Counter(labels)
    cw = {cls: 1.0 / max(count, 1) for cls, count in counts.items()}
    for c in range(num_classes):
        if c not in cw:
            cw[c] = 1.0
    sample_weights = [cw[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ===========================================================================
#  MAIN PIPELINE
# ===========================================================================

def build_dataloaders(
    data_config,
    class_names: List[str],
    num_classes: int,
    task_name: str,
    output_dir: str,
    minority_aug_threshold: int = 400,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:

    os.makedirs(output_dir, exist_ok=True)

    # 1. Discover
    print(f"\n[INFO] Discovering images in: {data_config.data_root}")
    raw_map = discover_images(data_config.data_root, class_names,
                              use_existing_split=data_config.use_existing_split)

    # 2. Clean + split
    cleaning_reports = {}

    if not data_config.use_existing_split:
        cleaned_all: Dict[str, List[str]] = {}
        for cls in class_names:
            raw = raw_map[cls].get("all", [])
            if not raw:
                print(f"[WARN] No images found for class '{cls}'")
                cleaned_all[cls] = []
                continue
            cl, rep = clean_images(raw, data_config.min_image_dim,
                                   data_config.min_pixel_std,
                                   data_config.phash_threshold, verbose=True)
            cleaned_all[cls] = cl
            cleaning_reports[cls] = rep
            print(f"  [{cls}] {len(raw)} raw → {len(cl)} clean")

        train_map, val_map, test_map = stratified_split(
            cleaned_all, data_config.val_split, data_config.test_split,
            data_config.seed, verbose=True)
    else:
        train_map, val_map, test_map = {}, {}, {}
        for cls in class_names:
            train_map[cls] = raw_map[cls].get("train", [])
            val_map[cls]   = raw_map[cls].get("val",   [])
            test_map[cls]  = raw_map[cls].get("test",  [])
        cleaning_reports = {c: {"skipped": "pre-split mode"} for c in class_names}

    # 3. Save cleaning report
    rp = os.path.join(output_dir, f"{task_name}_cleaning_report.json")
    with open(rp, "w") as f:
        json.dump({k: {k2: len(v2) if isinstance(v2, list) else v2
                       for k2, v2 in v.items()}
                   for k, v in cleaning_reports.items()}, f, indent=2)
    print(f"[INFO] Cleaning report saved to: {rp}")

    # 4. Detect minority classes
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    minority_label_set: Set[int] = set()
    for cls, paths in train_map.items():
        if len(paths) < minority_aug_threshold and cls in class_to_idx:
            minority_label_set.add(class_to_idx[cls])

    has_rare = any(len(v) < 30 for v in train_map.values())
    has_minority = len(minority_label_set) > 0

    if has_minority:
        minority_cls = [c for c, i in class_to_idx.items() if i in minority_label_set]
        print(f"[INFO] Minority classes (< {minority_aug_threshold} train images): {minority_cls}")
        print(f"[INFO] Applying heavy augmentation to these classes.")

    # 5. Build transforms
    train_transform    = get_train_transform(data_config.image_size,
                                             has_rare_classes=has_rare)
    minority_transform = get_minority_train_transform(data_config.image_size) \
                         if has_minority else None
    val_transform      = get_val_transform(data_config.image_size)

    # 6. Build datasets
    train_ds = TeacherDataset(train_map, class_names,
                              transform=train_transform,
                              minority_transform=minority_transform,
                              minority_label_set=minority_label_set)
    val_ds   = TeacherDataset(val_map,   class_names, transform=val_transform)
    test_ds  = TeacherDataset(test_map,  class_names, transform=val_transform)

    print(f"\n[INFO] Dataset sizes:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Val   : {len(val_ds)}")
    print(f"  Test  : {len(test_ds)}")

    # 7. Sampler
    sampler = build_weighted_sampler(train_ds.get_labels(), num_classes)

    # 8. DataLoaders
    pin = data_config.pin_memory and torch.cuda.is_available()
    nw  = data_config.num_workers

    train_loader = DataLoader(train_ds, batch_size=data_config.batch_size,
                              sampler=sampler, num_workers=nw,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=data_config.batch_size * 2,
                              shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=data_config.batch_size * 2,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    metadata = {
        "train_size": len(train_ds), "val_size": len(val_ds),
        "test_size":  len(test_ds),
        "has_rare_classes":    has_rare,
        "has_minority_classes": has_minority,
        "minority_label_set":  list(minority_label_set),
        "class_train_counts":  {c: len(v) for c, v in train_map.items()},
        "class_val_counts":    {c: len(v) for c, v in val_map.items()},
        "class_test_counts":   {c: len(v) for c, v in test_map.items()},
    }

    return train_loader, val_loader, test_loader, metadata
