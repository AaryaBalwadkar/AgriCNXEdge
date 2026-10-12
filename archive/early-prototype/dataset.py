"""
dataset.py — Multi-Task Dataset & DataLoader for Agri-CNX-Edge
==============================================================
Handles multi-root YOLO-style directory structure for 4 tasks:
- Tasks 1, 2, 3: Image Classification (EfficientNetV2-style heads)
- Task 4: Dense Object Detection (YOLO-Nano-style head)

Features:
- Reads standardized data: data/<task>/[train|val|test]/images/, labels/
- For classification: reads first int from .txt label file (empty = background/healthy class)
- For detection: reads YOLO format (class_id cx cy w h)
- Returns -1 for missing tasks per sample (ignored in loss)
- Integrates albumentations for edge-degradation (train only)

Author: AgriEdge Project
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

TASKS = [
    "task1_disease",
    "task2_virus",
    "task3_pest",
    "task4_yield"
]

SPLITS = ["train", "val", "test"]

# Map task index for output ordering
TASK_IDX = {name: i for i, name in enumerate(TASKS)}

class AgriMultiTaskDataset(Dataset):
    """
    Multi-task dataset for Agri-CNX-Edge. Handles 4 tasks with multi-root structure.
    Returns a dict with keys: 'images', 'labels' (list: [cls1, cls2, cls3, det4])
    """
    def __init__(self, data_root, split="train", image_size=640, augment=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.samples = self._gather_samples()
        self.transform = self._build_transform()

    def _gather_samples(self):
        # Collect all images for all tasks, build a master list
        sample_dict = {}
        for task in TASKS:
            img_dir = os.path.join(self.data_root, task, self.split, "images")
            label_dir = os.path.join(self.data_root, task, self.split, "labels")
            if not os.path.isdir(img_dir):
                continue
            for img_path in glob.glob(os.path.join(img_dir, "*")):
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, base + ".txt")
                sample_dict.setdefault(img_path, {})[task] = label_path
        # Each sample: {img_path: {task: label_path, ...}}
        samples = []
        for img_path, task_labels in sample_dict.items():
            entry = {"img": img_path}
            for task in TASKS:
                entry[task] = task_labels.get(task, None)
            samples.append(entry)
        return samples

    def _build_transform(self):
        if self.augment and self.split == "train":
            return A.Compose([
                A.MotionBlur(blur_limit=(3, 9), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ImageCompression(quality_lower=40, quality_upper=85, p=0.3),
                A.Resize(self.image_size, self.image_size),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img = np.array(Image.open(entry["img"]).convert("RGB"))
        img = self.transform(image=img)["image"]
        # Prepare per-task outputs
        labels = [-1, -1, -1, None]  # [cls1, cls2, cls3, det4]
        for i, task in enumerate(TASKS):
            label_path = entry[task]
            if label_path is None or not os.path.isfile(label_path):
                continue
            if task != "task4_yield":
                # Classification: read first int, empty = 0 (background)
                try:
                    with open(label_path, "r") as f:
                        line = f.readline().strip()
                        if line == "":
                            labels[i] = 0  # background/healthy
                        else:
                            labels[i] = int(line.split()[0])
                except Exception:
                    labels[i] = 0
            else:
                # Detection: read all YOLO rows (class_id cx cy w h)
                det_targets = []
                try:
                    with open(label_path, "r") as f:
                        for row in f:
                            vals = row.strip().split()
                            if len(vals) == 5:
                                det_targets.append([float(x) for x in vals])
                except Exception:
                    pass
                labels[3] = np.array(det_targets, dtype=np.float32) if det_targets else np.zeros((0,5), dtype=np.float32)
        return {
            "image": img,
            "labels": labels  # [cls1, cls2, cls3, det4]
        }

def agri_multitask_dataloader(data_root, split="train", image_size=640, batch_size=16, augment=False, num_workers=4):
    ds = AgriMultiTaskDataset(data_root, split, image_size, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True, drop_last=False)
