"""
teacher_train.py
================
Universal training loop for all three PDDD classification teachers.
Handles:
  - Training with Focal Loss + discriminative LRs
  - Validation with per-class F1 + macro F1 early stopping
  - Quality gate: flags classes below F1 threshold as UNSAFE for distillation
  - Best checkpoint saving
  - Soft label (pre-softmax logits) generation and CSV export
  - Full training log saved to JSON

Run via run_pest_teacher.py / run_leaf_teacher.py / run_fruit_teacher.py
"""

import os
import csv
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

from old.oldfruit1.teacher_model import PDDDTeacher, FocalLoss, count_parameters
from old.oldfruit1.teacher_data import build_dataloaders, get_val_transform, TeacherDataset


# ===========================================================================
#  TRAINING ENGINE
# ===========================================================================

class TeacherTrainer:
    """
    Universal trainer for PDDD ConvNeXt-Small teachers.
    Instantiate once, call .train(), then .generate_soft_labels().
    """

    def __init__(self, data_config, model_config, train_config, class_names: List[str]):
        self.dc = data_config
        self.mc = model_config
        self.tc = train_config
        self.class_names  = class_names
        self.num_classes  = len(class_names)
        self.task_name    = train_config.task_name
        self.output_dir   = train_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # Device
        # ----------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[INFO] Training on: {self.device}")
        if self.device.type == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # ----------------------------------------------------------------
        # Data
        # ----------------------------------------------------------------
        print(f"\n[INFO] Building dataloaders for task: {self.task_name}")
        self.train_loader, self.val_loader, self.test_loader, self.data_meta = \
            build_dataloaders(
                data_config    = data_config,
                class_names    = class_names,
                num_classes    = self.num_classes,
                task_name      = self.task_name,
                output_dir     = self.output_dir,
            )

        # ----------------------------------------------------------------
        # Model
        # ----------------------------------------------------------------
        self.model = PDDDTeacher(
            num_classes       = self.num_classes,
            pddd_weights_path = model_config.pddd_weights_path,
            dropout_rate      = model_config.dropout_rate,
        ).to(self.device)

        param_info = count_parameters(self.model)
        print(f"[INFO] Total params : {param_info['total_M']}")
        print(f"[INFO] Trainable    : {param_info['trainable_M']}")

        # ----------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------
        train_counts = self.data_meta.get("class_train_counts", {})
        if train_counts:
            counts = torch.tensor(
                [train_counts.get(name, 1) for name in self.class_names],
                dtype=torch.float32,
                device=self.device,
            )
            inv_freq = 1.0 / counts.clamp(min=1.0)
            class_weights = inv_freq / inv_freq.mean()
        else:
            class_weights = None

        label_smoothing = getattr(train_config, "label_smoothing", 0.0)
        self.criterion = FocalLoss(
            gamma=train_config.focal_gamma,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        # ----------------------------------------------------------------
        # Optimizer — discriminative learning rates
        # ----------------------------------------------------------------
        param_groups = self.model.get_parameter_groups(
            lr_backbone  = train_config.lr_backbone,
            lr_head      = train_config.lr_head,
            weight_decay = train_config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        backbone_params = sum(p.numel() for p in param_groups[0]["params"])
        head_params = sum(p.numel() for p in param_groups[1]["params"])
        print(f"[INFO] Backbone params: {backbone_params/1e6:.2f}M, Head params: {head_params/1e6:.2f}M")

        # ----------------------------------------------------------------
        # Scheduler — Cosine Annealing over full training duration
        # ----------------------------------------------------------------
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = train_config.epochs,
            eta_min = train_config.eta_min,
        )

        # ----------------------------------------------------------------
        # State tracking
        # ----------------------------------------------------------------
        self.best_macro_f1    = 0.0
        self.best_epoch       = 0
        self.epochs_no_improve = 0
        self.history: List[Dict] = []

        self.best_ckpt_path = os.path.join(
            self.output_dir, f"{self.task_name}_best_teacher.pth"
        )

    # -----------------------------------------------------------------------
    def train(self):
        """Full training loop. Saves best checkpoint. Returns best macro F1."""
        print("\n" + "=" * 70)
        print(f"  TRAINING: {self.task_name.upper()} TEACHER")
        print(f"  Classes  : {self.class_names}")
        print(f"  Epochs   : {self.tc.epochs}")
        print(f"  LR Backbone / Head: {self.tc.lr_backbone} / {self.tc.lr_head}")
        print("=" * 70)

        for epoch in range(1, self.tc.epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_loss, train_acc = self._train_epoch(epoch)

            # --- Validate ---
            val_loss, val_acc, val_macro_f1, val_per_class_f1 = self._eval_epoch(
                self.val_loader, split="val"
            )

            # --- Scheduler step ---
            self.scheduler.step()

            elapsed = time.time() - t0
            lr_backbone = self.optimizer.param_groups[0]["lr"]
            lr_head     = self.optimizer.param_groups[1]["lr"]

            print(
                f"  Epoch {epoch:03d}/{self.tc.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} MacroF1: {val_macro_f1:.4f} | "
                f"LR bb/hd: {lr_backbone:.2e}/{lr_head:.2e} | "
                f"{elapsed:.1f}s"
            )

            # --- Log ---
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_macro_f1": val_macro_f1,
                "val_per_class_f1": dict(zip(self.class_names, val_per_class_f1)),
                "lr_backbone": lr_backbone,
                "lr_head":     lr_head,
            })

            # --- Best model ---
            if val_macro_f1 > self.best_macro_f1 + self.tc.min_delta:
                self.best_macro_f1  = val_macro_f1
                self.best_epoch     = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_macro_f1)
                print(f"  ✓ New best model saved (MacroF1={val_macro_f1:.4f})")
            else:
                self.epochs_no_improve += 1

            # --- Early stopping ---
            if self.epochs_no_improve >= self.tc.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.tc.patience} epochs)")
                break

        # --- Final evaluation on test set ---
        print(f"\n[INFO] Loading best checkpoint (epoch {self.best_epoch})...")
        self._load_checkpoint()

        print("\n" + "=" * 70)
        print("  TEST SET EVALUATION")
        print("=" * 70)
        test_loss, test_acc, test_macro_f1, test_per_class_f1 = \
            self._eval_epoch(self.test_loader, split="test", verbose=True)

        # --- Quality gate ---
        gate_result = self._quality_gate(test_per_class_f1)

        # --- Save training log ---
        self._save_training_log(test_acc, test_macro_f1, test_per_class_f1, gate_result)

        return test_macro_f1, gate_result

    # -----------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)

            self.optimizer.step()

            # Stats
            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        acc      = total_correct / max(total_samples, 1)
        return avg_loss, acc

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        split: str = "val",
        verbose: bool = False,
    ) -> Tuple[float, float, float, List[float]]:
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            preds  = logits.argmax(dim=1)

            total_loss    += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss   = total_loss / max(total_samples, 1)
        acc        = accuracy_score(all_labels, all_preds)
        macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class  = f1_score(
            all_labels, all_preds,
            average=None, labels=list(range(self.num_classes)),
            zero_division=0,
        )

        if verbose:
            print(f"\n  Overall Accuracy : {acc:.4f}")
            print(f"  Macro F1         : {macro_f1:.4f}")
            print(f"\n  Per-class F1:")
            for cls, f1 in zip(self.class_names, per_class):
                flag = "  ← BELOW THRESHOLD" if f1 < self.tc.quality_gate_f1 else ""
                print(f"    {cls:<30} {f1:.4f}{flag}")

            print(f"\n  Classification Report:")
            # labels= forces sklearn to use all num_classes indices even if
            # some are absent from the test set (avoids ValueError crash)
            print(classification_report(
                all_labels, all_preds,
                labels=list(range(self.num_classes)),
                target_names=self.class_names,
                zero_division=0,
            ))

        return avg_loss, acc, macro_f1, per_class.tolist()

    # -----------------------------------------------------------------------
    def _quality_gate(self, per_class_f1: List[float]) -> Dict:
        """
        Checks each class against the minimum F1 threshold.
        Returns a dict marking each class as SAFE or UNSAFE for distillation.
        UNSAFE classes = teacher cannot be trusted to distil knowledge for them.
        """
        gate = {}
        failed = []
        for cls, f1 in zip(self.class_names, per_class_f1):
            if f1 >= self.tc.quality_gate_f1:
                gate[cls] = {"f1": f1, "status": "SAFE"}
            else:
                gate[cls] = {"f1": f1, "status": "UNSAFE"}
                failed.append((cls, f1))

        print("\n" + "=" * 70)
        print("  TEACHER QUALITY GATE")
        print("=" * 70)
        for cls, info in gate.items():
            marker = "✓" if info["status"] == "SAFE" else "✗"
            print(f"  {marker} {cls:<30} F1={info['f1']:.4f}  [{info['status']}]")

        if failed:
            print(f"\n  [WARNING] {len(failed)} class(es) FAILED the quality gate:")
            for cls, f1 in failed:
                print(f"    → {cls}: F1={f1:.4f} (threshold={self.tc.quality_gate_f1})")
            print(f"\n  These classes should NOT be used for distillation.")
            print(f"  Options: collect more data, or exclude them from the student task.")
        else:
            print(f"\n  ✓ ALL classes passed the quality gate (F1 ≥ {self.tc.quality_gate_f1}).")
            print(f"  This teacher is APPROVED for distillation.")
        print("=" * 70)

        return gate

    # -----------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_macro_f1: float):
        torch.save({
            "epoch":          epoch,
            "val_macro_f1":   val_macro_f1,
            "model_state":    self.model.state_dict(),
            "optimizer_state":self.optimizer.state_dict(),
            "class_names":    self.class_names,
            "task_name":      self.task_name,
        }, self.best_ckpt_path)

    def _load_checkpoint(self):
        ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded best checkpoint from epoch {ckpt['epoch']} "
              f"(Val MacroF1={ckpt['val_macro_f1']:.4f})")

    # -----------------------------------------------------------------------
    def _save_training_log(
        self,
        test_acc: float,
        test_macro_f1: float,
        test_per_class_f1: List[float],
        gate_result: Dict,
    ):
        log = {
            "task_name":          self.task_name,
            "class_names":        self.class_names,
            "best_epoch":         self.best_epoch,
            "best_val_macro_f1":  self.best_macro_f1,
            "test_accuracy":      test_acc,
            "test_macro_f1":      test_macro_f1,
            "test_per_class_f1":  dict(zip(self.class_names, test_per_class_f1)),
            "quality_gate":       gate_result,
            "data_meta":          self.data_meta,
            "history":            self.history,
        }
        log_path = os.path.join(self.output_dir, f"{self.task_name}_training_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n[INFO] Training log saved to: {log_path}")


# ===========================================================================
#  SOFT LABEL GENERATION
# ===========================================================================

def generate_soft_labels(
    model: PDDDTeacher,
    image_dir_map: Dict,        # { class_name -> [paths] } for ALL splits combined
    class_names: List[str],
    image_size: int,
    task_name: str,
    output_dir: str,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> str:
    """
    Runs the trained teacher on ALL images (train + val + test) and saves
    pre-softmax logits to a CSV file.

    CSV format (one row per image):
      image_path, class_name, true_label_idx, logit_0, logit_1, ..., logit_C

    These logits are consumed by the student's distillation loss via:
      softmax(logits / T)  where T is the distillation temperature.

    Returns the path to the saved CSV file.
    """
    from old.oldfruit1.teacher_data import get_val_transform, TeacherDataset

    model.eval()
    transform  = get_val_transform(image_size)
    dataset    = TeacherDataset(image_dir_map, class_names, transform=transform)
    loader     = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    csv_path   = os.path.join(output_dir, f"{task_name}_soft_labels.csv")
    header     = (
        ["image_path", "class_name", "true_label_idx"]
        + [f"logit_{i}" for i in range(len(class_names))]
    )

    print(f"\n[INFO] Generating soft labels for {len(dataset)} images...")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with torch.no_grad():
            for batch in loader:
                images  = batch["image"].to(device, non_blocking=True)
                labels  = batch["label"]
                paths   = batch["path"]

                logits  = model(images)                 # [B, C], raw pre-softmax
                logits_np = logits.cpu().numpy()

                for i in range(len(paths)):
                    label_idx  = labels[i].item()
                    class_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else "unknown"
                    row = [paths[i], class_name, label_idx] + logits_np[i].tolist()
                    writer.writerow(row)

    print(f"[INFO] Soft labels saved to: {csv_path}")
    print(f"       Rows: {len(dataset)} (one per image)")
    print(f"       Columns: image_path, class_name, true_label_idx, logit_0..{len(class_names)-1}")
    return csv_path


# Fix missing import
from typing import Dict