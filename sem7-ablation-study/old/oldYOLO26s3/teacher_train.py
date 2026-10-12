"""
teacher_train.py  (v2 — Two-Stage + Warmup + TTA + Ensemble Soft Labels)
=========================================================================
Changes from v1:
  train()       : Two-phase training (freeze backbone → full fine-tune + warmup)
  _train_epoch(): accepts optimizer as parameter (not self.optimizer)
  Phase 1       : Backbone frozen, head-only training at lr_head=1e-3 for freeze_epochs
  Phase 2       : Backbone unfrozen, LinearLR warmup → CosineAnnealingWarmRestarts
  Soft labels   : Now saved as probabilities (prob_0..C) not logits
  New function  : generate_ensemble_soft_labels() for multi-model + TTA
"""

import os
import csv
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingWarmRestarts, SequentialLR
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
)

from old.oldYOLO26s3.teacher_model import PDDDTeacher, FocalLoss, count_parameters
from old.oldYOLO26s3.teacher_data import build_dataloaders, get_val_transform, TeacherDataset


# ===========================================================================
#  TRAINING ENGINE
# ===========================================================================

class TeacherTrainer:

    def __init__(self, data_config, model_config, train_config,
                 class_names: List[str]):
        self.dc           = data_config
        self.mc           = model_config
        self.tc           = train_config
        self.class_names  = class_names
        self.num_classes  = len(class_names)
        self.task_name    = train_config.task_name
        self.output_dir   = train_config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Device ──────────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[INFO] Training on: {self.device}")
        if self.device.type == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[INFO] VRAM: {vram:.1f} GB")

        # ── Data ────────────────────────────────────────────────────────────
        minority_threshold = getattr(train_config, "minority_aug_threshold", 400)
        print(f"\n[INFO] Building dataloaders for task: {self.task_name}")
        self.train_loader, self.val_loader, self.test_loader, self.data_meta = \
            build_dataloaders(
                data_config            = data_config,
                class_names            = class_names,
                num_classes            = self.num_classes,
                task_name              = self.task_name,
                output_dir             = self.output_dir,
                minority_aug_threshold = minority_threshold,
            )

        # ── Model ───────────────────────────────────────────────────────────
        self.model = PDDDTeacher(
            num_classes       = self.num_classes,
            pddd_weights_path = model_config.pddd_weights_path,
            dropout_rate      = model_config.dropout_rate,
        ).to(self.device)

        pi = count_parameters(self.model)
        print(f"[INFO] Total params : {pi['total_M']}")
        print(f"[INFO] Trainable    : {pi['trainable_M']}")

        # # ── Loss (with label smoothing) ──────────────────────────────────────
        train_counts = self.data_meta.get("class_train_counts", {})
        if train_counts:
            counts = torch.tensor(
                [train_counts.get(n, 1) for n in self.class_names],
                dtype=torch.float32, device=self.device)
            inv = 1.0 / counts.clamp(min=1.0)
            class_weights = inv / inv.mean()
        else:
            class_weights = None

        ls = getattr(train_config, "label_smoothing", 0.1)
        self.criterion = FocalLoss(
            gamma=train_config.focal_gamma,
            class_weights=class_weights,
            label_smoothing=ls,
        )
        # ls = getattr(train_config, "label_smoothing", 0.1)
        # self.criterion = FocalLoss(
        #     gamma=train_config.focal_gamma,
        #     class_weights=None,   # WeightedRandomSampler already handles balance
        #     label_smoothing=ls,   # removing class_weights fixes Anthracnose over-prediction
        # )

        # ── State ───────────────────────────────────────────────────────────
        self.best_macro_f1     = 0.0
        self.best_epoch        = 0
        self.history: List[Dict] = []
        self.best_ckpt_path    = os.path.join(
            self.output_dir, f"{self.task_name}_best_teacher.pth")

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC: train()
    # ─────────────────────────────────────────────────────────────────────────
    def train(self) -> Tuple[float, Dict]:
        """
        Two-stage training:
          Phase 1 — backbone frozen, head-only, lr_head for freeze_epochs
          Phase 2 — backbone unfrozen, warmup + CosineAnnealingWarmRestarts
        Returns (test_macro_f1, quality_gate_dict).
        """
        print("\n" + "=" * 70)
        print(f"  TRAINING: {self.task_name.upper()} TEACHER  "
              f"(Two-Stage + Warmup + Label Smoothing)")
        print(f"  Classes      : {self.class_names}")
        print(f"  Phase 1      : {self.tc.freeze_epochs} epochs  "
              f"(backbone frozen, head lr={self.tc.lr_head})")
        print(f"  Phase 2      : {self.tc.epochs - self.tc.freeze_epochs} epochs  "
              f"(full fine-tune, warmup={self.tc.warmup_epochs})")
        print(f"  Label smooth : {getattr(self.tc, 'label_smoothing', 0.1)}")
        print("=" * 70)

        self._run_phase1()
        self._run_phase2()

        # ── Final test evaluation ────────────────────────────────────────────
        print(f"\n[INFO] Loading best checkpoint (epoch {self.best_epoch})...")
        self._load_checkpoint()

        print("\n" + "=" * 70)
        print("  TEST SET EVALUATION")
        print("=" * 70)
        _, test_acc, test_f1, test_pcf1 = self._eval_epoch(
            self.test_loader, split="test", verbose=True)

        gate = self._quality_gate(test_pcf1)
        self._save_training_log(test_acc, test_f1, test_pcf1, gate)
        return test_f1, gate

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 1 — backbone frozen
    # ─────────────────────────────────────────────────────────────────────────
    def _run_phase1(self):
        print(f"\n--- PHASE 1: Backbone FROZEN — training head only "
              f"({self.tc.freeze_epochs} epochs) ---")

        # Freeze backbone
        for name, param in self.model.named_parameters():
            param.requires_grad = ("head.fc" in name)

        head_params = [p for n, p in self.model.named_parameters()
                       if "head.fc" in n]
        opt_p1 = torch.optim.AdamW(
            head_params, lr=self.tc.lr_head,
            weight_decay=self.tc.weight_decay)

        for epoch in range(1, self.tc.freeze_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(epoch, opt_p1)
            # tr_loss, tr_acc = self._train_epoch(epoch, opt_p1, use_mixup=False)
            _, _, val_f1, _ = self._eval_epoch(self.val_loader, "val")
            elapsed = time.time() - t0

            print(f"  [P1] Epoch {epoch:02d}/{self.tc.freeze_epochs} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                  f"Val MacroF1: {val_f1:.4f} | {elapsed:.1f}s")

            self.history.append({"phase": 1, "epoch": epoch,
                                  "train_loss": tr_loss, "val_macro_f1": val_f1})

            if val_f1 > self.best_macro_f1 + self.tc.min_delta:
                self.best_macro_f1 = val_f1
                self.best_epoch    = epoch
                self._save_checkpoint(epoch, val_f1)
                print(f"  ✓ New best saved (MacroF1={val_f1:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    #  PHASE 2 — full fine-tune with warmup
    # ─────────────────────────────────────────────────────────────────────────
    def _run_phase2(self):
        remaining = self.tc.epochs - self.tc.freeze_epochs
        print(f"\n--- PHASE 2: Backbone UNFROZEN — warmup {self.tc.warmup_epochs} "
              f"epochs → CosineRestarts ({remaining} epochs total) ---")

        # Unfreeze all
        for param in self.model.parameters():
            param.requires_grad = True

        lr_head_p2 = getattr(self.tc, "lr_head_phase2", 1e-4)
        param_groups = self.model.get_parameter_groups(
            lr_backbone  = self.tc.lr_backbone,
            lr_head      = lr_head_p2,
            weight_decay = self.tc.weight_decay,
        )
        opt_p2 = torch.optim.AdamW(param_groups)

        warmup = LinearLR(opt_p2, start_factor=0.01, end_factor=1.0,
                          total_iters=self.tc.warmup_epochs)
        cosine = CosineAnnealingWarmRestarts(opt_p2, T_0=15, T_mult=2,
                                              eta_min=self.tc.eta_min)
        scheduler = SequentialLR(opt_p2, schedulers=[warmup, cosine],
                                  milestones=[self.tc.warmup_epochs])

        no_improve = 0
        start = self.tc.freeze_epochs + 1

        for epoch in range(start, self.tc.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(epoch, opt_p2)
            # tr_loss, tr_acc = self._train_epoch(epoch, opt_p2, use_mixup=True)
            _, _, val_f1, _ = self._eval_epoch(self.val_loader, "val")
            scheduler.step()
            elapsed = time.time() - t0

            lr_bb = opt_p2.param_groups[0]["lr"]
            lr_hd = opt_p2.param_groups[1]["lr"]
            p2ep  = epoch - self.tc.freeze_epochs

            print(f"  [P2-{p2ep:02d}] Epoch {epoch:03d}/{self.tc.epochs} | "
                  f"Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"LR bb/hd: {lr_bb:.2e}/{lr_hd:.2e} | {elapsed:.1f}s")

            self.history.append({"phase": 2, "epoch": epoch,
                                  "train_loss": tr_loss, "val_macro_f1": val_f1,
                                  "lr_backbone": lr_bb, "lr_head": lr_hd})

            if val_f1 > self.best_macro_f1 + self.tc.min_delta:
                self.best_macro_f1 = val_f1
                self.best_epoch    = epoch
                no_improve = 0
                self._save_checkpoint(epoch, val_f1)
                print(f"  ✓ New best saved (MacroF1={val_f1:.4f})")
            else:
                no_improve += 1

            if no_improve >= self.tc.patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {self.tc.patience} epochs)")
                break

    # ─────────────────────────────────────────────────────────────────────────
    #  INNER LOOPS
    # ─────────────────────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int,
                     optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.tc.max_grad_norm)
            optimizer.step()

            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        return (total_loss / max(total_samples, 1),
                total_correct / max(total_samples, 1))
    # def _train_epoch(self, epoch: int,
    #              optimizer: torch.optim.Optimizer,
    #              use_mixup: bool = False) -> Tuple[float, float]:
    #     """
    #     use_mixup=True in Phase 2 only.
    #     Mixup forces learning of Anthracnose/Black Rot boundary instead of
    #     maximising recall on one class at the expense of the other.
    #     """
    #     self.model.train()
    #     total_loss, total_correct, total_samples = 0.0, 0, 0
    #     MIXUP_ALPHA = 0.4   # Beta distribution parameter; 0.4 is optimal for 3-class problems

    #     for batch in self.train_loader:
    #         images = batch["image"].to(self.device, non_blocking=True)
    #         labels = batch["label"].to(self.device, non_blocking=True)

    #         if use_mixup:
    #             lam    = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
    #             idx    = torch.randperm(images.size(0), device=self.device)
    #             images = lam * images + (1.0 - lam) * images[idx]
    #             labels_b = labels[idx]

    #         optimizer.zero_grad()
    #         logits = self.model(images)

    #         if use_mixup:
    #             loss = (lam       * self.criterion(logits, labels) +
    #                     (1 - lam) * self.criterion(logits, labels_b))
    #         else:
    #             loss = self.criterion(logits, labels)

    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.max_grad_norm)
    #         optimizer.step()

    #         total_loss    += loss.item() * images.size(0)
    #         preds          = logits.argmax(dim=1)
    #         total_correct += (preds == labels).sum().item()
    #         total_samples += images.size(0)

    #     return (total_loss / max(total_samples, 1),
    #             total_correct / max(total_samples, 1))

    @torch.no_grad()
    def _eval_epoch(self, loader, split="val",
                    verbose=False) -> Tuple[float, float, float, List[float]]:
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

        avg_loss = total_loss / max(total_samples, 1)
        acc      = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro",
                             zero_division=0)
        per_cls  = f1_score(all_labels, all_preds, average=None,
                             labels=list(range(self.num_classes)),
                             zero_division=0)

        if verbose:
            print(f"\n  Overall Accuracy : {acc:.4f}")
            print(f"  Macro F1         : {macro_f1:.4f}")
            print(f"\n  Per-class F1:")
            for cls, f1 in zip(self.class_names, per_cls):
                flag = "  ← BELOW THRESHOLD" if f1 < self.tc.quality_gate_f1 else ""
                print(f"    {cls:<30} {f1:.4f}{flag}")
            print(f"\n  Classification Report:")
            print(classification_report(
                all_labels, all_preds,
                labels=list(range(self.num_classes)),
                target_names=self.class_names,
                zero_division=0))

        return avg_loss, acc, macro_f1, per_cls.tolist()

    # ─────────────────────────────────────────────────────────────────────────
    #  QUALITY GATE / CHECKPOINT / LOG
    # ─────────────────────────────────────────────────────────────────────────
    def _quality_gate(self, per_class_f1: List[float]) -> Dict:
        gate, failed = {}, []
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
            m = "✓" if info["status"] == "SAFE" else "✗"
            print(f"  {m} {cls:<30} F1={info['f1']:.4f}  [{info['status']}]")
        if failed:
            print(f"\n  [WARNING] {len(failed)} class(es) FAILED the quality gate.")
            for cls, f1 in failed:
                print(f"    → {cls}: F1={f1:.4f} (threshold={self.tc.quality_gate_f1})")
        else:
            print(f"\n  ✓ ALL classes passed the quality gate (F1 ≥ {self.tc.quality_gate_f1}).")
            print(f"  This teacher is APPROVED for distillation.")
        print("=" * 70)
        return gate

    def _save_checkpoint(self, epoch: int, val_macro_f1: float):
        torch.save({
            "epoch":        epoch,
            "val_macro_f1": val_macro_f1,
            "model_state":  self.model.state_dict(),
            "class_names":  self.class_names,
            "task_name":    self.task_name,
        }, self.best_ckpt_path)

    def _load_checkpoint(self):
        ckpt = torch.load(self.best_ckpt_path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(Val MacroF1={ckpt['val_macro_f1']:.4f})")

    def _save_training_log(self, test_acc, test_f1, test_pcf1, gate):
        log = {
            "task_name":         self.task_name,
            "class_names":       self.class_names,
            "best_epoch":        self.best_epoch,
            "best_val_macro_f1": self.best_macro_f1,
            "test_accuracy":     test_acc,
            "test_macro_f1":     test_f1,
            "test_per_class_f1": dict(zip(self.class_names, test_pcf1)),
            "quality_gate":      gate,
            "data_meta":         self.data_meta,
            "history":           self.history,
        }
        lp = os.path.join(self.output_dir, f"{self.task_name}_training_log.json")
        with open(lp, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n[INFO] Training log saved to: {lp}")


# ===========================================================================
#  ENSEMBLE + TTA SOFT LABEL GENERATION
# ===========================================================================

def generate_ensemble_soft_labels(
    models: List,
    image_dir_map: Dict,
    class_names: List[str],
    image_size: int,
    task_name: str,
    output_dir: str,
    device: torch.device,
    tta_transforms: List,
    report_every: int = 250,
) -> str:
    """
    For each image in image_dir_map:
      1. Apply every TTA transform to the image
      2. Run each trained model on every TTA view
      3. Average all (n_models × n_tta) softmax probability vectors
      4. Write averaged probabilities to CSV

    CSV columns: image_path, class_name, true_label_idx, prob_0 ... prob_C
    (probabilities, not logits — student should use directly, T=1)

    Returns path to saved CSV.
    """
    n_models = len(models)
    n_tta    = len(tta_transforms)

    # Move all models to device and set eval
    for m in models:
        m.eval()
        m.to(device)

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    all_samples: List[Tuple[str, str, int]] = []
    for cls, paths in image_dir_map.items():
        if cls in class_to_idx:
            for p in paths:
                all_samples.append((p, cls, class_to_idx[cls]))

    csv_path = os.path.join(output_dir, f"{task_name}_soft_labels.csv")
    header   = (["image_path", "class_name", "true_label_idx"] +
                 [f"prob_{i}" for i in range(len(class_names))])

    print(f"\n[INFO] Generating ensemble soft labels for {len(all_samples)} images...")
    print(f"[INFO] Models: {n_models}  |  TTA views: {n_tta}  "
          f"|  Total passes/image: {n_models * n_tta}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, (path, cls_name, label_idx) in enumerate(all_samples):
            try:
                img_np = np.array(Image.open(path).convert("RGB"))
            except Exception:
                continue

            all_probs = []
            with torch.no_grad():
                for model in models:
                    for transform in tta_transforms:
                        try:
                            aug   = transform(image=img_np)["image"]
                            aug   = aug.unsqueeze(0).to(device)
                            logit = model(aug)
                            prob  = torch.softmax(logit, dim=1).squeeze(0).cpu()
                            all_probs.append(prob)
                        except Exception:
                            continue

            if not all_probs:
                continue

            avg_prob = torch.stack(all_probs).mean(0).numpy()
            writer.writerow([path, cls_name, label_idx] + avg_prob.tolist())

            if (idx + 1) % report_every == 0:
                print(f"  Processed {idx+1}/{len(all_samples)} images...")

    print(f"[INFO] Soft labels saved to: {csv_path}")
    print(f"       Rows    : {len(all_samples)} (one per image)")
    print(f"       Columns : image_path, class_name, true_label_idx, "
          f"prob_0..{len(class_names)-1}")
    print(f"       Method  : {n_models} models × {n_tta} TTA = "
          f"{n_models*n_tta} passes/image")
    return csv_path
