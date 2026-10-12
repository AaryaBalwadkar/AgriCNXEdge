"""
train.py — Master Training Loop for Agri-CNX-Edge
==================================================
- Dynamic masked loss for multi-task, multi-label batches
- Homoscedastic uncertainty weighting
- PyTorch AMP, gradient clipping, cosine annealing LR
- Compression & recovery fine-tuning trigger
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig
from model import AgriculturalMultiTaskModel
from dataset import agri_multitask_dataloader
from compress_export import prune_heads_unstructured, recovery_finetune_hook

def masked_classification_loss(pred, target, criterion):
    # pred: (B, num_classes), target: (B,)
    mask = (target != -1) & (~torch.isnan(target))
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return criterion(pred[mask], target[mask].long())

def masked_detection_loss(pred, target, criterion):
    # pred: (B, A*(5+C), H, W), target: list of [N_obj, 5] or empty
    # Only compute for samples with at least one object
    batch_loss = []
    for p, t in zip(pred, target):
        if t is None or (isinstance(t, torch.Tensor) and t.numel() == 0):
            continue
        batch_loss.append(criterion(p.unsqueeze(0), t.unsqueeze(0)))
    if not batch_loss:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(batch_loss).mean()

def main():
    cfg = ProjectConfig()
    cfg.paths.make_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = AgriculturalMultiTaskModel(
        active_tasks=cfg.model.active_tasks,
        pretrained_backbone=cfg.model.pretrained_backbone,
        load_backbone_4bit=cfg.model.load_backbone_4bit,
    ).to(device)

    # --- Data ---
    train_loader = agri_multitask_dataloader(
        data_root=str(cfg.paths.data_root),
        split="train",
        image_size=cfg.model.image_size,
        batch_size=cfg.train.batch_size,
        augment=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = agri_multitask_dataloader(
        data_root=str(cfg.paths.data_root),
        split="val",
        image_size=cfg.model.image_size,
        batch_size=cfg.train.batch_size,
        augment=False,
        num_workers=cfg.train.num_workers,
    )

    # --- Losses ---
    # Map task name to loss function
    criterion_map = {}
    for task, params in cfg.model.active_tasks.items():
        if task == "yield_counting":
            # Placeholder: swap for YOLO loss as needed
            criterion_map[task] = nn.BCEWithLogitsLoss()
        else:
            criterion_map[task] = nn.CrossEntropyLoss()

    # --- Optimizer, Scheduler, AMP ---
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.max_epochs,
        eta_min=cfg.train.eta_min,
    )

    # --- Training Loop ---
    for epoch in range(cfg.train.max_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.max_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["labels"]  # [cls1, cls2, cls3, det4]
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                raw_losses = {}
                for task, pred in outputs.items():
                    idx = list(cfg.model.active_tasks.keys()).index(task)
                    target = labels[idx]
                    if task == "yield_counting":
                        # Detection: target is list of arrays
                        # Convert to torch tensors, move to device
                        det_targets = []
                        for t in target:
                            if t is None or (isinstance(t, float) and (t == -1 or t != t)):
                                det_targets.append(torch.zeros((0,5), device=device))
                            else:
                                arr = torch.tensor(t, dtype=torch.float32, device=device)
                                det_targets.append(arr)
                        loss = masked_detection_loss(pred, det_targets, criterion_map[task])
                    else:
                        # Classification: target is array
                        t = torch.tensor(target, device=device)
                        loss = masked_classification_loss(pred, t, criterion_map[task])
                    raw_losses[task] = loss

                # Uncertainty weighting
                loss, weights = model.loss_weighter(raw_losses)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", **{f"{k}_w": f"{v:.2f}" for k, v in weights.items()}})

        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")

        # --- Validation (optional, add as needed) ---

    # --- Compression & Recovery Fine-tuning ---
    print("==> Pruning heads and starting recovery fine-tuning ...")
    prune_heads_unstructured(model, amount=cfg.train.prune_amount, make_permanent=False)

    def recovery_loop(m):
        m.train()
        for epoch in range(cfg.train.recovery_finetune_epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Recovery Epoch {epoch+1}/{cfg.train.recovery_finetune_epochs}"):
                images = batch["image"].to(device)
                labels = batch["labels"]
                optimizer.zero_grad()
                with autocast():
                    outputs = m(images)
                    raw_losses = {}
                    for task, pred in outputs.items():
                        idx = list(cfg.model.active_tasks.keys()).index(task)
                        target = labels[idx]
                        if task == "yield_counting":
                            det_targets = []
                            for t in target:
                                if t is None or (isinstance(t, float) and (t == -1 or t != t)):
                                    det_targets.append(torch.zeros((0,5), device=device))
                                else:
                                    arr = torch.tensor(t, dtype=torch.float32, device=device)
                                    det_targets.append(arr)
                            loss = masked_detection_loss(pred, det_targets, criterion_map[task])
                        else:
                            t = torch.tensor(target, device=device)
                            loss = masked_classification_loss(pred, t, criterion_map[task])
                        raw_losses[task] = loss
                    loss, _ = m.loss_weighter(raw_losses)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(m.trainable_parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Recovery Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    recovery_finetune_hook(model, finetune_fn=recovery_loop, make_permanent_after=True)
    print("==> Training and compression complete.")

if __name__ == "__main__":
    main()