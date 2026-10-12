import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config
from model import AgriculturalModel
from dataset import AgriculturalDataset
from losses import MultiTaskDistillationLoss

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# BBOX_PARAMS is required because dataset.py passes gt_boxes through the transforms
BBOX_PARAMS   = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1)

# ---------------------------------------------------------------
# The Multi-Task Super Transform
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# The Multi-Task Super Transform
# ---------------------------------------------------------------

def get_multitask_transform(image_size):
    """
    Because dataset.py now transforms Teacher boxes with the image, 
    we can safely use powerful spatial augmentations for all 4 tasks!
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(p=0.3),
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=0.4), 
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5), 
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    # EXPERT FIX: Added min_area=4.0 to instantly drop 0-pixel boxes before CoarseDropout
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.1, min_area=4.0)) # type: ignore


def get_val_transform(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_area=4.0)) # type: ignore


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ---------------------------------------------------------------
# Train One Epoch
# ---------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, accum_steps, epoch, config, best_val_loss):
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(loader):
        # 1. Move image to device
        images = batch["image"].to(device)
        
        # 2. Move Nested GT Dictionaries to device safely
        labels = {
            "gt_leaf":  batch["gt_leaf"].to(device),
            "gt_fruit": batch["gt_fruit"].to(device),
            "gt_pest":  batch["gt_pest"].to(device),
            "yield_gt": {k: v.to(device) for k, v in batch["yield_gt"].items()}
        }
        
        # 3. Move Nested Teacher Dictionaries to device safely
        t_preds = {
            "t_leaf":  batch["t_leaf"].to(device),
            "t_fruit": batch["t_fruit"].to(device),
            "t_pest":  batch["t_pest"].to(device),
            "t_yield_maps": {k: v.to(device) for k, v in batch["t_yield_maps"].items()}
        }

        # Forward pass
        with torch.amp.autocast('cuda', enabled=config.train.use_amp):
            preds = model(images)
            loss_dict = criterion(preds, t_preds, labels)
            loss = loss_dict["total_loss"] / accum_steps

        # Backward pass
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Track running loss
        batch_loss = loss_dict["total_loss"].item()
        running_loss += batch_loss
        
        # ---------------------------------------------------------
        # THE MISSING LOGGING BLOCK
        # Prints an update to the terminal every 50 batches
        # ---------------------------------------------------------
        if (i + 1) % 50 == 0:
            print(f"  Epoch [{epoch}] Batch [{i + 1}/{len(loader)}]  "
                  f"Loss: {batch_loss:.4f}  "
                  f"Yield Loss: {loss_dict.get('yield_loss', 0.0):.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
    return running_loss / len(loader)


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            
            labels = {
                "gt_leaf":  batch["gt_leaf"].to(device),
                "gt_fruit": batch["gt_fruit"].to(device),
                "gt_pest":  batch["gt_pest"].to(device),
                "yield_gt": {k: v.to(device) for k, v in batch["yield_gt"].items()}
            }
            t_preds = {
                "t_leaf":  batch["t_leaf"].to(device),
                "t_fruit": batch["t_fruit"].to(device),
                "t_pest":  batch["t_pest"].to(device),
                "t_yield_maps": {k: v.to(device) for k, v in batch["t_yield_maps"].items()}
            }

            preds = model(images)
            loss_dict = criterion(preds, t_preds, labels)
            running_loss += loss_dict["total_loss"].item()
            
    return running_loss / len(loader)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    config      = Config()
    device      = torch.device(config.train.device)
    accum_steps = config.train.gradient_accumulation_steps
    image_size  = config.data.image_size

    val_transform = get_val_transform(image_size)

    train_dataset = AgriculturalDataset(
        config=config,
        split_file_path=os.path.join(config.data.splits_dir, "train.txt"),
        transforms=get_multitask_transform(image_size),
    )
    val_dataset = AgriculturalDataset(
        config=config,
        split_file_path=os.path.join(config.data.splits_dir, "val.txt"),
        transforms=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    model     = AgriculturalModel(model_config=config.model).to(device)
    criterion = MultiTaskDistillationLoss(train_config=config.train).to(device)

    # 1. Isolate the Center-Point Head parameters (Yield)
    yield_params = [p for p in model.heads.yield_head.parameters() if p.requires_grad]
    yield_param_ids = {id(p) for p in yield_params}
    
    # 2. Grab everything else (Classification heads + Loss functions)
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in yield_param_ids] + \
                   [p for p in criterion.parameters() if p.requires_grad]

    # 3. Create parameter groups with decoupled velocities
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": 1e-4}, # Slow step for global classification
        {"params": yield_params, "lr": 1e-3}  # Fast step for high-res heatmap regression 
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=1e-6,
    )

    start_epoch   = 0
    best_val_loss = float("inf")

    if config.train.resume_from_checkpoint and os.path.isfile(config.train.resume_from_checkpoint):
        ckpt          = torch.load(config.train.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}  (best_val_loss={best_val_loss:.4f})")

    print(f"\nTraining on {device}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Effective batch: {config.train.batch_size * accum_steps}")
    print(f"Image size    : {image_size}px\n")

    for epoch in range(start_epoch, config.train.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, device, accum_steps, epoch + 1,
            config, best_val_loss,
        )

        val_loss = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1}/{config.train.epochs} "
              f"Train={train_loss:.4f}  Val={val_loss:.4f}  "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
            save_checkpoint({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "best_val_loss": best_val_loss,
            }, best_path)
            print(f"  * New best model saved  (val_loss={best_val_loss:.4f})\n")


if __name__ == "__main__":
    main()