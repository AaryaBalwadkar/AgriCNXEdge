import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torchvision.transforms.functional as TF
from matplotlib.patches import Rectangle

from student_config import get_config, IDX_TO_LEAF, IDX_TO_PEST, IDX_TO_FRUIT
from student_model import ADCStudent, decode_detections
from student_dataset import build_dataloaders

# ==========================================
# 1. Q1 PLOT STYLING
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 600,
})

def denormalize_image(tensor):
    """Reverts ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# ==========================================
# 2. EVALUATION PIPELINE
# ==========================================
def evaluate_and_visualize(model_path, output_dir):
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    print(f"\n[INFO] Loading Model from: {model_path}")
    model = ADCStudent(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
   
    print("[INFO] Building DataLoader...")
    _, val_loader = build_dataloaders(cfg)
   
    # Storage for metrics
    y_true_leaf, y_pred_leaf = [], []
    y_true_pest, y_pred_pest = [], []
    y_true_fruit, y_pred_fruit = [], []
   
    visualized_images = 0
    max_visualizations = 5 # Number of bounding box images to save
    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Running Inference on Validation Set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            roi_boxes = batch["roi_boxes"].to(device)
            roi_img_idx = batch["roi_img_idx"].to(device)
           
            out = model(images, gt_boxes=roi_boxes, gt_box_img_idx=roi_img_idx)
           
            # --- 1. COLLECT CLASSIFICATION PREDICTIONS ---
            # Leaf
            mask_leaf = batch["leaf_label"] != -1
            if mask_leaf.any():
                y_true_leaf.extend(batch["leaf_label"][mask_leaf].cpu().numpy())
                y_pred_leaf.extend(out["leaf_logits"][mask_leaf].argmax(dim=1).cpu().numpy())
               
            # Pest
            mask_pest = batch["pest_label"] != -1
            if mask_pest.any():
                y_true_pest.extend(batch["pest_label"][mask_pest].cpu().numpy())
                y_pred_pest.extend(out["pest_logits"][mask_pest].argmax(dim=1).cpu().numpy())
               
            # Fruit
            # Fruit targets are tied to regions. Decode ROI indices.
            if out["fruit_logits"].shape[0] > 0:
                ri = out["box_img_idx"].cpu()
                fh = batch["fruit_label"][ri]
                mask_fruit = fh != -1
                if mask_fruit.any():
                    y_true_fruit.extend(fh[mask_fruit].cpu().numpy())
                    y_pred_fruit.extend(out["fruit_logits"][mask_fruit].argmax(dim=1).cpu().numpy())

            # --- 2. BOUNDING BOX VISUALIZATION ---
            if visualized_images < max_visualizations:
                # Find an image in this batch that has ground truth boxes
                for i in range(len(images)):
                    if batch["gt_mask"][i].any() and visualized_images < max_visualizations:
                        img_vis = denormalize_image(images[i])
                       
                        # Get GT boxes
                        valid_gt = batch["gt_mask"][i]
                        gt_boxes = batch["gt_boxes"][i][valid_gt].cpu().numpy()
                       
                        # Get Pred boxes
                        raw_det = out["raw_det"][i:i+1] # Keep batch dim
                        pred_boxes, scores, _ = decode_detections(
                            raw_det, conf_thresh=cfg.model.conf_threshold, nms_iou=cfg.model.nms_iou)
                        pred_boxes = pred_boxes.cpu().numpy()
                       
                        # Plot
                        fig, ax = plt.subplots(1, figsize=(8, 8))
                        ax.imshow(img_vis)
                       
                        # Draw GT (Green)
                        H, W = img_vis.shape[:2]
                        for box in gt_boxes:
                            cx, cy, w, h = box
                            rect = Rectangle(((cx - w/2)*W, (cy - h/2)*H), w*W, h*H,
                                             linewidth=2, edgecolor='lime', facecolor='none', label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else "")
                            ax.add_patch(rect)
                           
                        # Draw Pred (Red)
                        for box in pred_boxes:
                            cx, cy, w, h = box
                            rect = Rectangle(((cx - w/2)*W, (cy - h/2)*H), w*W, h*H,
                                             linewidth=2, edgecolor='red', linestyle='--', facecolor='none', label='Pred' if 'Pred' not in ax.get_legend_handles_labels()[1] else "")
                            ax.add_patch(rect)
                           
                        ax.axis('off')
                        ax.legend()
                        plt.title("Yield Detection: Ground Truth vs Prediction")
                        plt.savefig(os.path.join(output_dir, f'bbox_vis_{visualized_images}.png'), bbox_inches='tight')
                        plt.close()
                        visualized_images += 1

    # ==========================================
    # 3. PRINT F1 SCORES & SAVE CONFUSION MATRICES
    # ==========================================
    tasks = [
        ("Leaf", y_true_leaf, y_pred_leaf, IDX_TO_LEAF),
        ("Pest", y_true_pest, y_pred_pest, IDX_TO_PEST),
        ("Fruit", y_true_fruit, y_pred_fruit, IDX_TO_FRUIT)
    ]
   
    print("\n" + "="*60)
    print("  Q1 CLASSIFICATION METRICS (MACRO F1)")
    print("="*60)
   
    for task_name, y_true, y_pred, idx_map in tasks:
        if len(y_true) == 0:
            print(f"\n[WARN] No samples found for {task_name} task.")
            continue
           
        labels = list(range(len(idx_map)))
        target_names = [idx_map[i] for i in labels]
       
        # Print Classification Report (Added labels and zero_division)
        print(f"\n--- {task_name.upper()} TASK ---")
        print(classification_report(
            y_true, y_pred, 
            labels=labels, 
            target_names=target_names, 
            digits=4, 
            zero_division=0
        ))
       
        # Generate Confusion Matrix (Added labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
       
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
        plt.title(f'{task_name} Task Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cm_{task_name.lower()}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
       
    print(f"\n[INFO] Confusion matrices and Bounding Box visuals saved to: {output_dir}")

if __name__ == "__main__":
    # Update this path to evaluate whichever model you want!
    MODEL_CHECKPOINT = "outputs/student/checkpoints/best_model.pth"
    OUTPUT_FOLDER = "outputs/student/visual_eval/"
   
    evaluate_and_visualize(MODEL_CHECKPOINT, OUTPUT_FOLDER)