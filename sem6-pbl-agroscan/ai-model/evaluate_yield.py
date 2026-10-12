import os
import cv2
cv2.setNumThreads(0)
import torch
import numpy as np
import torchvision.ops as ops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from config import Config
from model import AgriculturalModel
from dataset import AgriculturalDataset
from train import get_val_transform

# --- HYPERPARAMETERS ---
CONFIDENCE_THRESHOLD = 0.30
IOU_MATCH_THRESHOLD  = 0.50  # Overlap required to count as a "True Positive"

def decode_center_points(pred_dict, stride=4, conf_thresh=0.25, max_objects=500):
    """Converts CenterNet heatmaps to bounding boxes using MaxPool NMS."""
    heatmap = pred_dict["heatmap"] 
    wh      = pred_dict["wh"]      
    offset  = pred_dict["offset"]  
    
    batch_size, _, height, width = heatmap.shape
    heatmap = torch.sigmoid(heatmap)
    
    hmax = F.max_pool2d(heatmap, kernel_size=3, padding=1, stride=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep 
    
    heatmap = heatmap.view(batch_size, -1)
    scores, indices = torch.topk(heatmap, k=max_objects, dim=1)
    valid_mask = scores > conf_thresh
    
    batch_boxes = []
    for b in range(batch_size):
        b_scores  = scores[b][valid_mask[b]]
        b_indices = indices[b][valid_mask[b]]
        
        if len(b_scores) == 0:
            batch_boxes.append(torch.empty((0, 5), device=heatmap.device))
            continue
            
        ys = (b_indices // width).float()
        xs = (b_indices % width).float()
        
        b_wh     = wh[b].view(2, -1)[:, b_indices]
        b_offset = offset[b].view(2, -1)[:, b_indices]
        
        final_xs = xs + b_offset[0, :]
        final_ys = ys + b_offset[1, :]
        
        box_cx = final_xs * stride
        box_cy = final_ys * stride
        box_w  = b_wh[0, :] * stride
        box_h  = b_wh[1, :] * stride
        
        boxes = torch.stack([box_cx, box_cy, box_w, box_h, b_scores], dim=1)
        batch_boxes.append(boxes)
        
    return batch_boxes

def cxcywh_to_xyxy(boxes, img_size=None):
    """Converts [cx, cy, w, h] to [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return torch.empty((0, 4), device=boxes.device if isinstance(boxes, torch.Tensor) else 'cpu')
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    out_boxes = torch.stack((x1, y1, x2, y2), dim=1)
    if img_size is not None:
        out_boxes = out_boxes * img_size
    return out_boxes

def denormalize_image(tensor):
    """Reverses ImageNet normalization for plotting."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def plot_4_images(vis_data, save_path):
    """Plots a 2x2 grid of images with Actual and Predicted bounding boxes."""
    if not vis_data:
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        if i < len(vis_data):
            img_tensor, gt_boxes, pred_boxes = vis_data[i]
            img = denormalize_image(img_tensor)
            ax.imshow(img)
            
            # Draw Ground Truth (Solid Lime Green)
            for box in gt_boxes:
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
            
            # Draw Predictions (Dashed Red)
            for box in pred_boxes:
                x1, y1, x2, y2 = box[:4]
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect)
            
            # Invisible patches just for the legend
            lime_patch = patches.Patch(color='lime', label='Actual Ground Truth')
            red_patch = patches.Patch(color='red', label='Model Prediction', linestyle='--')
            ax.legend(handles=[lime_patch, red_patch], loc='upper right', framealpha=0.8)
            
            ax.set_title(f"Test Image {i+1} | Actual Apples: {len(gt_boxes)} | Predicted: {len(pred_boxes)}", fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved bounding box visualization to: {save_path}")

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Yield on: {device}")

    # 1. Load Model
    model = AgriculturalModel(model_config=config.model).to(device)
    checkpoint_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load Dataset (We use the test set)
    test_dataset = AgriculturalDataset(
        config=config,
        split_file_path=os.path.join(config.data.splits_dir, "test.txt"),
        transforms=get_val_transform(config.data.image_size),
    )

    total_images_with_apples = 0
    total_apples_actual = 0
    total_apples_predicted = 0
    total_error = 0
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Store up to 4 images that actually contain apples for visualization
    vis_data = []

    print(f"Starting inference on {len(test_dataset)} test images...")

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Testing Center-Point Maps"):
            image_id = test_dataset.image_ids[idx]
            gt_boxes_raw = test_dataset.gt_yield_boxes_dict.get(image_id, [])
            gt_count = len(gt_boxes_raw)
            
            if gt_count == 0:
                continue
                
            data = test_dataset[idx]
            image = data["image"].unsqueeze(0).to(device)
            
            preds = model(image)
            yield_map = preds["yield_map"]
            
            pred_boxes = decode_center_points(
                pred_dict=yield_map, 
                stride=4, 
                conf_thresh=CONFIDENCE_THRESHOLD, 
                max_objects=500
            )[0]
            
            pred_count = len(pred_boxes)
            
            total_images_with_apples += 1
            total_apples_actual += gt_count
            total_apples_predicted += pred_count
            total_error += abs(pred_count - gt_count)
            
            if pred_count > 0:
                pred_xyxy = cxcywh_to_xyxy(pred_boxes[:, :4]).to(device)
            else:
                pred_xyxy = torch.empty((0, 4), device=device)
                
            gt_xyxy = cxcywh_to_xyxy(gt_boxes_raw, img_size=config.data.image_size).to(device)
            
            # Save data for the visualization grid
            if len(vis_data) < 4:
                vis_data.append((data["image"], gt_xyxy.cpu().numpy(), pred_xyxy.cpu().numpy()))
            
            if pred_count == 0:
                total_fn += gt_count
                continue
                
            iou_matrix = ops.box_iou(pred_xyxy, gt_xyxy)
            matches = iou_matrix >= IOU_MATCH_THRESHOLD
            
            gt_matched = torch.zeros(gt_count, dtype=torch.bool, device=device)
            pred_matched = torch.zeros(pred_count, dtype=torch.bool, device=device)
            
            for p_idx in range(pred_count):
                best_gt_idx = -1
                best_iou = -1.0
                for g_idx in range(gt_count):
                    if matches[p_idx, g_idx] and not gt_matched[g_idx]:
                        if iou_matrix[p_idx, g_idx] > best_iou:
                            best_iou = iou_matrix[p_idx, g_idx]
                            best_gt_idx = g_idx
                if best_gt_idx != -1:
                    gt_matched[best_gt_idx] = True
                    pred_matched[p_idx] = True
                    total_tp += 1
                    
            total_fp += (~pred_matched).sum().item()
            total_fn += (~gt_matched).sum().item()

    # --- METRICS CALCULATION ---
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if total_images_with_apples > 0:
        mae = total_error / total_images_with_apples
        
        print(f"\n{'='*60}")
        print(" 📊 CENTERNET YIELD DETECTION METRICS")
        print(f"{'='*60}")
        print(f"Total Test Images Analysed : {total_images_with_apples}")
        print(f"Total Apples (Actual)      : {total_apples_actual}")
        print(f"Total Apples (Predicted)   : {total_apples_predicted}")
        print("-" * 60)
        print(f"Mean Absolute Error (MAE)  : {mae:.2f} apples miscounted per image")
        print("-" * 60)
        print(f"Precision (Exact Match)    : {precision:.4f} (When it guesses an apple, is it right?)")
        print(f"Recall (Exact Match)       : {recall:.4f} (Out of all real apples, how many did it find?)")
        print(f"F1-Score                   : {f1_score:.4f} (Harmonic mean of Precision and Recall)")
        print(f"{'='*60}")
        
        # Generate the visualization image
        vis_save_path = os.path.join(config.train.checkpoints_dir, "yield_visualization.png")
        plot_4_images(vis_data, vis_save_path)
    else:
        print("No validation images containing apples were found.")

if __name__ == "__main__":
    main()