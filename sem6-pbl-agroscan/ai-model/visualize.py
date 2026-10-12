import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.ops as ops

# Import your custom modules
from config import Config
from model import AgriculturalModel

# --- 1. CONFIGURATION ---
CHECKPOINT_PATH = "/content/drive/MyDrive/newmodel/checkpoints/best_model.pth" # Changed to best_model!
TEST_IMAGE_PATH = "/content/drive/MyDrive/newmodel/final_dataset/images/yield_images_AG_20170714_Diwa120_16S.jpg" 
CONFIDENCE_THRESHOLD = 0.3 # Ignore boxes with < 30% confidence
IOU_THRESHOLD = 0.45       # Overlap threshold for NMS

def load_model_for_inference(checkpoint_path, device):
    print("Loading configuration and model...")
    config = Config()
    model = AgriculturalModel(model_config=config.model)
    
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle if the checkpoint saved a dictionary or just the state_dict directly
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, config

def preprocess_image(image_path, image_size):
    # Load image for drawing (RGB, original size)
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocess image for the model (No spatial augs for inference!)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    tensor_img = transform(image=orig_rgb)["image"].unsqueeze(0) # Add batch dimension
    return orig_rgb, tensor_img

def yolo_to_pascal_voc(yolo_boxes, img_width, img_height):
    """
    Converts YOLO format [x_center, y_center, width, height] (normalized 0-1)
    to Pascal VOC format [x_min, y_min, x_max, y_max] (absolute pixels).
    """
    boxes_pixel = torch.zeros_like(yolo_boxes)
    
    # cx, cy, w, h
    cx = yolo_boxes[:, 0]
    cy = yolo_boxes[:, 1]
    w  = yolo_boxes[:, 2]
    h  = yolo_boxes[:, 3]
    
    # Convert to xmin, ymin, xmax, ymax in pixels
    boxes_pixel[:, 0] = (cx - w / 2) * img_width
    boxes_pixel[:, 1] = (cy - h / 2) * img_height
    boxes_pixel[:, 2] = (cx + w / 2) * img_width
    boxes_pixel[:, 3] = (cy + h / 2) * img_height
    
    return boxes_pixel

def visualize_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model, config = load_model_for_inference(CHECKPOINT_PATH, device)
    
    # Load Image
    orig_img, tensor_img = preprocess_image(TEST_IMAGE_PATH, config.data.image_size)
    tensor_img = tensor_img.to(device)
    img_h, img_w, _ = orig_img.shape
    
    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        preds = model(tensor_img) # Running purely in native float32
            
    # --- EXPERT FIX: Decode the new YOLO Decoupled Grid ---
    # raw_yield shape: [Batch, Channels(6), Height, Width]
    raw_yield = preds["raw_yield"][0].cpu() 
    channels, grid_h, grid_w = raw_yield.shape
    
    # Flatten the spatial grid into a list of predictions [H*W, 6]
    flat_preds = raw_yield.view(channels, -1).t()
    
    # In YOLODecoupledLoss, we apply sigmoid to get the actual normalized values
    boxes_normalized = torch.sigmoid(flat_preds[:, :4]) # cx, cy, w, h (0-1)
    obj_scores       = torch.sigmoid(flat_preds[:, 4])  # Objectness score
    cls_scores       = torch.sigmoid(flat_preds[:, 5])  # Apple class score
    
    # Final confidence is Objectness * Class Probability
    scores = obj_scores * cls_scores
    
    # Filter by confidence
    mask = scores > CONFIDENCE_THRESHOLD
    filtered_boxes  = boxes_normalized[mask]
    filtered_scores = scores[mask]
    
    print(f"Found {len(filtered_boxes)} candidate boxes above {CONFIDENCE_THRESHOLD*100}% confidence.")
    
    if len(filtered_boxes) == 0:
        print("No objects detected above threshold.")
        plt.imshow(orig_img)
        plt.axis("off")
        plt.show()
        return

    # Convert YOLO boxes to Pixel coordinates relative to original image size
    pixel_boxes = yolo_to_pascal_voc(filtered_boxes, img_w, img_h)
    
    # Apply Non-Maximum Suppression (NMS) to remove overlapping duplicates
    keep_indices = ops.nms(pixel_boxes, filtered_scores, IOU_THRESHOLD)
    final_boxes  = pixel_boxes[keep_indices]
    final_scores = filtered_scores[keep_indices]
    
    print(f"Final detected apples after NMS: {len(final_boxes)}")
    
    # Draw boxes on the image
    img_drawn = orig_img.copy()
    for box, score in zip(final_boxes, final_scores):
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        
        # Draw bounding box (Green)
        cv2.rectangle(img_drawn, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        # Draw confidence label
        label = f"{score:.2f}"
        cv2.putText(img_drawn, label, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # Display the result
    plt.figure(figsize=(12, 12))
    plt.imshow(img_drawn)
    plt.axis("off")
    plt.title(f"Yield Detection (Count: {len(final_boxes)})")
    
    # Save the image
    save_path = "/content/drive/MyDrive/newmodel/test_prediction.jpg"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Image successfully saved to: {save_path}")

if __name__ == "__main__":
    visualize_predictions()