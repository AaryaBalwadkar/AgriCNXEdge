# import torch
# from student_config import get_config

# cfg = get_config()
# weights_path = cfg.paths.dinov3_weights

# print(f"Loading checkpoint: {weights_path}")
# ckpt = torch.load(weights_path, map_location="cpu")

# # Handle nested state_dict wrappers
# if isinstance(ckpt, dict):
#     if "model" in ckpt:
#         ckpt = ckpt["model"]
#     elif "state_dict" in ckpt:
#         ckpt = ckpt["state_dict"]

# print("\n--- FIRST 15 KEYS IN YOUR DINOv3 WEIGHT FILE ---")
# for i, key in enumerate(list(ckpt.keys())[:15]):
#     shape = list(ckpt[key].shape) if hasattr(ckpt[key], "shape") else "N/A"
#     print(f"{i+1:2d}. {key:<50} shape: {shape}")


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_test_set(model, test_loader, tasks, device):
    """
    Evaluates a PyTorch model on a holdout test set across multiple tasks.
    
    Args:
        model (torch.nn.Module): Loaded PyTorch model.
        test_loader (DataLoader): DataLoader for the holdout test set.
        tasks (list): List of task names (e.g., ['task_a', 'task_b']).
        device (torch.device): Device to run evaluation on ('cuda' or 'cpu').
    """
    model.eval()
    
    # Store predictions and ground truths for each task
    all_targets = {task: [] for task in tasks}
    all_preds = {task: [] for task in tasks}
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Expecting dict: {'task_a': logits, 'task_b': logits}
            
            for task in tasks:
                task_logits = outputs[task]
                preds = torch.argmax(task_logits, dim=1)
                
                all_preds[task].extend(preds.cpu().numpy())
                all_targets[task].extend(targets[task].numpy())

    # Calculate metrics and plot confusion matrix per task
    for task in tasks:
        print(f"\n" + "="*50)
        print(f" EVALUATION REPORT FOR TASK: {task.upper()}")
        print("="*50)
        
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_preds[task])
        
        # Precision, Recall, F1-Score per class + overall averages
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Visualizing Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {task}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

# ==========================================
# Script Execution & Loading best_model.pth
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize model architecture (replace with your actual model class)
    # model = MyMultiTaskModel().to(device)
    
    # 2. Load best saved weights
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # 3. Define task list matching your model dictionary outputs
    task_names = ["task_1", "task_2"] 
    
    # 4. Run evaluation (replace test_loader with your actual test DataLoader)
    # evaluate_test_set(model, test_loader, task_names, device)