# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import DataLoader

# from config import Config, LEAF_CLASS_MAP, FRUIT_CLASS_MAP, PEST_CLASS_MAP
# from model import AgriculturalModel
# from dataset import AgriculturalDataset
# from train import get_val_transform # Reusing your validation transform

# def plot_confusion_matrix(y_true, y_pred, class_map, task_name, output_dir):
#     """Generates and saves a styled confusion matrix."""
#     if len(y_true) == 0:
#         print(f"No ground truth data found for {task_name}. Skipping plot.")
#         return

#     # Invert the dictionary to get Index -> Class Name
#     idx_to_class = {v: k for k, v in class_map.items()}
#     labels = [idx_to_class[i] for i in range(len(class_map))]

#     cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_map))))
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#     plt.title(f'{task_name} Confusion Matrix')
#     plt.ylabel('Actual / Ground Truth')
#     plt.xlabel('Predicted')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     save_path = os.path.join(output_dir, f"{task_name.lower()}_confusion_matrix.png")
#     plt.savefig(save_path, dpi=150)
#     plt.close()
#     print(f"Saved {task_name} matrix to: {save_path}")


# def main():
#     config = Config()
    
#     # Auto-detect your awesome RTX A4000 or fallback to CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Evaluating on: {device}")

#     # 1. Load the Model
#     model = AgriculturalModel(model_config=config.model)
#     checkpoint_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
#     print(f"Loading weights from {checkpoint_path}...")
#     checkpoint = torch.load(checkpoint_path, map_location=device)
    
#     if "model_state" in checkpoint:
#         model.load_state_dict(checkpoint["model_state"])
#     else:
#         model.load_state_dict(checkpoint)
        
#     model.to(device)
#     model.eval()

#     # 2. Load the Test Data
#     test_dataset = AgriculturalDataset(
#         config=config,
#         split_file_path=os.path.join(config.data.splits_dir, "test.txt"),
#         transforms=get_val_transform(config.data.image_size),
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config.train.batch_size,
#         shuffle=False,
#         num_workers=config.train.num_workers,
#         pin_memory=True if torch.cuda.is_available() else False
#     )

#     # 3. Tracking variables
#     all_true_leaf, all_pred_leaf = [], []
#     all_true_fruit, all_pred_fruit = [], []
#     all_true_pest, all_pred_pest = [], []

#     print(f"Starting inference on {len(test_dataset)} test images...")

#     with torch.no_grad():
#         for batch in test_loader:
#             images = batch["image"].to(device)
#             gt_leaf = batch["gt_leaf"].numpy()
#             gt_fruit = batch["gt_fruit"].numpy()
#             gt_pest = batch["gt_pest"].numpy()

#             # --- EXPERT FIX: Removed AMP (Autocast) to run natively in pure FP32 ---
#             preds = model(images)

#             # Convert raw logits to class predictions (argmax)
#             pred_leaf = torch.argmax(preds["leaf"], dim=1).cpu().numpy()
#             pred_fruit = torch.argmax(preds["fruit"], dim=1).cpu().numpy()
#             pred_pest = torch.argmax(preds["pest"], dim=1).cpu().numpy()

#             # 4. Filter out the missing labels (ignore_index = -1)
#             for i in range(len(images)):
#                 if gt_leaf[i] != -1:
#                     all_true_leaf.append(gt_leaf[i])
#                     all_pred_leaf.append(pred_leaf[i])
                
#                 if gt_fruit[i] != -1:
#                     all_true_fruit.append(gt_fruit[i])
#                     all_pred_fruit.append(pred_fruit[i])
                    
#                 if gt_pest[i] != -1:
#                     all_true_pest.append(gt_pest[i])
#                     all_pred_pest.append(pred_pest[i])

#     # 5. Plot and save
#     print("\nGenerating Confusion Matrices...")
#     plot_confusion_matrix(all_true_leaf, all_pred_leaf, LEAF_CLASS_MAP, "Leaf", config.train.checkpoints_dir)
#     plot_confusion_matrix(all_true_fruit, all_pred_fruit, FRUIT_CLASS_MAP, "Fruit", config.train.checkpoints_dir)
#     plot_confusion_matrix(all_true_pest, all_pred_pest, PEST_CLASS_MAP, "Pest", config.train.checkpoints_dir)

#     print("Done!")

# if __name__ == "__main__":
#     main()


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader

from config import Config, LEAF_CLASS_MAP, FRUIT_CLASS_MAP, PEST_CLASS_MAP
from model import AgriculturalModel
from dataset import AgriculturalDataset
from train import get_val_transform 

def plot_confusion_matrix(y_true, y_pred, class_map, task_name, output_dir):
    """Generates and saves a styled confusion matrix."""
    if len(y_true) == 0:
        print(f"No ground truth data found for {task_name}. Skipping plot.")
        return

    # Invert the dictionary to get Index -> Class Name
    idx_to_class = {v: k for k, v in class_map.items()}
    labels = [idx_to_class[i] for i in range(len(class_map))]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_map))))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{task_name} Confusion Matrix')
    plt.ylabel('Actual / Ground Truth')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{task_name.lower()}_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {task_name} matrix to: {save_path}")

def print_metrics(y_true, y_pred, class_map, task_name):
    """Prints Accuracy, Precision, Recall, and F1-Score to the terminal."""
    if len(y_true) == 0:
        return

    idx_to_class = {v: k for k, v in class_map.items()}
    target_names = [idx_to_class[i] for i in range(len(class_map))]
    
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f" 📊 {task_name.upper()} METRICS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    
    # zero_division=0 prevents warnings if a class has 0 predictions
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        labels=list(range(len(class_map))),
        zero_division=0
    )
    print(report)
    print("="*60)

def main():
    config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load the Model
    model = AgriculturalModel(model_config=config.model)
    checkpoint_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # 2. Load the Test Data
    test_dataset = AgriculturalDataset(
        config=config,
        split_file_path=os.path.join(config.data.splits_dir, "test.txt"),
        transforms=get_val_transform(config.data.image_size),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 3. Tracking variables
    all_true_leaf, all_pred_leaf = [], []
    all_true_fruit, all_pred_fruit = [], []
    all_true_pest, all_pred_pest = [], []

    print(f"Starting inference on {len(test_dataset)} test images...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_leaf = batch["gt_leaf"].numpy()
            gt_fruit = batch["gt_fruit"].numpy()
            gt_pest = batch["gt_pest"].numpy()

            preds = model(images)

            # Convert raw logits to class predictions (argmax)
            pred_leaf = torch.argmax(preds["leaf"], dim=1).cpu().numpy()
            pred_fruit = torch.argmax(preds["fruit"], dim=1).cpu().numpy()
            pred_pest = torch.argmax(preds["pest"], dim=1).cpu().numpy()

            # 4. Filter out the missing labels (ignore_index = -1)
            for i in range(len(images)):
                if gt_leaf[i] != -1:
                    all_true_leaf.append(gt_leaf[i])
                    all_pred_leaf.append(pred_leaf[i])
                
                if gt_fruit[i] != -1:
                    all_true_fruit.append(gt_fruit[i])
                    all_pred_fruit.append(pred_fruit[i])
                    
                if gt_pest[i] != -1:
                    all_true_pest.append(gt_pest[i])
                    all_pred_pest.append(pred_pest[i])

    # 5. Calculate and print numerical metrics
    print_metrics(all_true_leaf, all_pred_leaf, LEAF_CLASS_MAP, "Leaf")
    print_metrics(all_true_fruit, all_pred_fruit, FRUIT_CLASS_MAP, "Fruit")
    print_metrics(all_true_pest, all_pred_pest, PEST_CLASS_MAP, "Pest")

    # 6. Plot and save matrices
    print("\nGenerating Confusion Matrices...")
    plot_confusion_matrix(all_true_leaf, all_pred_leaf, LEAF_CLASS_MAP, "Leaf", config.train.checkpoints_dir)
    plot_confusion_matrix(all_true_fruit, all_pred_fruit, FRUIT_CLASS_MAP, "Fruit", config.train.checkpoints_dir)
    plot_confusion_matrix(all_true_pest, all_pred_pest, PEST_CLASS_MAP, "Pest", config.train.checkpoints_dir)

    print("Done!")

if __name__ == "__main__":
    main()