import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PHASE 1: DATA EXTRACTION
# ==========================================
def extract_data(output_dir):
    out_path = Path(output_dir)
    hist_file = out_path / "training_history.json"
    yield_file = out_path / "yield_eval.json"

    if not hist_file.exists():
        raise FileNotFoundError(f"Missing history file: {hist_file}")
   
    with open(hist_file, 'r') as f:
        history = json.load(f)

    yield_data = None
    if yield_file.exists():
        with open(yield_file, 'r') as f:
            yield_data = json.load(f)

    return history, yield_data

# ==========================================
# PHASE 2: THE CORE GRAPHS (Q1 GRADE)
# ==========================================
def generate_q1_plots(history, yield_data, output_dir):
    out_path = Path(output_dir)
   
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 600,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.axisbelow': True
    })

    epochs = [x["epoch"] for x in history]

    # 1. Detection Performance (Bar Chart)
    if yield_data:
        fig, ax = plt.subplots(figsize=(6, 5))
        metrics = ['Count MAE', 'Count RMSE']
        scores = [yield_data["count_mae"], yield_data["count_rmse"]]
        x = np.arange(len(metrics))
       
        rects = ax.bar(x, scores, 0.4, color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=0.8)
        ax.set_ylabel('Error Score')
        ax.set_title('Yield Detection Error')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
       
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
           
        plt.tight_layout()
        plt.savefig(out_path / 'q1_yield_metrics.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # 2. Loss Convergence Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    train_loss = [x.get("train_loss", 0) for x in history]
    val_loss = [x.get("val_loss_total", 0) for x in history]
   
    ax.plot(epochs, train_loss, label='Train Loss', color='#2ca02c', linewidth=2)
    ax.plot(epochs, val_loss, label='Val Loss', color='#d62728', linewidth=2, linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total Loss')
    ax.set_title('Model Convergence')
    ax.legend(frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig(out_path / 'q1_convergence_curve.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # 3. Confidence/Calibration (Task-Specific Validation Loss)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [x.get("val_loss_leaf", 0) for x in history], label='Leaf', color='#1f77b4', linewidth=1.5)
    ax.plot(epochs, [x.get("val_loss_pest", 0) for x in history], label='Pest', color='#ff7f0e', linewidth=1.5)
    ax.plot(epochs, [x.get("val_loss_fruit", 0) for x in history], label='Fruit', color='#2ca02c', linewidth=1.5)
   
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Task-Specific Calibration')
    ax.legend(frameon=True, edgecolor='black')
    plt.tight_layout()
    plt.savefig(out_path / 'q1_calibration_curve.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # 4. Task Balancing Dynamics (Homoscedastic Weights)
    w_leaf = [x.get("weights", {}).get("leaf", 0) for x in history]
    if any(w_leaf):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, w_leaf, label='Leaf', color='#1f77b4', linewidth=1.5)
        ax.plot(epochs, [x["weights"]["pest"] for x in history], label='Pest', color='#ff7f0e', linewidth=1.5)
        ax.plot(epochs, [x["weights"]["fruit"] for x in history], label='Fruit', color='#2ca02c', linewidth=1.5)
        ax.plot(epochs, [x["weights"]["yield"] for x in history], label='Yield', color='#9467bd', linewidth=1.5)
       
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Learned Precision Weight')
        ax.set_title('Homoscedastic Task Balancing')
        ax.legend(frameon=True, edgecolor='black')
        plt.tight_layout()
        plt.savefig(out_path / 'q1_homoscedastic_weights.pdf', format='pdf', bbox_inches='tight')
        plt.close()

# ==========================================
# PHASE 3: DEEP-DIVE ANALYSIS SUMMARY
# ==========================================
def print_analysis_summary(history, yield_data):
    best_epoch_data = min(history, key=lambda x: x.get("val_loss_total", float('inf')))
   
    print("\n" + "="*50)
    print(" PHASE 3: ABLATION ANALYSIS SUMMARY")
    print("="*50)
    print(f" Best Epoch        : {best_epoch_data['epoch']}")
    print(f" Best Val Loss     : {best_epoch_data['val_loss_total']:.4f}")
    print(f" Train Loss (Best) : {best_epoch_data['train_loss']:.4f}")
   
    print("\n Terminal Classification Accuracies:")
    print(f" Leaf Acc          : {best_epoch_data.get('val_acc_leaf', 0):.4f}")
    print(f" Pest Acc          : {best_epoch_data.get('val_acc_pest', 0):.4f}")
    print(f" Fruit Acc         : {best_epoch_data.get('val_acc_fruit', 0):.4f}")
   
    if yield_data:
        print("\n Detection Spatial Metrics (Yield):")
        print(f" Count MAE         : {yield_data['count_mae']:.4f}")
        print(f" Count RMSE        : {yield_data['count_rmse']:.4f}")
   
    if "weights" in best_epoch_data:
        print("\n Terminal Task Weights:")
        for k, v in best_epoch_data["weights"].items():
            print(f" {k.capitalize():<15} : {v:.4f}")
    print("="*50 + "\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q1-grade plots for a single model.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the model's output directory")
    args = parser.parse_args()

    print(f"[INFO] Processing directory: {args.output_dir}")
    hist, yld = extract_data(args.output_dir)
    generate_q1_plots(hist, yld, args.output_dir)
    print_analysis_summary(hist, yld)
    print("[INFO] PDF generation complete.")