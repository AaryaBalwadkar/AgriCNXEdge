# Hierarchical Multi-Task Learning with Foundation Vision Backbone and Multi-Teacher Distillation for Edge-Based Agricultural Monitoring

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9AB00.svg?style=for-the-badge&logo=googlecolab&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005C84.svg?style=for-the-badge&logo=onnx&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## Abstract / Overview

This repository presents a novel offline distillation pipeline designed for highly constrained edge hardware. We distill the knowledge of four distinct expert model architectures (MobileNetV4, EfficientNetV2, YOLOn) into a single unified student model powered by a frozen DINOv3 ConvNeXt-Tiny foundation backbone. By employing hierarchical feature routing, the student model seamlessly handles four disparate agricultural monitoring tasks simultaneously: Leaf Disease Classification, Pest Classification, Fruit Disease Classification, and Apple Yield Estimation (Detection/Counting).

The entire pipeline is engineered for zero-OOM execution on Google Colab's free-tier T4 GPU (16GB VRAM) using rigorous memory optimizations, O(1) dictionary mapping, and Automatic Mixed Precision (AMP).

## Repository Structure

- `config.py`: The single source of truth for all hyperparameters, dimensions, and hardware configurations.
- `dataset.py`: A highly optimized O(1) multi-task PyTorch DataLoader with dynamic padding and dynamic BBox-aware validation.
- `backbone.py`: The frozen DINOv3 ConvNeXt-Tiny structural feature extractor.
- `heads.py`: The decoupled task-specific heads aligned to ConvNeXt's hierarchical output scales (`f1`, `f2`, `f3`, `f4`).
- `model.py`: The composite student architecture wiring the backbone to the multi-task heads.
- `train.py`: The master training loop featuring gradient accumulation and AMP scaling.
- `losses.py`: The multi-task custom loss functions (Focal Loss, Masked Knowledge Distillation Loss, L1/MSE Box Loss).
- `export.py`: The script to export the trained PyTorch student model to ONNX with dynamic batch axes.
- `quantize.py`: The pipeline for INT8 Post-Training Quantization (PTQ) targeting TensorFlow Lite for on-device mobile deployment.

## Dataset Preparation

The training loop expects a flattened, pre-processed directory structure. Run the data preparation scripts inside `data_prep/` to achieve this layout before launching training:

```text
final_dataset/
├── images/             # All merged dataset images (.jpg, .png)
├── splits/
│   ├── train.txt       # List of image filenames for the training split
│   ├── val.txt         # List of image filenames for the validation split
│   └── test.txt        # List of image filenames for the testing split
├── labels/             # Ground Truth annotations
│   ├── leaf_labels.csv
│   ├── pest_labels.csv
│   ├── fruit_labels.csv
│   ├── yield_boxes.csv
│   └── yield_counts.csv
└── teacher_outputs/    # Soft-label predictions from expert models
    ├── leaf_soft.csv
    ├── pest_soft.csv
    ├── fruit_soft.csv
    └── yield_soft.csv
```

## Execution Guide

### Step 1: Configure Hyperparameters
Open `config.py` and adjust paths or scaling factors. The default settings are already heavily optimized for a 16GB Colab T4 GPU (Batch Size: 8, Accumulation: 4, Resolution: 512px).

### Step 2: Train the Multi-Task Student Model
Execute the main training loop. The script automatically applies Automatic Mixed Precision (AMP) and gradient accumulation to simulate an effective batch size of 32 without encountering CUDA Out-of-Memory errors.

```bash
python train.py
```

### Step 3: Export to ONNX
Once training completes, export the best model checkpoint to the ONNX intermediate format for inference graph constant folding.

```bash
python export.py
```

### Step 4: Quantize to INT8 (TensorFlow Lite)
*(Note: Requires conversion of the `.onnx` file to a TensorFlow `SavedModel` via `onnx2tf` first).*
Run the INT8 Post-Training Quantization pipeline. This script leverages a real-image representative dataset generator to accurately calibrate activation scales, outputting a mobile-ready `.tflite` model.

```bash
python quantize.py
```

## Research Contributions

1. **Frozen Backbone Multi-Task Learning**: Demonstrates that freezing a modern foundation feature extractor (DINOv3) significantly reduces VRAM footprint while providing rich, general-purpose hierarchical features capable of adapting to complex downstream agricultural tasks.
2. **O(1) Memory CSV Distillation**: Bypasses traditional Pandas dataframe overhead during dataloading by parsing soft-labels into native Python dictionaries at initialization, enabling zero-latency `__getitem__` mapping.
3. **Dynamic Distillation Masking**: Utilizes an `ignore_index` routing mechanism to safely distill partial datasets—allowing the student to learn from images that only contain annotations for a single task without penalizing unstructured heads.
4. **End-to-End Edge Deployment**: Provides a comprehensive path from multi-teacher PyTorch ensembles down to a fully quantized INT8 TensorFlow Lite model optimized for mobile smartphone inference.
