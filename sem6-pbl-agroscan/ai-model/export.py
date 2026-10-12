import os
import sys

# Forces Python to look inside the GitHub repo for its own files (fixes the logging error)
sys.path.insert(0, os.path.abspath("./dinov3"))

import torch
import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType

from config import Config
from model import AgriculturalModel

def export_model():
    print("🚀 Starting ONNX Export Pipeline (Uncompromised Mode)...")
    config = Config()
    
    # =================================================================
    # EXPERT FIX: Set to True. Meta's code crashes if weights=None.
    # We will let it load the base weights to build the architecture, 
    # then instantly overwrite them with your trained checkpoint.
    # =================================================================
    config.model.dinov3_repo_dir = "./dinov3"
    config.model.dinov3_weights = "./models/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    config.model.pretrained_backbone = True
    
    # 1. Prepare the Environment (CPU for NPU/Mobile compatibility)
    device = torch.device("cpu") 
    checkpoint_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Cannot find your trained checkpoint at {checkpoint_path}")
        return

    # 2. Load the PyTorch Model Architecture
    print("📦 Building DINOv3 Architecture...")
    model = AgriculturalModel(model_config=config.model).to(device)
    
    # Load YOUR custom trained weights over the architecture
    print(f"🧠 Loading custom trained weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Create a Dummy Input Tensor (Strictly Static for NPU)
    print(f"📐 Generating static dummy input (1, 3, {config.data.image_size}, {config.data.image_size})...")
    dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size, device=device)

    # File paths
    onnx_raw_path = "model_raw.onnx"
    onnx_sim_path = "model_sim.onnx"
    onnx_quant_path = "model_int8.onnx"

    # 4. Standard ONNX Export
    print("⚙️ Exporting PyTorch graph to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_raw_path,
        export_params=True,
        opset_version=14,          # Highly stable for Android NNAPI
        do_constant_folding=True,  
        input_names=["input_image"],
        
        # -----------------------------------------------------------------
        # EXPERT FIX: PyTorch alphabetically flattens dictionaries!
        # In heads.py we return: fruit, leaf, pest, yield_map (heatmap, offset, wh)
        # This list MUST match that exact alphabetical extraction order.
        # -----------------------------------------------------------------
        output_names=[
            "fruit_logits", 
            "leaf_logits", 
            "pest_logits", 
            "yield_heatmap", 
            "yield_offset", 
            "yield_wh"
        ],
        
        dynamic_axes=None          # STRICTLY NONE! NPUs demand static graphs
    )
    print(f"✅ Raw ONNX saved to {onnx_raw_path} ({os.path.getsize(onnx_raw_path) / (1024*1024):.1f} MB)")

    # 5. ONNX Simplify (Removes PyTorch bloat for Mobile)
    print("🧹 Running ONNX Simplifier...")
    raw_model = onnx.load(onnx_raw_path)
    model_simp, check = simplify(raw_model)
    
    if not check:
        print("⚠️ Warning: ONNX Simplifier failed validation. Proceeding with raw model.")
        onnx_sim_path = onnx_raw_path
    else:
        onnx.save(model_simp, onnx_sim_path)
        print(f"✅ Simplified ONNX saved to {onnx_sim_path} ({os.path.getsize(onnx_sim_path) / (1024*1024):.1f} MB)")

    # 6. INT8 Quantization (Crushes file size to < 40MB)
    print("🗜️ Applying INT8 Dynamic Quantization...")
    quantize_dynamic(
        model_input=onnx_sim_path,
        model_output=onnx_quant_path,
        per_channel=False,         
        reduce_range=True,         
        weight_type=QuantType.QUInt8
    )
    print(f"✅ Quantized ONNX saved to {onnx_quant_path} ({os.path.getsize(onnx_quant_path) / (1024*1024):.1f} MB)")

    # 7. Cleanup
    if os.path.exists(onnx_raw_path):
        os.remove(onnx_raw_path)
    if os.path.exists(onnx_sim_path) and onnx_sim_path != onnx_raw_path:
        os.remove(onnx_sim_path)

    print("\n🎉 Export Complete!")
    print(f"Final Model: {onnx_quant_path}")
    print("Run this to get your Android file: python -m onnxruntime.tools.convert_onnx_models_to_ort model_int8.onnx")

if __name__ == "__main__":
    export_model()