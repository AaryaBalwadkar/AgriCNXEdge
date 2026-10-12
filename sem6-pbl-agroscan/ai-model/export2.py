import os
import sys

sys.path.insert(0, os.path.abspath("./dinov3"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify

from config import Config
from model import AgriculturalModel

# =========================================================================
# EXPERT FIX: The ONNX Export Wrapper
# ONNX destroys Python dictionaries. This wrapper locks the outputs into a 
# strict tuple so the Android app gets exactly the tensors it asks for.
# =========================================================================
class ONNXExportWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Tweak this number! 
        # 0.33 perfectly counters your training temperature of 3.0.
        # If confidence is STILL too low, lower this to 0.20 or 0.15.
        self.sharpening_factor = 0.33

    def forward(self, x):
        # 1. Run the base model to get the dictionary
        preds = self.base_model(x)
        
        # 2. Extract logits and apply Softmax so Android gets 0-100% probabilities 
        # instead of raw, unreadable logit numbers.
        # Divide by the sharpening factor BEFORE Softmax
        fruit_probs = F.softmax(preds["fruit"] / self.sharpening_factor, dim=1)
        leaf_probs = F.softmax(preds["leaf"] / self.sharpening_factor, dim=1)
        pest_probs = F.softmax(preds["pest"] / self.sharpening_factor, dim=1)
        
        # 3. Extract the nested Yield heatmaps
        yield_heatmap = preds["yield_map"]["heatmap"]
        yield_offset = preds["yield_map"]["offset"]
        yield_wh = preds["yield_map"]["wh"]
        
        # 4. Return as a STRICT Tuple that matches output_names EXACTLY
        return (
            fruit_probs,
            leaf_probs,
            pest_probs,
            yield_heatmap,
            yield_offset,
            yield_wh
        )

def export_model():
    print("🚀 Starting ONNX Export Pipeline (FLOAT32 Pure Mode)...")
    config = Config()
    
    config.model.dinov3_repo_dir = "./dinov3"
    config.model.dinov3_weights = "./models/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    config.model.pretrained_backbone = True
    
    device = torch.device("cpu") 
    checkpoint_path = os.path.join(config.train.checkpoints_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Cannot find checkpoint at {checkpoint_path}")
        return

    print("📦 Building DINOv3 Architecture...")
    base_model = AgriculturalModel(model_config=config.model).to(device)
    
    print(f"🧠 Loading custom trained weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state" in checkpoint:
        base_model.load_state_dict(checkpoint["model_state"])
    else:
        base_model.load_state_dict(checkpoint)
        
    base_model.eval()

    # Wrap the model!
    print("🔒 Wrapping model to secure output tensor mapping...")
    wrapped_model = ONNXExportWrapper(base_model)
    wrapped_model.eval()

    print(f"📐 Generating static dummy input...")
    dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size, device=device)

    onnx_raw_path = "model_raw.onnx"
    onnx_fp32_path = "model_fp32.onnx"

    print("⚙️ Exporting PyTorch graph to ONNX...")
    torch.onnx.export(
        wrapped_model,               # <-- Pass the WRAPPED model, not the base model!
        dummy_input, 
        onnx_raw_path,
        export_params=True,
        opset_version=18,          
        do_constant_folding=True,  
        input_names=["input_image"],
        # These names now perfectly map to the Tuple returned by the wrapper
        output_names=["fruit_probs", "leaf_probs", "pest_probs", "yield_heatmap", "yield_offset", "yield_wh"],
        dynamic_axes=None
    )

    print("🧹 Running ONNX Simplifier...")
    raw_model = onnx.load(onnx_raw_path)
    model_simp, check = simplify(raw_model)
    
    if check:
        onnx.save(model_simp, onnx_fp32_path)
        print(f"✅ Pristine FP32 ONNX saved to {onnx_fp32_path} ({os.path.getsize(onnx_fp32_path) / (1024*1024):.1f} MB)")
    else:
        print("⚠️ Simplifier failed. Saving raw model.")
        onnx.save(raw_model, onnx_fp32_path)

    if os.path.exists(onnx_raw_path):
        os.remove(onnx_raw_path)

    print("\n🎉 Export Complete!")
    print("Convert to Android format using:")
    print("python -m onnxruntime.tools.convert_onnx_models_to_ort model_fp32.onnx")

if __name__ == "__main__":
    export_model()