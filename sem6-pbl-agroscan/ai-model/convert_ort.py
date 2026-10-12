import onnxruntime as ort
import os

def convert_to_ort():
    onnx_path = "model_fp32.onnx"
    ort_path = "model_fp32.ort"

    if os.path.exists(ort_path):
        os.remove(ort_path)
        print(f"🗑️ Deleted old/corrupted {ort_path}")

    print(f"⚙️ Loading {onnx_path}...")

    so = ort.SessionOptions()
    
    # =========================================================================
    # EXPERT FIX: Mobile-Safe Optimization
    # Downgrade to BASIC/EXTENDED. Never use 'ALL' for mobile exports, 
    # as it injects PC-specific hardware nodes that crash the ORT saver.
    # =========================================================================
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    so.add_session_config_entry('session.save_model_format', 'ORT')
    
    absolute_ort_path = os.path.abspath(ort_path)
    so.optimized_model_filepath = absolute_ort_path

    try:
        # Force CPU Provider so CUDA doesn't interfere with the mobile export
        session = ort.InferenceSession(
            onnx_path, 
            so, 
            providers=['CPUExecutionProvider']
        )
        print(f"\n✅ SUCCESS! Mobile-ready ORT model saved to: {absolute_ort_path}")
    except Exception as e:
        print(f"\n❌ FAILED. Error details:\n{e}")

if __name__ == "__main__":
    convert_to_ort()