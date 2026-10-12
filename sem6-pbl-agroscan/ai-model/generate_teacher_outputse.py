import os
import csv
import json
import gc
import torch
import numpy as np
from PIL import Image

# --- COLAB LOCAL PATHS (Where you will unzip the data) ---
IMAGES_DIR = "/content/final_dataset/images"
OUTPUT_DIR = "/content/final_dataset/teacher_outputs"

# --- GOOGLE DRIVE PATHS (Where your models live) ---
FRUIT_MODEL_PATH = "/content/drive/MyDrive/newmodel/models/fruit_model.pth"
LEAF_MODEL_PATH  = "/content/drive/MyDrive/newmodel/models/leaf_model.keras"
PEST_MODEL_PATH  = "/content/drive/MyDrive/newmodel/models/insect_model"
YIELD_MODEL_PATH = "/content/drive/MyDrive/newmodel/models/yolo26s.pt"

def get_image_paths(images_dir):
    valid = {".jpg", ".jpeg", ".png"}
    return [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if os.path.splitext(f)[1].lower() in valid
    ]

def run_pytorch_teachers(image_paths):
    import torchvision.transforms as T
    import timm 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    print("Loading PyTorch Fruit Teacher (MobileNetV4-Small)...")
    model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(FRUIT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    output_path = os.path.join(OUTPUT_DIR, "fruit_soft.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "fruit_logits"])
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(device)
                    logits = model(tensor).squeeze(0).cpu().tolist()
                    writer.writerow([os.path.basename(img_path), json.dumps(logits)])
                except Exception:
                    writer.writerow([os.path.basename(img_path), json.dumps([])])

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Phase 1 complete: fruit_soft.csv written.")

def run_keras_teachers(image_paths):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    def preprocess(img_path, target_size=(224, 224)):
        img = Image.open(img_path).convert("RGB").resize(target_size)
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def load_keras_model(path):
        import shutil
        import tempfile
        import os
        import tensorflow as tf

        # 1. Handle the case where the .keras file was accidentally unzipped into a folder
        if os.path.isdir(path):
            # If it happens to be a SavedModel, it will have a .pb file
            if os.path.exists(os.path.join(path, "saved_model.pb")):
                return tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default"), True
            
            # If no .pb file, it is an unzipped .keras archive. Let's re-zip it!
            if path.endswith(".keras"):
                print(f"Unzipped directory detected for {os.path.basename(path)}. Re-zipping into memory...")
                
                # Create a temporary zip file
                temp_base = os.path.join(tempfile.gettempdir(), "temp_keras_archive")
                shutil.make_archive(temp_base, 'zip', path)
                temp_zip_path = temp_base + ".zip"
                
                # Keras strictly demands the .keras extension, so we rename the .zip file
                temp_keras_path = temp_base + ".keras"
                if os.path.exists(temp_keras_path):
                    os.remove(temp_keras_path) # Clean up from previous runs
                os.rename(temp_zip_path, temp_keras_path)
                
                # Load the newly zipped AND correctly named file
                return tf.keras.models.load_model(temp_keras_path, compile=False), False

        # 2. Standard loading if it is correctly recognized as a file
        if path.endswith(".keras") or path.endswith(".h5"):
            return tf.keras.models.load_model(path, compile=False), False
            
        # 3. Fallback for files without extensions
        h5_path = path + ".h5"
        if not os.path.exists(h5_path):
            shutil.copy(path, h5_path)
        return tf.keras.models.load_model(h5_path, compile=False), False

    def run_inference(model, inp, is_tfsm):
        output = model(inp, training=False)
        if is_tfsm:
            output = list(output.values())[0]
        if hasattr(output, "numpy"):
            return output.numpy()[0].tolist()
        return output[0].tolist()

    # --- LEAF TEACHER ---
    print("Loading Keras Leaf Teacher (EfficientNetV2-B0)...")
    leaf_model, leaf_is_tfsm = load_keras_model(LEAF_MODEL_PATH)
    output_path = os.path.join(OUTPUT_DIR, "leaf_soft.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "leaf_logits"])
        for img_path in image_paths:
            try:
                inp   = preprocess(img_path)
                probs = run_inference(leaf_model, inp, leaf_is_tfsm)
                writer.writerow([os.path.basename(img_path), json.dumps(probs)])
            except Exception:
                writer.writerow([os.path.basename(img_path), json.dumps([])])

    del leaf_model
    tf.keras.backend.clear_session()
    gc.collect()
    print("Phase 2a complete: leaf_soft.csv written.")

    # --- PEST TEACHER ---
    # print("Loading Keras Pest Teacher (EfficientNetV2-B0)...")
    # pest_model, pest_is_tfsm = load_keras_model(PEST_MODEL_PATH)
    # output_path = os.path.join(OUTPUT_DIR, "pest_soft.csv")
    # with open(output_path, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["image_id", "pest_logits"])
    #     for img_path in image_paths:
    #         try:
    #             inp   = preprocess(img_path)
    #             probs = run_inference(pest_model, inp, pest_is_tfsm)
    #             writer.writerow([os.path.basename(img_path), json.dumps(probs)])
    #         except Exception:
    #             writer.writerow([os.path.basename(img_path), json.dumps([])])

    # del pest_model
    # tf.keras.backend.clear_session()
    # gc.collect()
    # print("Phase 2b complete: pest_soft.csv written.")

def run_yolo_teacher(image_paths):
    from ultralytics import YOLO

    print("Loading YOLOv8 Yield Teacher...")
    model = YOLO(YIELD_MODEL_PATH)

    output_path = os.path.join(OUTPUT_DIR, "yield_soft.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "json_boxes", "count"])
        for img_path in image_paths:
            try:
                results = model(img_path, imgsz=832, verbose=False)
                boxes_data = []
                for result in results:
                    for box in result.boxes:
                        xyxy  = box.xyxy[0].tolist()
                        conf  = float(box.conf[0])
                        cls   = int(box.cls[0])
                        boxes_data.append({"xyxy": xyxy, "conf": conf, "cls": cls})
                writer.writerow([
                    os.path.basename(img_path),
                    json.dumps(boxes_data),
                    len(boxes_data)
                ])
            except Exception:
                writer.writerow([os.path.basename(img_path), json.dumps([]), 0])

    del model
    gc.collect()
    print("Phase 3 complete: yield_soft.csv written.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = get_image_paths(IMAGES_DIR)

    print(f"\nOffline distillation started. {len(image_paths)} image(s) found.\n")

    # run_keras_teachers(image_paths)
    # run_yolo_teacher(image_paths)
    run_pytorch_teachers(image_paths)

    print(f"\nAll 4 teacher outputs saved to: {os.path.abspath(OUTPUT_DIR)}\n")