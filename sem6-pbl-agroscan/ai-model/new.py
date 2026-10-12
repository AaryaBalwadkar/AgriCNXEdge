import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load your teammate's model
LEAF_MODEL_PATH = "./models/leaf_model.h5" # Or .keras
model = tf.keras.models.load_model(LEAF_MODEL_PATH, compile=False)

def test_image(image_path, true_label):
    # Preprocess exactly how your teammate's script did
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    
    # Get the 5 probabilities
    preds = model.predict(arr, verbose=0)[0]
    
    # Find which index (0-4) has the highest confidence
    predicted_index = np.argmax(preds)
    confidence = preds[predicted_index]
    
    print(f"I fed the model a {true_label} image.")
    print(f"The model spiked at INDEX {predicted_index} with {confidence*100:.1f}% confidence.\n")

# 2. Test one clear, obvious image from each of your 5 main folders
print("--- REVERSE ENGINEERING THE CLASS INDICES ---")
test_image("/content/final_dataset/leaf/Apple___Apple_scab/image1.jpg", "SCAB")
test_image("/content/final_dataset/leaf/Apple___Black_rot/image1.jpg", "BLACK ROT")
test_image("/content/final_dataset/leaf/Apple___Cedar_apple_rust/image1.jpg", "CEDAR RUST")
test_image("/content/final_dataset/leaf/Apple___healthy/image1.jpg", "HEALTHY")
test_image("/content/final_dataset/leaf/Alternaria/image1.jpg", "ALTERNARIA")