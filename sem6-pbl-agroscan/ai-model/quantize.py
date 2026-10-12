import os
import random
import numpy as np
import tensorflow as tf

from config import Config


SAVED_MODEL_DIR  = "saved_model_dir"
OUTPUT_TFLITE    = "agricultural_model_int8.tflite"
NUM_CALIB_IMAGES = 150

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_calib_image_paths(images_dir, n):
    valid = {".jpg", ".jpeg", ".png"}
    all_paths = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in valid
    ]
    return random.sample(all_paths, min(n, len(all_paths)))


def preprocess(img_path, image_size):
    raw = tf.io.read_file(img_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return tf.expand_dims(img, axis=0)


def make_representative_data_gen(images_dir, image_size, n=NUM_CALIB_IMAGES):
    calib_paths = get_calib_image_paths(images_dir, n)

    def generator():
        for img_path in calib_paths:
            try:
                tensor = preprocess(img_path, image_size)
                yield [tensor.numpy().astype(np.float32)]
            except Exception:
                continue

    return generator


def quantize(config):
    image_size = config.data.image_size
    images_dir = config.data.images_dir

    print(f"\nLoading SavedModel from: {os.path.abspath(SAVED_MODEL_DIR)}")
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = make_representative_data_gen(
        images_dir, image_size, NUM_CALIB_IMAGES
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    print(f"Running INT8 calibration on {NUM_CALIB_IMAGES} images at {image_size}x{image_size}...")
    tflite_model = converter.convert()

    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nINT8 quantization complete.")
    print(f"  Output : {os.path.abspath(OUTPUT_TFLITE)}")
    print(f"  Size   : {size_mb:.2f} MB\n")


def main():
    config = Config()
    quantize(config)


if __name__ == "__main__":
    main()
