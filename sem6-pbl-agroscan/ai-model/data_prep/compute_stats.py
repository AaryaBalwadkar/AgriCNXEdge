import os
import json
import numpy as np
from PIL import Image


IMAGES_DIR  = "../final_dataset/images"
TRAIN_SPLIT = "../final_dataset/splits/train.txt"
META_DIR    = "../final_dataset/meta"

PROGRESS_INTERVAL = 500


def load_train_ids(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_stats(image_ids, images_dir):
    channel_sums         = np.zeros(3, dtype=np.float64)
    channel_squared_sums = np.zeros(3, dtype=np.float64)
    total_pixels         = 0

    for idx, image_id in enumerate(image_ids, start=1):
        img_path = os.path.join(images_dir, image_id)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        pixels = np.array(img, dtype=np.float64) / 255.0
        pixels = pixels.reshape(-1, 3)

        channel_sums         += pixels.sum(axis=0)
        channel_squared_sums += (pixels ** 2).sum(axis=0)
        total_pixels         += pixels.shape[0]

        if idx % PROGRESS_INTERVAL == 0:
            print(f"  Processed {idx}/{len(image_ids)} images...")

    mean = channel_sums / total_pixels
    std  = np.sqrt((channel_squared_sums / total_pixels) - (mean ** 2))

    return mean, std


def main():
    os.makedirs(META_DIR, exist_ok=True)

    print(f"\nReading train split from: {os.path.abspath(TRAIN_SPLIT)}")
    image_ids = load_train_ids(TRAIN_SPLIT)
    print(f"Found {len(image_ids)} training image(s).\n")

    mean, std = compute_stats(image_ids, IMAGES_DIR)

    mean_list = mean.tolist()
    std_list  = std.tolist()

    print(f"\n  Mean (R, G, B) : {[round(v, 6) for v in mean_list]}")
    print(f"  Std  (R, G, B) : {[round(v, 6) for v in std_list]}\n")

    stats = {"mean": mean_list, "std": std_list}
    out_path = os.path.join(META_DIR, "dataset_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats saved to : {os.path.abspath(out_path)}\n")


if __name__ == "__main__":
    main()
