import os
import csv
from collections import Counter
from PIL import Image

IMAGES_DIR = "../final_dataset/images"
YIELD_CSV = "../final_dataset/labels/yield_counts.csv"

def get_yield_image_ids(csv_path):
    image_ids = set()
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return image_ids
        
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.add(row["image_id"])
    return image_ids

def check_yield_resolutions(image_ids, images_dir):
    resolutions = Counter()
    total_images = 0
    missing_images = 0
    
    print(f"Scanning {len(image_ids)} Yield images in: {images_dir}...")
    
    for img_id in image_ids:
        file_path = os.path.join(images_dir, img_id)
        if os.path.exists(file_path):
            try:
                with Image.open(file_path) as img:
                    w, h = img.size
                    resolutions[(w, h)] += 1
                    total_images += 1
            except Exception as e:
                print(f"Error reading {img_id}: {e}")
        else:
            missing_images += 1
                
    return resolutions, total_images, missing_images

def main():
    yield_ids = get_yield_image_ids(YIELD_CSV)
    if not yield_ids:
        return
        
    resolutions, total, missing = check_yield_resolutions(yield_ids, IMAGES_DIR)
    
    print(f"\nAnalyzed {total} YIELD images. Here is the resolution breakdown:\n")
    
    print(f"+{'-'*22}+{'-'*10}+")
    print(f"| {'Resolution (W x H)':<20} | {'Count':<8} |")
    print(f"+{'-'*22}+{'-'*10}+")
    
    for (w, h), count in resolutions.most_common():
        res_str = f"{w} x {h}"
        print(f"| {res_str:<20} | {count:<8} |")
        
    print(f"+{'-'*22}+{'-'*10}+\n")
    
    if missing > 0:
        print(f"Warning: {missing} images listed in the CSV were not found in the folder.")
        
    if len(resolutions) == 1:
        print("Result: All your YIELD images are EXACTLY the same size. This makes normalization easy!")
    else:
        print(f"Result: You have {len(resolutions)} different image resolutions just in the Yield dataset!")

if __name__ == "__main__":
    main()