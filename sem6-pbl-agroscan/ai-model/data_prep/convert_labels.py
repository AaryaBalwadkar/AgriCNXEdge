import os
import csv
from pathlib import Path

RAW_DATA_DIR = "../data"
LABELS_DIR   = "../final_dataset/labels"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def clean_key(filename):
    """
    Strips extensions and annoying Roboflow/Folder prefixes.
    This ensures that 'yield_images_123.jpg' and 'yield_labels_123.txt' 
    both resolve to the exact same dictionary key: '123'
    """
    name = os.path.splitext(filename)[0].lower()
    name = name.replace("yield_labels_", "").replace("yield_images_", "")
    name = name.replace("labels_", "").replace("images_", "")
    return name

def write_csv(filepath, header, rows):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)

def build_label_tables(raw_dir):
    leaf_data    = []
    fruit_data   = []
    pest_data    = []
    yield_boxes  = []
    yield_counts = []

    raw_root = Path(raw_dir).resolve()
    
    print(f"\nScanning for text files in: {raw_root}...")
    
    # Create a smart lookup dictionary that strips out mismatches
    txt_lookup = {}
    for p in raw_root.rglob("*.[tT][xX][tT]"):
        key = clean_key(p.name)
        txt_lookup[key] = p
        
    print(f"--> Found {len(txt_lookup)} total .txt files in the dataset.\n")

    for file_path in raw_root.rglob("*"):
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        filename   = file_path.name
        path_parts = [p.lower() for p in file_path.parts]
        class_name = file_path.parent.name

        if "leaf" in path_parts:
            leaf_data.append([filename, class_name])

        elif "fruit" in path_parts:
            fruit_data.append([filename, class_name])

        elif "pest" in path_parts:
            pest_data.append([filename, class_name])

        elif "yield" in path_parts:
            # Generate the clean key to match against the dictionary
            key = clean_key(filename)
            txt_path = txt_lookup.get(key)
            
            if txt_path is not None and txt_path.exists():
                lines = [
                    line.strip()
                    for line in txt_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for annotation in lines:
                    yield_boxes.append([filename, annotation])
                yield_counts.append([filename, len(lines)])
            else:
                # Fallback: Just look in sibling labels directory if dictionary somehow misses
                parent_str = str(file_path.parent)
                if "images" in parent_str.lower():
                    sibling_dir = Path(parent_str.replace("images", "labels").replace("Images", "labels"))
                    expected_txt = file_path.with_suffix(".txt").name
                    fallback_path = sibling_dir / expected_txt
                    if fallback_path.exists():
                        lines = [
                            line.strip()
                            for line in fallback_path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                        for annotation in lines:
                            yield_boxes.append([filename, annotation])
                        yield_counts.append([filename, len(lines)])
                        continue
                
                # If absolutely no text file exists, it's a background image with 0 apples
                yield_counts.append([filename, 0])

    return leaf_data, fruit_data, pest_data, yield_boxes, yield_counts


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    leaf_data, fruit_data, pest_data, yield_boxes, yield_counts = build_label_tables(RAW_DATA_DIR)

    cls_header   = ["image_id", "class_name"]
    det_header   = ["image_id", "yolo_annotation"]
    count_header = ["image_id", "fruit_count"]

    labels_root = Path(LABELS_DIR)

    n_leaf   = write_csv(labels_root / "leaf_labels.csv",  cls_header,   leaf_data)
    n_fruit  = write_csv(labels_root / "fruit_labels.csv", cls_header,   fruit_data)
    n_pest   = write_csv(labels_root / "pest_labels.csv",  cls_header,   pest_data)
    n_boxes  = write_csv(labels_root / "yield_boxes.csv",  det_header,   yield_boxes)
    n_counts = write_csv(labels_root / "yield_counts.csv", count_header, yield_counts)

    print(f"Label CSVs written to: {os.path.abspath(LABELS_DIR)}")
    print(f"  leaf_labels.csv  : {n_leaf:>6} row(s)")
    print(f"  fruit_labels.csv : {n_fruit:>6} row(s)")
    print(f"  pest_labels.csv  : {n_pest:>6} row(s)")
    print(f"  yield_boxes.csv  : {n_boxes:>6} row(s)  (one row per bounding box)")
    print(f"  yield_counts.csv : {n_counts:>6} row(s)  (one row per image)")
    print(f"\n  Total rows       : {n_leaf + n_fruit + n_pest + n_boxes + n_counts:>6}\n")

if __name__ == "__main__":
    main()