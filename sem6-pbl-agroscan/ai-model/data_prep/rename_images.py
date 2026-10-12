import os
import hashlib
from pathlib import Path

RAW_DATA_DIR = "../data"

def main():
    raw_root = Path(RAW_DATA_DIR).resolve()
    renamed = 0
    skipped = 0

    for file_path in raw_root.rglob("*"):
        if not file_path.is_file():
            continue

        # 1. Create a "canonical string" that is identical for matching YOLO images/labels,
        # but completely different for identical filenames in different train/test folders.
        canonical_parent = str(file_path.parent).replace(os.sep + "images", "").replace(os.sep + "labels", "")
        unique_string = f"{canonical_parent}_{file_path.stem}"

        # 2. Generate a 6-character hash from this unique string
        stem_hash = hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:6]
        prefix = f"{stem_hash}_"

        # 3. If the file already has this exact hash prefix, skip it
        if file_path.name.startswith(prefix):
            skipped += 1
            continue

        # 4. Rename the file in place
        new_name = f"{prefix}{file_path.name}"
        new_path = file_path.parent / new_name

        try:
            file_path.rename(new_path)
            renamed += 1
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")

    print(f"\nForce Rename Complete.")
    print(f"Successfully Renamed: {renamed}")
    print(f"Skipped (Already Hashed): {skipped}\n")

if __name__ == "__main__":
    main()
