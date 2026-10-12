import os
import shutil
from pathlib import Path


RAW_DATA_DIR     = "../data"
FINAL_IMAGES_DIR = "../final_dataset/images"

VALID_EXTENSIONS  = {".jpg", ".jpeg", ".png"}
PROGRESS_INTERVAL = 1000


def merge(raw_dir, final_dir):
    raw_root   = Path(raw_dir).resolve()
    final_root = Path(final_dir).resolve()
    copied = 0

    for file_path in raw_root.rglob("*"):
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        if final_root in file_path.parents:
            continue

        dest = final_root / file_path.name
        if not dest.exists():
            shutil.copy2(str(file_path), str(dest))
            copied += 1
            if copied % PROGRESS_INTERVAL == 0:
                print(f"Copied {copied} images...")

    return copied


def main():
    os.makedirs(FINAL_IMAGES_DIR, exist_ok=True)

    print(f"\nSource      : {os.path.abspath(RAW_DATA_DIR)}")
    print(f"Destination : {os.path.abspath(FINAL_IMAGES_DIR)}\n")

    total = merge(RAW_DATA_DIR, FINAL_IMAGES_DIR)

    print(f"\nDone. {total} image(s) successfully merged into {os.path.abspath(FINAL_IMAGES_DIR)}\n")


if __name__ == "__main__":
    main()
