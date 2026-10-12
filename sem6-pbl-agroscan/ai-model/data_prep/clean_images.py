import os
import shutil
from pathlib import Path
from PIL import Image


RAW_DATA_DIR   = "../data"
QUARANTINE_DIR = "../data_corrupted"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def quarantine_file(file_path, raw_root, quarantine_root):
    relative_path    = Path(file_path).relative_to(raw_root)
    quarantine_path  = Path(quarantine_root) / relative_path
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file_path), str(quarantine_path))
    return quarantine_path


def clean_dataset(raw_dir, quarantine_dir):
    raw_root       = Path(raw_dir).resolve()
    quarantine_root = Path(quarantine_dir).resolve()
    removed_count  = 0

    for file_path in raw_root.rglob("*"):
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        is_corrupted = False

        if os.path.getsize(file_path) == 0:
            is_corrupted = True
        else:
            try:
                Image.open(file_path).verify()
            except Exception:
                is_corrupted = True

        if is_corrupted:
            dest = quarantine_file(file_path, raw_root, quarantine_root)
            print(f"  [QUARANTINED] {file_path}  ->  {dest}")
            removed_count += 1

    return removed_count


def main():
    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    print(f"\nScanning: {os.path.abspath(RAW_DATA_DIR)}")
    print(f"Quarantine destination: {os.path.abspath(QUARANTINE_DIR)}\n")

    removed = clean_dataset(RAW_DATA_DIR, QUARANTINE_DIR)

    print(f"\n{'=' * 52}")
    print(f"  Scan complete. {removed} corrupted file(s) quarantined.")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
