import os
from collections import defaultdict
from pathlib import Path
from PIL import Image


RAW_DATA_DIR = "../data"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def audit_dataset(data_dir):
    valid_counts = defaultdict(int)
    corrupted_counts = defaultdict(int)

    for dirpath, _, filenames in os.walk(data_dir):
        for fname in filenames:
            if Path(fname).suffix.lower() in VALID_EXTENSIONS:
                file_path = os.path.join(dirpath, fname)
                try:
                    Image.open(file_path).verify()
                    valid_counts[dirpath] += 1
                except Exception:
                    corrupted_counts[dirpath] += 1

    return valid_counts, corrupted_counts


def print_report(valid_counts, corrupted_counts):
    all_dirs = sorted(set(valid_counts.keys()) | set(corrupted_counts.keys()))

    col1_header = "Folder Path"
    col2_header = "Valid Images"
    col3_header = "Corrupted Images"

    col1_width = max((len(d) for d in all_dirs), default=len(col1_header))
    col1_width = max(col1_width, len(col1_header))
    col2_width = max(len(col2_header), 12)
    col3_width = max(len(col3_header), 16)

    separator = f"+{'-' * (col1_width + 2)}+{'-' * (col2_width + 2)}+{'-' * (col3_width + 2)}+"
    header    = f"| {col1_header:<{col1_width}} | {col2_header:<{col2_width}} | {col3_header:<{col3_width}} |"

    print(separator)
    print(header)
    print(separator)

    total_valid = 0
    total_corrupted = 0

    if not all_dirs:
        empty_msg = "No images found."
        print(f"| {empty_msg:<{col1_width}} | {'0':<{col2_width}} | {'0':<{col3_width}} |")
    else:
        for dirpath in all_dirs:
            v = valid_counts[dirpath]
            c = corrupted_counts[dirpath]
            total_valid += v
            total_corrupted += c
            print(f"| {dirpath:<{col1_width}} | {v:<{col2_width}} | {c:<{col3_width}} |")

    print(separator)

    total_label = "TOTAL"
    print(f"| {total_label:<{col1_width}} | {total_valid:<{col2_width}} | {total_corrupted:<{col3_width}} |")
    print(separator)


def main():
    print(f"\nAuditing dataset at: {os.path.abspath(RAW_DATA_DIR)}\n")
    valid_counts, corrupted_counts = audit_dataset(RAW_DATA_DIR)
    print_report(valid_counts, corrupted_counts)


if __name__ == "__main__":
    main()
