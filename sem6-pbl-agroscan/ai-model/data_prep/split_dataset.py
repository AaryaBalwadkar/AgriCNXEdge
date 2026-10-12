import os
import csv
import random
import math
from collections import defaultdict


LABELS_DIR = "../final_dataset/labels"
SPLITS_DIR = "../final_dataset/splits"

random.seed(42)

MAX_PER_CLASS = 600


def read_cls_csv(filepath):
    rows = []
    if not os.path.exists(filepath):
        return rows
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_id"], row["class_name"]))
    return rows


def read_det_csv(filepath):
    image_ids = set()
    if not os.path.exists(filepath):
        return image_ids
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_ids.add(row["image_id"])
    return image_ids


def write_split(filepath, ids):
    with open(filepath, "w", encoding="utf-8") as f:
        for image_id in ids:
            f.write(image_id + "\n")
    return len(ids)


def smart_split(images):
    n = len(images)

    if n < 3:
        return images, [], []

    train_idx = max(1, math.floor(n * 0.70))
    val_idx   = train_idx + max(1, math.floor(n * 0.15))

    if val_idx >= n:
        val_idx = n - 1

    if train_idx >= val_idx:
        train_idx = val_idx - 1

    train = images[:train_idx]
    val   = images[train_idx:val_idx]
    test  = images[val_idx:]

    return train, val, test


def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    class_groups = defaultdict(list)

    for csv_file in ["leaf_labels.csv", "fruit_labels.csv", "pest_labels.csv"]:
        for image_id, class_name in read_cls_csv(os.path.join(LABELS_DIR, csv_file)):
            class_groups[class_name].append(image_id)

    for image_id in read_det_csv(os.path.join(LABELS_DIR, "yield_boxes.csv")):
        class_groups["yield_task"].append(image_id)

    train_ids = []
    val_ids   = []
    test_ids  = []

    summary_rows = []

    for class_name, images in sorted(class_groups.items()):
        if len(images) > MAX_PER_CLASS:
            images = random.sample(images, MAX_PER_CLASS)
        random.shuffle(images)

        train, val, test = smart_split(images)

        train_ids.extend(train)
        val_ids.extend(val)
        test_ids.extend(test)

        total = len(train) + len(val) + len(test)
        summary_rows.append((class_name, total, len(train), len(val), len(test)))

    n_train = write_split(os.path.join(SPLITS_DIR, "train.txt"), train_ids)
    n_val   = write_split(os.path.join(SPLITS_DIR, "val.txt"),   val_ids)
    n_test  = write_split(os.path.join(SPLITS_DIR, "test.txt"),  test_ids)

    col_w = [
        max(len(r[0]) for r in summary_rows + [("Class", 0, 0, 0, 0)]),
        10, 8, 7, 8
    ]
    headers = ["Class", "Kept Total", "Train", "Val", "Test"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(f"\nSplit summary — seed=42  |  max_per_class={MAX_PER_CLASS}  |  all classes kept\n")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in summary_rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)
    print(fmt.format("TOTAL", n_train + n_val + n_test, n_train, n_val, n_test))
    print(sep)
    print(f"\nFiles written to: {os.path.abspath(SPLITS_DIR)}")
    print(f"  train.txt : {n_train} image(s)")
    print(f"  val.txt   : {n_val} image(s)")
    print(f"  test.txt  : {n_test} image(s)\n")


if __name__ == "__main__":
    main()
