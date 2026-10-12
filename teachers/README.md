# Teacher models (PDDD + YOLO)

The pretrained teachers the student learns from: PDDD fruit/leaf/pest classifiers and
three YOLO apple-detection runs. Code and small config files only.

- `PDDD-Fruit-1` — fruit + pest teachers (complete).
- `PDDD-Fruit-3-and-PDDD-Leaf-1` — newest unified set (fruit + leaf + pest).
- `PDDD-Pest-1` — pest only, incomplete (missing training scripts, see
  `../KNOWN-GAPS.md`, A3).
- `YOLO26s-apple-1` — the complete detection run with its dataset config.
- `YOLO26s-apple-2` — rerun without data folder (different seed).
- `YOLO26s-apple-3` — staging experiment, incomplete.

Training images and teacher weight files are on Drive (`DATA_SETUP.md`).
