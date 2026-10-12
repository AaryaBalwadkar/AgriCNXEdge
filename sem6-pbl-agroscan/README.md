# Semester 6 PBL — Agroscan (green apple)

The starting point of the project: green-apple disease data preparation and the first
multi-task model (`Agroscan/` in the old folders).

- `Green_Apple_Data_Processing.ipynb` — renames, checks, measures and augments the raw
  green-apple images. Runs on Google Drive data.
- `ai-model/` — training code: network (`backbone.py`, `heads.py`, `model.py`), data
  loading (`dataset.py`, `config.py`), losses (`losses.py`), training and evaluation
  (`train.py`, `evaluate.py`, `evaluate_yield.py`), export and quantization
  (`export.py`, `export2.py`, `convert_ort.py`, `quantize.py`), design notes
  (`architecture.md`, `README.md`).
- `ai-model/data_prep/` — eight dataset preparation scripts with their run logs.

Needs from Drive (`DATA_SETUP.md`): `final_dataset/` and the DINOv3 backbone weights.
Known issues: see `../KNOWN-GAPS.md`, items C1–C9 and C11.
