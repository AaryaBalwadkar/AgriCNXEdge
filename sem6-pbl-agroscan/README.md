# Sem 6 PBL — Agroscan (Green Apple)

Source: `Agroscan/` (Colab notebook + `newmodel/` frozen-DINOv3 multi-task student).
Kept: `Green_Apple_Data_Processing.ipynb`, `ai-model/*.py` (17), `*.md/*.txt`, `data_prep/` + logs.
Excluded (see `../DATA_SETUP.md`): `data/` (375 MB), `final_dataset/` (4.6 GB), `dinov3/` vendor (use submodule/pip),
`checkpoints/*.pth`, `models/*.pth`, `*.onnx/*.ort`, `__pycache__`, `data_corrupted/` (was empty).
Known bugs: see `../MISSING_FILES.md` §C (C1–C9, C11) — fix `config.py` weights path first.
