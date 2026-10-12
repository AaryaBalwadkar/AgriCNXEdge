# AgriCNXEdge — BTech Project (Semester 6 PBL continued in Semester 7)

Edge-AI system for apple crop health: one model that classifies leaf disease, fruit
disease and pests, and counts fruit (yield), running on a phone (Android app) and on a
server (web API). The work started as the Semester 6 project-based-learning project
(Agroscan) and was continued and completed as the Semester 7 BTech project.

This repository holds the complete code, the Android app, the web service, and the
small result files (tables, training logs, evaluation summaries). Large files —
trained models, datasets and videos — are stored on Google Drive (link below)
because GitHub does not accept files over 100 MB. `DATA_SETUP.md` lists every
large file and exactly where to place it.

Drive: https://drive.google.com/drive/folders/1ZAEEz724iC9ZOZ7GiSSvtMCLUDnvcTqx?usp=sharing

## What's inside

### `sem6-pbl-agroscan/` — Semester 6 PBL (Agroscan)
The starting point: green-apple disease data preparation and the first multi-task model.
- `Green_Apple_Data_Processing.ipynb` — notebook that renames, checks, measures and
  augments the raw green-apple images (runs on Google Drive data).
- `ai-model/` — the Sem 6 training code: model (`backbone.py`, `heads.py`, `model.py`),
  data loading (`dataset.py`, `config.py`), losses (`losses.py`), training and evaluation
  (`train.py`, `evaluate.py`, `evaluate_yield.py`), export and quantization
  (`export.py`, `export2.py`, `convert_ort.py`, `quantize.py`), plus design notes
  (`architecture.md`, `README.md`).
- `ai-model/data_prep/` — the eight dataset preparation scripts with their run logs.
- Needs from Drive to run: `final_dataset/` and the DINOv3 backbone weights
  (see `DATA_SETUP.md`). The raw `data/` folder is not needed — every image it holds
  is already contained in `final_dataset/`.

### `sem7-proposed-model/` — Semester 7 final model (canonical version)
The final proposed student model that the app and the web service use.
- The ten training files (`student_backbone.py`, `student_config.py`,
  `student_dataset.py`, `student_loss.py`, `student_model.py`, `student_train.py`,
  `run_student.py`, `verify_backbone.py`, `convert_fruit_annotations.py`, `test.py`).
- `outputs/` — the four teacher soft-label tables used for training.
- `student/` — training history, evaluation summary and the four result charts.
- Needs from Drive to run or serve: `adc_student_full.onnx`,
  `checkpoints/best_model.pth`, the `data/` images and the DINOv3 weights.

### `sem7-ablation-study/` — ablation experiments (renamed BTechProject)
The experiment track: an earlier model variant plus every ablation run.
- The twelve root Python files and `run_ablations.sh` (same roles as above, older variant).
- `scripts/` — result aggregation and the ablation runner.
- `old/` — the legacy teacher-training code it grew out of.
- `not_used/` — the held-out Blotch/Scab image pairs.
- `evidence/` — small result files only (training logs, summaries, dataset configs,
  evaluation curves) from all `outputs/` and `runs/`.
- `ablation-configs/` — per-configuration training histories (A: no distillation,
  B: static weights, C: no LoRA, D: no warmup, plus frozen-backbone).
- Needs from Drive to retrain: the ablation checkpoints in `01-Weights/ablation/`
  and the `02-Datasets/sem7-data/` images.

### `teachers/` — teacher models
The pretrained teachers the student learns from: PDDD fruit/leaf/pest classifiers and
three YOLO apple-detection runs. Code and small config files only; training images and
teacher weight files live on Drive.

### `android-app/` — Android application
The Kotlin phone app source (camera, on-device inference, results UI, benchmark
harness) with build scripts. Includes the small benchmark photo sets with their
manifests. Needs from Drive: `adc_student_full.onnx` placed in
`app/src/main/assets/`.

### `web/` — web API and deployment
FastAPI server (serves the model and the UI from one port), React UI, Dockerfile,
Kubernetes manifests, Prometheus/Grafana monitoring configs, Ansible playbooks and
setup guides. Includes lab screenshots as evidence. Needs from Drive:
`adc_student_full.onnx` placed in `backend/models/`, then `pip install` / `npm install`
generate the remaining runtime folders.

### `docs/` — college documents
Ontology document, Patent Annexure II, phone screenshots and demo-link list
(`DEMO_LINKS.md`; videos stay on Drive).

### `archive/` — early material, kept for record
The first prototype (`dataset.py`, `model.py`, `train.py`) and the hackathon fork's
notes. The fork's code was identical to the final model, so only its documents are kept.

## How to run

```bash
git clone <YOUR-PRIVATE-REPO-URL> AgriCNXEdge
cd AgriCNXEdge
# fetch the Drive folders listed in DATA_SETUP.md first (models need 01-Weights,
# training needs 02-Datasets, videos are in 04-Videos-Demo)

# Web API + UI
cd web
python -m venv .venv && .\.venv\Scripts\activate
pip install -r backend/requirements.txt
pytest backend/tests -q
cd backend && uvicorn app.main:app --reload --port 8000
cd ../frontend && npm install && npm run dev

# Final model (needs Drive weights + data)
pip install -r requirements-ml.txt
python sem7-proposed-model/verify_backbone.py
python sem7-proposed-model/run_student.py --help
```

## How to submit

The repository is already a git repo (branch `main`). To publish it as a private
submission repo and link the Drive:

```bash
cd E:\Projects\AgriCNXEdge
gh repo create AgriCNXEdge --private --source=. --remote=origin --push
```

Then add the Drive link (`AgriCNXEdge-Drive`, URL at the top) to the repo's About
section. Open issues: see `KNOWN-GAPS.md`.
