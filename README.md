# AgriCNXEdge — Unified BTech Project (Sem 6 PBL → Sem 7 BTech Project)

Monorepo for college GitHub submission. One repo, full continuation history:
**Sem 6 PBL (Agroscan)** → **Sem 7 proposed multi-task edge model** → **ablation study** → **Android app** → **MLOps web service** → **research paper**.

> Large binaries (>100 MB) are intentionally NOT in git. See `DATA_SETUP.md` for where to download them and where to place them locally. See `MISSING_FILES.md` for every missing/broken file and its manual fix.

## Repo map

| Path | Source (old folder) | What it is |
|---|---|---|
| `web/` | `agricnxedge-web-v2` (73 files, git `main`) | FastAPI + ONNX Runtime + React/Vite + Docker + K8s + Prometheus/Grafana + Ansible. Canonical web copy (older `BTechProjectmodels/agricnxedge-web` discarded; CI workflow excluded — submission only, screenshots kept as evidence). |
| `sem6-pbl-agroscan/` | `Agroscan/` | Sem 6 PBL. `Green_Apple_Data_Processing.ipynb` + `ai-model/` (17 `.py`, `data_prep/`, `architecture.md`, `README.md`). Excludes `data/`, `final_dataset/` (5.6 GB), `dinov3/` vendor, weights. |
| `sem7-proposed-model/` | `BTechProjectmodels/Proposed Model/` | Canonical Sem 7 model (hash `98AF…`). 10 `.py` + `outputs/*.csv` + `student/*.json,*.pdf`. Excludes `*.onnx` (122–126 MB), `checkpoints/best_model.pth`. |
| `sem7-ablation-study/` | `BTechProject/` (renamed) + `BTechProjectmodels/Ablation/`, `Ablation Frozen Backbone/` | Ablation-variant code (hash `0B44…`, differs from proposed). 12 root `.py`, `scripts/`, `old/` legacy teachers, `not_used/` (Blotch/Scab pairs), `evidence/` (small json/csv/yaml/png only), `ablation-configs/`. Excludes venv, `models-*.zip` (9.6 GB), weights. |
| `teachers/` | `BTechProjectmodels/PDDD Fruit 1/`, `PDDD Fruit 3 and PDDD Leaf 1/`, `PDDD Pest 1/`, `YOLO26s apple 1/2/3/` | Teacher code + small meta only (`*.py,*.txt,*.yaml,*.json,*.csv`). Excludes `data/` images/labels bulk, `*.pt/weights`, val-batch jpgs. |
| `android-app/` | `BTechProjectmodels/AgriCNXEdge/` | Kotlin app source (`app/src/main/java`, `res`, gradle scripts). Excludes `build/`, `.gradle/`, `.idea/`, `.kotlin/`, `local.properties`, `assets/*.onnx`, `bench/` bulk. |
| `research/` | `BTechProjectmodels/` papers + tex + figures + scripts | `papers/` (4 ref PDFs), `journal-submission/main.tex`, `elsarticle/`, `paper_figures/` (pdf+png+tikz+captions; tif/print/green_ov bulk excluded), `figure-scripts/phase*.py` (12), `evaluation/` (parity + phase2 json/csv/pdf; overlays bulk excluded), `archive/` (IEEE template + early draft). |
| `docs/` | `BTechProjectmodels/*.docx,*.png` | Ontology, Patent Annexure II, mobile screenshots. Videos NOT in git (see DATA_SETUP.md). |
| `archive/` | `E:\Projects\dataset.py,model.py,train.py` + `bunniex-agricnxedge` meta | Early prototype (broken imports, see MISSING_FILES.md) + hackathon fork docs only (code was exact duplicate of Proposed Model, hash-verified, not re-copied). |

## What was deduplicated (repetitive files, no changes)

- `BTechProject/models/` (live, 409 files, 0.30 GB) == `BTechProject/models-20261012T041829Z-1-004/models/` → kept once (as evidence meta only, bulk excluded). Other `models-*.zip` + extracted dups (`034147Z-1-001`, `041829Z-1-001`, `041829Z-1-003`) NOT copied — richest backup is `041829Z-1-001`; see DATA_SETUP.md.
- `venv/` (2.18 GB) + `venv-*.zip` (1.48 GB) + empty `venv-*/` extract → none copied. Use `requirements-ml.txt` + `web/backend/requirements.txt`.
- `bunniex-agricnxedge/ai-model/*.py` == `Proposed Model/*.py` (10/10 SHA256 identical) → not re-copied; only `*.md` kept in `archive/bunniex-meta/`.
- `BTechProjectmodels/agricnxedge-web/` (older) superseded by `web/` (v2) → not copied.
- `conference_101719.tex` is untouched IEEE template → moved to `research/archive/`, not a paper.
- `New Ablation/` was empty → not copied.
- `model_fp32.with_runtime_opt.ort` (0 bytes, broken) → not copied.

## Quickstart

```bash
# 1. clone + setup data (see DATA_SETUP.md for Drive links)
git clone <YOUR-REPO-URL> AgriCNXEdge
cd AgriCNXEdge

# 2a. Web (needs backend/models/adc_student_full.onnx — see DATA_SETUP.md)
cd web
python -m venv .venv && .\.venv\Scripts\activate
pip install -r backend/requirements.txt
pytest backend/tests -q
uvicorn app.main:app --reload --port 8000   # from web/backend
cd frontend && npm install && npm run dev

# 2b. ML (proposed model, needs weights/ + data/ — see DATA_SETUP.md)
pip install -r requirements-ml.txt
python sem7-proposed-model/verify_backbone.py
python sem7-proposed-model/run_student.py --help

# 2c. Figures (CPU-only, no weights needed for most)
python research/figure-scripts/phase1_figures.py --help
python research/figure-scripts/phase2_eval.py --preflight-only
```

## GitHub push

```bash
cd E:\Projects\AgriCNXEdge
git init -b main
git add .
git commit -m "AgriCNXEdge unified: sem6 PBL + sem7 proposed + ablation + app + web + research"
gh repo create AgriCNXEdge --private --source=. --remote=origin --push
# college wants public? then --public. Add Drive link to repo About + README.
```

## Status / known gaps

- All items in `MISSING_FILES.md` §A–D are intentional gaps you fill manually (weights, datasets, `.env`, DOI/URLs, SSH IPs).
- `web/scripts/gcp-trackb.bat` does not exist upstream (only `.ps1`/`.sh`) — README in `web/` references it; either create it or edit `web/README.md`.
- Do NOT force-push `web/.git` history — this is a fresh unified repo. Old `agricnxedge-web-v2/.git` (branches `main`, `group10-ca2-submission`, remotes `origin/crop-health-mlops`, `fork-origin/DevOps-CA2_2023_27`) is preserved in the old folder.
