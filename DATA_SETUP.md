# DATA_SETUP — Large files NOT in git (Ignore + Drive-link policy)

College asked for "each and everything", but GitHub rejects files >100 MB and repos >5 GB.
This repo keeps **code + small evidence** (~50 MB). Everything below stays on your Drive;
table gives **old source path → new local target path** so you can add them manually
for local runs without committing them.

## Verified Drive inventory (checked online 2026-10-12 via public embed view)

- Drive 1 `BTechProject` — https://drive.google.com/drive/folders/1VIScZ5jnB5VhGMnVm5VQ7MXNPC1VJz2t?usp=sharing
  Root: `__pycache__/ models/ not_used/ old/ outputs/ runs/ scripts/ venv/` + 12 root `.py` +
  `run_ablations.sh`, `yolo26n.pt`, `yolo26s.pt`. `models/` = Ablation, Ablation Frozen Backbone,
  PDDD Fruit 1, PDDD Fruit 3 and PDDD Leaf 1, PDDD Pest 1, phase2_results, Proposed Model,
  YOLO26s ×3, `phase2_eval.py`. `outputs/` = Ablation, Ablation Frozen Backbone, fruit, leaf,
  pest, Proposed Model, student, try1, yield.
- Drive 1 → Proposed Model — `data/{fruit,leaf,pest,yield,yield_enhanced,yield_mixed}/`,
  `weights/{dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth, convnext_small-model-99.pth}`,
  `student/{adc_student.onnx, q1_*.pdf, training_history.json, yield_eval.json}`,
  `student/checkpoints/{latest_checkpoint.pth}`. `outputs/student/` = same set + `visual_eval/`.
- Drive 2 `data processing` — https://drive.google.com/drive/folders/1ZtoJ5zgvGNBjk-hyRmXoq7zuIMcQFILX?usp=sharing
  Only `data_prep/` + `final_dataset/{images,labels,meta,splits,teacher_outputs}/`.
- Drive 3 `AgroScan_YOLO_Dataset` — https://drive.google.com/drive/folders/1pvbIe-2JHZnBBbQ7UGk_tXkEXw6H96WJ?usp=drive_link
  Only `images/`, `labels/`, `data.yaml`.
- NOT found on any Drive (ask teammates — see bottom): `adc_student_full.onnx` (126,586,742 bytes),
  `best_model.pth` (Proposed copy = 155,668,669 bytes; only `latest_checkpoint.pth` is on Drive),
  any `*.mp4` video, Agroscan `checkpoints/*.pth`, `models/dinov3/*.pth`, `*.onnx/*.ort`.
- Listing check only: names/dates verified, file *contents* not verified. If a file is
  "distorted", compare its Drive-shown size against the byte counts in §A–B, or compare
  SHA256 (`Get-FileHash`) with the local copies in `E:\Projects\`.

## A. Model weights (never commit; gitignored)

| File (old location) | Size | Place manually at (new repo) | Used by |
|---|---|---|---|
| `BTechProjectmodels/Proposed Model/student/adc_student_full.onnx` | 126.6 MB | `sem7-proposed-model/student/adc_student_full.onnx` AND `web/backend/models/adc_student_full.onnx` | `web/backend/app/services/inference.py`, `research/figure-scripts/phase4_*.py`, Android `assets/` |
| `BTechProjectmodels/Proposed Model/student/adc_student.onnx` | 122.2 MB | `sem7-proposed-model/student/adc_student.onnx` | `phase4_export.py` input |
| `BTechProjectmodels/Proposed Model/student/checkpoints/best_model.pth` | ~118 MB | `sem7-proposed-model/student/checkpoints/best_model.pth` | `run_student.py --resume`, `phase4_export.py` |
| `Agroscan/newmodel/checkpoints/best_model.pth` | 118 MB | `sem6-pbl-agroscan/ai-model/checkpoints/best_model.pth` | sem6 `train.py` resume |
| `Agroscan/newmodel/models/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` (also dup at `Agroscan/newmodel/*.pth`) | 107 MB | `sem6-pbl-agroscan/ai-model/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` AND sem7 `weights/` (see MISSING_FILES.md §A: config path is wrong, must match) | `backbone.py`, `student_backbone.py`, `verify_backbone.py` |
| `Agroscan/newmodel/model_fp32.onnx` + `model_raw.onnx.data` (117 MB) | ~118 MB | Drive only (export artifact, superseded by `adc_student_full.onnx`) | — |
| `BTechProject/yolo26s.pt` (20.4 MB), `yolo26n.pt` (5.5 MB) | — | `sem7-ablation-study/evidence/` (local only) or `~/.cache/ultralytics` | `run_yield_teacher.py`, teachers |
| `BTechProject/outputs/yield/*/weights/{best.pt,last.pt}` + `BTechProjectmodels/YOLO*/outputs/**/best.pt` | various | Drive only | evidence already has `results.csv/args.yaml` |

## B. Datasets (never commit; gitignored)

| Dataset (old location) | Size | Place manually at | Used by |
|---|---|---|---|
| `Agroscan/newmodel/final_dataset/` (images 13,581, 4.6 GB + labels + splits + teacher_outputs) | 4.6 GB | `sem6-pbl-agroscan/ai-model/final_dataset/` (same layout: `images/ labels/ splits/ teacher_outputs/ meta/`) | sem6 `config.py`, `dataset.py`, `train.py` |
| `Agroscan/newmodel/data/` (fruit 3,192 + leaf 10,401 + pest 980 files, 375 MB) | 375 MB | `sem6-pbl-agroscan/ai-model/data/` | `data_prep/*.py`, notebook |
| `BTechProject` has NO `data/` / `final_dataset/` (referenced by `student_config.py` but absent) — stage from Drive or from `BTechProjectmodels/Proposed Model` soft-label CSVs + YOLO data below | — | `sem7-ablation-study/data/` + `sem7-proposed-model/data/` | `student_dataset.py` |
| `BTechProjectmodels/YOLO26s apple object detection model 1/data/yield/` (train 1,347 + val 165 images+labels) | ~500 MB | `teachers/YOLO26s-apple-1/data/yield/` | `run_yield_teacher.py`, `phase2_eval.py --stage-csvs` |
| `BTechProject/models-20261012T041829Z-1-001/models/` (richest backup: Ablation + PDDD + YOLO + phase2_results + `phase2_eval.py`) | 2.17 GB unpacked | Drive only (do NOT unpack into repo) | fallback if anything above is lost |
| `BTechProject/models-*.zip` (5 files, 9.6 GB) + `E:\Projects\models-20260917T142925Z-1-001.zip` (2.1 GB) | ~12 GB | Drive only | backups |
| Green Apple raw (Drive `/Green_Apple/dataset/raw`, 3,111 imgs) + `split/` + `yolo_clean_v3/` (notebook outputs) | — | Drive only | `sem6-pbl-agroscan/Green_Apple_Data_Processing.ipynb` |

## C. App / web runtime (never commit; gitignored)

| File | Place manually at |
|---|---|
| `BTechProjectmodels/AgriCNXEdge/app/src/main/assets/adc_student_full.onnx` (126 MB) | `android-app/app/src/main/assets/adc_student_full.onnx` |
| `.../assets/bench/` + `bench_v2/` (+ `manifest_v2.csv`) | `android-app/app/src/main/assets/bench[_v2]/` (needed by `phase4_green_overlays.py`, `phase4_parity_v2.py`) |
| `web/backend/models/adc_student_full.onnx` (copy of above) | `web/backend/models/` (only `.gitkeep` is committed) |
| `web/frontend/node_modules/`, `web/frontend/dist/`, `web/backend/.venv/` | generated by `npm install` / `npm run build` / `python -m venv` |
| `web/.env` | copy `web/.env.example` (5 keys: APP_NAME, APP_ENV, MODEL_PATH, CORS_ORIGINS, MAX_UPLOAD_MB) |

## D. Media / submission PDFs (in repo except videos)

- IN repo: `research/papers/*.pdf`, `research/journal-submission/main.pdf+figs`, `research/paper_figures/*.pdf,*.png`, `research/evaluation/**`, `docs/*.png`, `sem7-*/student/*.pdf`, `evidence/**/*.png`.
- DRIVE ONLY: `BTechDemo.mp4`, `az_recorder_20260921_*.mp4`, `E:\Projects\POEVideo*.mp4`, `Video Project 3.mp4`. Add unlisted YouTube/Drive links to `docs/DEMO_LINKS.md` (create manually) and repo About.
- EXCLUDED bulk (Drive only): `paper_figures/*.tif`, `*_print.png`, `green_ov/*.png` (60 overlays), `v1/` originals, `phase2_results/*/overlays/`, `runs/detect/val_batch*.jpg`, `journal-submission.zip` (3.5 MB dup of folder — not copied).

## E. What to upload to Drive (checklist before GitHub push)

1. Zip per row in §A–B (keep folder names identical).
2. Paste share links into: repo README (About), `docs/DEMO_LINKS.md`, `journal-submission/main.tex` Zenodo DOI placeholder (see MISSING_FILES.md §D).
3. Verify `.gitignore` blocks them: `git status --porcelain` must show none of the above as untracked.
