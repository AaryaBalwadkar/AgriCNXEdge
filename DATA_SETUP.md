# Large files (Drive setup)

The trained models, datasets and videos (about 18 GB) cannot live in this git repo —
GitHub rejects any single file over 100 MB. They live in the shared Drive folder:

**AgriCNXEdge-Drive:** https://drive.google.com/drive/folders/1ZAEEz724iC9ZOZ7GiSSvtMCLUDnvcTqx?usp=sharing

Download what you need, place each file where the tables below say, and run normally.
Never commit these files — `.gitignore` already excludes them, so `git status` stays clean.

## Drive contents

### `01-Weights/` — all model and checkpoint files
| File | Size | Where to place it in this repo |
|---|---|---|
| `adc_student_full.onnx` | 126,586,742 bytes | `sem7-proposed-model/student/`, copy to `web/backend/models/` and `android-app/app/src/main/assets/` |
| `adc_student.onnx` | 122,199,706 bytes | `sem7-proposed-model/student/` |
| `best_model.pth` | 155,668,669 bytes | `sem7-proposed-model/student/checkpoints/` |
| `latest_checkpoint.pth` | — | same folder as above |
| `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` | about 111 MB | `sem6-pbl-agroscan/ai-model/weights/` and `sem7-*/weights/` |
| `convnext_small-model-99.pth` | 198,305,809 bytes | `teachers/` reference only |
| `yolo26s.pt` | 20,422,725 bytes | local use with `run_yield_teacher.py` |
| `yolo26n.pt` | 5,544,453 bytes | local use with `run_yield_teacher.py` |
| `agroscan_best_model.pth` | 123,781,094 bytes | `sem6-pbl-agroscan/ai-model/checkpoints/` |
| `agroscan_model_fp32.onnx` | 123,637,667 bytes | `sem6-pbl-agroscan/ai-model/` |

`01-Weights/ablation/` — one file per ablation configuration (renamed so names differ):
`ablation_B_best.pth` (155,665,329), `ablation_B_latest.pth` (155,669,627),
`ablation_C_latest.pth` (153,808,987), `ablation_D_best.pth` (155,668,669),
`frozen_latest.pth` (153,808,859), `ablation_A_student.onnx` (121,061,281).
Place under `sem7-ablation-study/` when retraining or comparing.

`01-Weights/teachers/` — `fruit_teacher_s123.pth` and `fruit_teacher_s42.pth`
(197,969,881 each, different files), `leaf_teacher_s456.pth` (197,972,667).

### `02-Datasets/` — training data (each image stored once)
| Folder | Contents | Where to place it |
|---|---|---|
| `final_dataset/` | `images/` (13,581), `labels/`, `splits/`, `teacher_outputs/`, `meta/` — about 4.6 GB | `sem6-pbl-agroscan/ai-model/final_dataset/` |
| `sem7-variants/` | `yield_enhanced/`, `yield_mixed/` (ablation preprocessing variants) | `sem7-ablation-study/data/` |
| `yolo/` | `images/`, `labels/`, `data.yaml` (1,512-image detection set) | `teachers/YOLO26s-apple-1/data/yield/` |

Note: the old raw `data/` folder (375 MB) is fully contained in `final_dataset/` and is
not uploaded. The Sem 7 `fruit/leaf/pest/` folders likewise duplicate it and are skipped.

### `03-YOLO-Teacher-Weights/`
`yield_teacher_best.pt` and `yield_teacher_last.pt` per batch (A/B/C/D), plus the four
`batch_summary.json` files. Local use only.

### `04-Videos-Demo/`
`BTechDemo.mp4`, `az_recorder_20260921_194127.mp4`, `az_recorder_20260921_194423.mp4`.
Linked from `docs/DEMO_LINKS.md`; never committed.

### `05-Paper-Supplementary/` — removed from this submission
Paper materials (TIFFs, overlays, drafts) are out of scope for this code-only repo.
They remain in the old local folders; nothing is stored here.

## Generated locally (not on Drive, not in git)

| What | How |
|---|---|
| `web/backend/.venv/`, `web/frontend/node_modules/`, `web/frontend/dist/` | `python -m venv .venv` + `pip install -r backend/requirements.txt`; `npm install` + `npm run build` |
| `web/.env` | copy `web/.env.example` (5 settings: name, environment, model path, allowed UI origin, max upload MB) |
| Android `build/`, `.gradle/`, `local.properties` | created by the first Gradle build / Android Studio |
