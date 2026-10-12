# MISSING_FILES — everything absent/broken and where to add it manually

This is the planning-phase register you asked for: each missing file, **where it goes in
THIS new repo**, and how to fix it. §A = must-author, §B = excluded-by-policy (see
DATA_SETUP.md for sources), §C = code bugs to fix, §D = placeholders to fill.

## §A. Files that never existed — author them

| # | Missing file (new path) | Why | How to add manually |
|---|---|---|---|
| A1 | `requirements-ml.txt` (repo root) | No root ML requirements anywhere. Only `teachers/PDDD-Fruit-1/teacher_requirements.txt` (torch/timm/albumentations/sklearn/imagehash/Pillow/tqdm) and `web/backend/requirements.txt` exist | Create from: `torch torchvision timm albumentations opencv-python-headless scikit-learn matplotlib tqdm onnx onnxruntime ultralytics pillow numpy`. Pin after `pip freeze` on your training machine |
| A2 | `archive/early-prototype/config.py` + `archive/early-prototype/compress_export.py` | `archive/early-prototype/train.py:18,21` imports `ProjectConfig` and `prune_heads_unstructured, recovery_finetune_hook` — both files absent in `E:\Projects\` | Either (recommended) leave prototype archived as-is with this note, or copy `sem7-proposed-model/student_config.py` → `config.py` and implement `compress_export.py` stubs |
| A3 | `teachers/PDDD-Pest-1/teacher_train.py` + `run_pest_teacher.py` | `PDDD Pest 1/` has only 3 `.py` vs siblings' 7–8; pipeline incomplete | Copy from `teachers/PDDD-Fruit-1/teacher_train.py` + `run_pest_teacher.py` (in `old/` launcher) and adapt class map (6 pest classes) |
| A4 | `web/scripts/gcp-trackb.bat` | `web/README.md` Quickstart D + Troubleshooting E1/E3 reference `scripts\gcp-trackb.bat create/delete`, but only `.ps1` + `.sh` exist | Create CMD wrapper calling `powershell -File scripts\gcp-trackb.ps1 %*`, or edit `web/README.md` + `web/docs/*` to remove `.bat` references |
| A5 | `docs/DEMO_LINKS.md` | Videos excluded from git (`BTechDemo.mp4`, `az_recorder_*.mp4`, POE/Video Project 3) | Create with Drive/YouTube links + `MobileScreenshot1/2.png` (already in `docs/`) |
| A6 | `sem6-pbl-agroscan/ai-model/saved_model_dir/` + `*.tflite` | `quantize.py` needs `saved_model_dir` (from `onnx2tf`, not present) | Run export → `onnx2tf` → `quantize.py` with 150 calib images, or document as future work |
| A7 | `sem7-proposed-model/data/` + `weights/` + `sem7-ablation-study/data/` | Referenced by `student_config.py` (`PathConfig`) but absent in both `BTechProject/` and `Proposed Model/` | Stage per DATA_SETUP.md §B, or run `research/figure-scripts/phase2_eval.py --stage-csvs` (auto-stages 4 `*_soft_labels.csv` from teachers/YOLO) |
| A8 | `sem6-pbl-agroscan/ai-model/dinov3/` vendor | Excluded deliberately (337 files + own `.git`, Meta upstream) | Do NOT vendor. `pip install` or `git submodule add https://github.com/facebookresearch/dinov3.git`, then set `dinov3_repo_dir` in `config.py` (see C1) |

## §B. Files excluded by policy — restore locally, do NOT commit

(All sources/targets in DATA_SETUP.md. Summary: `*.onnx/*.pt/*.pth/*.ort`, `final_dataset/` 4.6 GB,
`data/` 375 MB, `models-*.zip` 9.6+2.1 GB, `venv/` 2.2 GB, `bench/`, `*.mp4`, `*.tif`, `green_ov/` bulk,
`phase2_results/*/overlays/`, `runs/val_batch*.jpg`, `node_modules/`, `dist/`, `.venv/`, `build/`,
`.gradle/`, `local.properties`, `web/.env`, `journal-submission.zip` dup.)

## §C. Broken references — fix in code (file:line in NEW repo)

| # | Location | Bug | Fix |
|---|---|---|---|
| C1 | `sem6-pbl-agroscan/ai-model/config.py:77` | `dinov3_weights="./weights/..."` does not exist; real files were `Agroscan/newmodel/models/dinov3/*.pth` + root dup | Point to `weights/dinov3_convnext...pth` after DATA_SETUP.md §A, or override like `export.py`/`export2.py` do (`./models/dinov3/...`) |
| C2 | `sem6-pbl-agroscan/ai-model/dataset.py:50-61` | Only `leaf_soft.csv` + `yield_soft.csv` loaded; `fruit_soft`/`pest_soft` is `...Repeat...` placeholder → `t_fruit/t_pest` zeros, KD noise for 2/3 heads | Replicate leaf loader for fruit/pest; verify writer keys match (`generate_teacher_outputse.py` YOLO writer emits `xyxy/conf/cls`, reader expects `yolo`) |
| C3 | `sem6-pbl-agroscan/ai-model/heads.py:314-316` vs `architecture.md:47-50` | Code routes F4→all cls + F1→yield; docs say F1→Leaf/F2→Pest/F3→Fruit/F4→Yield; `LeafHead/PestHead/FruitHead` dead code | Decide one routing, update the other + `architecturediagramcode.txt:93-94` |
| C4 | `sem6-pbl-agroscan/ai-model/visualize.py:94` | `preds["raw_yield"]` key no longer emitted (now `yield_map{heatmap,wh,offset}`) → KeyError; hardcoded `/content/drive/...` paths | Update key + parametrize paths |
| C5 | `sem6-pbl-agroscan/ai-model/test_dinov3.py:4` | `sys.path.append('e:/Projects/newmodel')` — missing `Agroscan` segment | Change to repo-relative path |
| C6 | `sem6-pbl-agroscan/ai-model/train.py:217-224 vs 252-253` | Resume expects `optimizer_state/scheduler_state`, saver writes only `epoch/model_state/best_val_loss` → resume crashes | Save/load optimizer+scheduler states |
| C7 | `sem6-pbl-agroscan/ai-model/generate_teacher_outputse.py` (typo `e`, Colab paths) | `/content/...` + `/content/drive/.../models/{fruit_model.pth,leaf_model.keras,insect_model,yolo26s.pt}` absent locally; pest+YOLO blocks commented out but `__main__` claims "All 4" | Rename to `generate_teacher_outputs.py` (keep alias), parametrize paths, uncomment/implement pest+YOLO or fix claim |
| C8 | `sem6-pbl-agroscan/Green_Apple_Data_Processing.ipynb` cells 5–7 | Cell 5 Drive remount error (`.../raw` not found → 60 640px imgs maybe un-normalized); cell 6 Blotch 24 vs 36 drift + excludes Black Pox/Powdery Mildew; cell 7 albumentations warnings (`cval/var_limit/fill_value` invalid) | Re-run with Drive mounted, reconcile counts, pin `albumentations` version |
| C9 | `sem6-pbl-agroscan/ai-model/config.py:19-25` | `LEAF_CLASS_MAP` 5 entries but comment says 7; `new.py` probes `Cedar_apple_rust` not in map; `data/leaf` has `Apple___healthy` (lowercase) vs map `Healthy` → `-1` masked | Unify class maps across `config.py`, `new.py`, `leaf_labels.csv`, `data/leaf/` |
| C10 | `sem7-ablation-study/evidence/outputs/yield/yield_dataset.yaml` + `*args.yaml` | Baked absolute Linux paths (`/home/CL407-33/Downloads/BTechProject-...`) invalid on Windows | Repoint `path:` to local `data/yield_*` and re-validate |
| C11 | `sem6-pbl-agroscan/ai-model/export.py` vs `export2.py` | `export.py` (opset14, logits+INT8, fragile dict-order) vs `export2.py` (opset18, probs+0.33 sharpen, FP32-only); committed `model_fp32.onnx` matches export2 semantics | Delete one path or document which ONNX is canonical; remove leftover `model_raw.onnx.data` / 0-byte `with_runtime_opt.ort` |
| C12 | `sem7-*/student_config.py` vs evidence YAMLs | Hyperparams live in `student_config.py`, not YAML; duplication risk | Keep `student_config.py` canonical, note in README |

## §D. Placeholders to fill manually

| # | Location | Placeholder |
|---|---|---|
| D1 | `research/journal-submission/main.tex` (92 KB, SciRep) | Cover-letter + checklist `TO-BE-INSERTED`: collection, reviewers, Zenodo DOI, **GitHub URL (paste after push)** |
| D2 | `web/ansible/inventory.ini`, `inventory.gcp.ini` | `YOUR_SERVER_IP`, `YOUR_GCE_EPHEMERAL_IP`, `ubuntu` user — fill per deploy |
| D3 | `web/.env.example` → `web/.env` | 5 keys (APP_NAME/APP_ENV/MODEL_PATH/CORS_ORIGINS/MAX_UPLOAD_MB) — copy, never commit |
| D4 | `web/.github/workflows/ci-cd.yml` | `vars.ENABLE_DEPLOY`, secrets `SERVER_HOST/USER/KEY/APP_DIR` — set in GitHub repo settings |
| D5 | `research/elsarticle/agri-cnx-edge.tex` vs `research/archive/elsarticle-template-num.tex` | `elsarticle/` copy is working paper, root copy was edited draft — pick one canonical before submission |
| D6 | Repo About + `docs/DEMO_LINKS.md` | Drive/YouTube links for datasets, weights, demo videos (see DATA_SETUP.md §E) |
