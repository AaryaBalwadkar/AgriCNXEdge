# Known gaps and fixes

Everything below is a real gap in this repo: a file to create, a code issue to fix, or
a value to fill in. Each item says where it goes and how to resolve it. Large model and
dataset files are not listed here — those are in `DATA_SETUP.md`.

## Files to create

| # | File (in this repo) | Why it is missing | How to resolve |
|---|---|---|---|
| A1 | `requirements-ml.txt` (root, exists but unpinned) | Written from the teacher/web requirement lists, but versions were never frozen | Run `pip freeze` on the training machine and pin the versions |
| A2 | `archive/early-prototype/config.py` + `compress_export.py` | The archived prototype's `train.py` imports them; they never existed | Leave the prototype archived as-is, or copy `sem7-proposed-model/student_config.py` as `config.py` and write the two small functions |
| A3 | `teachers/PDDD-Pest-1/teacher_train.py` + runner script | The pest teacher has only 3 of the usual 7–8 files | Copy from `teachers/PDDD-Fruit-1/` and adjust to the 6 pest classes |
| A5 | `docs/DEMO_LINKS.md` (exists, links pending) | Videos live on Drive, not in git | Add the `04-Videos-Demo/` share links once uploaded |
| A6 | Sem 6 `saved_model_dir/` + `.tflite` files | Phone quantization needs an intermediate export that was never run | Run export → `onnx2tf` → `quantize.py`, or record as future work |
| A7 | `sem7-proposed-model/data/`, `weights/`, `sem7-ablation-study/data/` | Referenced by the configs but kept on Drive | Place per `DATA_SETUP.md` |
| A8 | Sem 6 `dinov3/` code folder | Deliberately excluded (third-party upstream project) | Install it separately and set `dinov3_repo_dir` in `config.py` (see C1) |

Resolved: the missing Windows helper script (fixed by making `web/README.md`
PowerShell-only) and the deleted CI workflow (submission has no CI by decision).

## Code issues to fix

| # | File and line | Problem | Fix |
|---|---|---|---|
| C1 | `sem6-pbl-agroscan/ai-model/config.py:77` | Backbone weights path points to a folder that is not in the repo | Point it at `weights/` after placing the file per `DATA_SETUP.md` |
| C2 | `sem6-pbl-agroscan/ai-model/dataset.py:50-61` | Only the leaf and yield teacher tables are loaded; fruit/pest stay zero | Copy the leaf-loading block for fruit and pest; check the writer/reader key names match |
| C3 | `sem6-pbl-agroscan/ai-model/heads.py:314-316` vs `architecture.md` | Code and design note disagree on which network layer feeds which task | Pick one routing and update the other side plus the diagram text file |
| C4 | `sem6-pbl-agroscan/ai-model/visualize.py:94` | Uses an output name the model no longer returns; paths point at Colab | Update the output name and make paths arguments |
| C5 | `sem6-pbl-agroscan/ai-model/test_dinov3.py:4` | Hard-coded folder path is missing one level | Use a path relative to the repo |
| C6 | `sem6-pbl-agroscan/ai-model/train.py:217-224, 252-253` | Resume expects optimizer state that is never saved | Save and load the optimizer/scheduler state |
| C7 | `sem6-pbl-agroscan/ai-model/generate_teacher_outputse.py` (note typo) | Colab-only paths; pest/yield sections commented out but claimed done | Fix the filename, make paths arguments, finish or correct the claim |
| C8 | `sem6-pbl-agroscan/Green_Apple_Data_Processing.ipynb`, cells 5–7 | One cell failed on a remount, class counts drift, old library warnings | Re-run mounted, reconcile the Blotch counts, pin the library version |
| C9 | `sem6-pbl-agroscan/ai-model/config.py:19-25` | Class list (5) disagrees with its own comment (7); letter-case mismatch hides one class | Unify the class lists across config, probe script, labels and folders |
| C10 | `sem7-ablation-study/evidence/**/yield_dataset.yaml`, `args.yaml` | Absolute Linux paths from the training machine, invalid elsewhere | Repoint `path:` at the local data folder |
| C11 | `sem6-pbl-agroscan/ai-model/export.py` vs `export2.py` | Two export scripts with different settings; unclear which made the saved model | Keep one canonical path and document it; delete the 0-byte leftover |
| C12 | `student_config.py` vs evidence YAMLs | Settings live in two places | Treat `student_config.py` as canonical |

## Values to fill in

| # | File | Value |
|---|---|---|
| D2 | `web/ansible/inventory.ini`, `inventory.gcp.ini` | Server IP / user, only when deploying to a real machine |
| D3 | `web/.env` (local copy of `.env.example`) | 5 settings; never committed |
| D6 | Repo About page + `docs/DEMO_LINKS.md` | Drive links after upload finishes |
