# Sem 7 — Proposed Model (canonical)

Source: `BTechProjectmodels/Proposed Model/` (SHA256 `student_model.py` = `98AF…`, differs from ablation `0B44…`).
Kept: 10 `student_*.py` + `run_student.py` + `verify_backbone.py` + `convert_fruit_annotations.py` + `test.py`,
`outputs/*.csv` (4 soft-label files), `student/*.json` + `q1_*.pdf`.
Excluded: `student/*.onnx` (122–126 MB), `student/checkpoints/best_model.pth` — see `../DATA_SETUP.md` §A.
Run: `pip install -r ../requirements-ml.txt` then `python verify_backbone.py`, `python run_student.py --help`.
Eval/figs: `../research/figure-scripts/phase2_eval.py`, `phase4_*.py`.
