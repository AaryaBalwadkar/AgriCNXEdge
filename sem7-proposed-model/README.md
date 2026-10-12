# Semester 7 — final model (the version everything uses)

The final proposed student model (`Proposed Model` in the old folders). The paper, the
app and the web service all use this version.

- The ten training files (`student_backbone.py`, `student_config.py`,
  `student_dataset.py`, `student_loss.py`, `student_model.py`, `student_train.py`,
  `run_student.py`, `verify_backbone.py`, `convert_fruit_annotations.py`, `test.py`).
- `outputs/` — the four teacher soft-label tables used in training.
- `student/` — training history, evaluation summary and four result charts.

Needs from Drive (`DATA_SETUP.md`): `adc_student_full.onnx`,
`checkpoints/best_model.pth`, the `data/` images and the DINOv3 weights.
Run: `pip install -r ../requirements-ml.txt`, then `python verify_backbone.py`.
Figures: `../research/figure-scripts/phase2_eval.py`, `phase4_*.py`.
