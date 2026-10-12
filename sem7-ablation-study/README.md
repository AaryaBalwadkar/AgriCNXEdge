# Semester 7 — ablation experiments (renamed BTechProject)

The experiment track: an earlier model variant plus every ablation run
(`BTechProject/`, `Ablation/` and `Ablation Frozen Backbone/` in the old folders).

- The twelve root Python files and `run_ablations.sh`.
- `scripts/` — result aggregation and the ablation runner.
- `old/` — the earlier teacher-training code this grew out of.
- `not_used/` — held-out Blotch/Scab image pairs.
- `evidence/` — small result files only (training logs, summaries, dataset configs,
  evaluation curves) from `outputs/` and `runs/`. Large weights and images are on Drive.
- `ablation-configs/` — per-configuration training histories (A: no distillation,
  B: static weights, C: no LoRA, D: no warmup, plus frozen-backbone).

Aggregate: `python scripts/aggregate_results.py`. Plot: `python plot_yield_ablation.py`.
Note: some dataset configs contain absolute paths from the training machine
(see `../KNOWN-GAPS.md`, C10). Checkpoints: `DATA_SETUP.md`, `01-Weights/ablation/`.
