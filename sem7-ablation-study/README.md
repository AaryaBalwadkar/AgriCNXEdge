# Sem 7 — Ablation Study (renamed BTechProject)

Source: `BTechProject/` (root code hash `0B44…` = ablation variant) + `BTechProjectmodels/Ablation/` +
`Ablation Frozen Backbone/` (configs only).
Kept: 12 root `.py`, `run_ablations.sh`, `scripts/` (aggregate + runner), `old/` (legacy teachers),
`not_used/` (Blotch/Scab pairs), `evidence/` (small `json/csv/yaml/png` from `outputs/` + `runs/` curves),
`ablation-configs/` (A/B/C/D + frozen training histories).
Excluded: `venv/`, `models-*.zip` (9.6 GB), extracted `models-*/` dups, `*.pt/*.pth/*.onnx`, val-batch jpgs,
absolute-path YAMLs (see `../MISSING_FILES.md` C10).
Aggregate: `python scripts/aggregate_results.py`; plot: `python plot_yield_ablation.py`.
