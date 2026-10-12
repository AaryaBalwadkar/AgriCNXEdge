#!/usr/bin/env bash
set -euo pipefail

# Runner for reproducible ablation: A (baseline), B (enhanced), C (mixed), D (on-the-fly)
# Uses `run_yield_teacher.py --auto` to ensure each group is saved under a distinct
# prefix (A/B/C/D). Run and leave for 24h.

SEEDS="42,7,123"
ENH="lab_edge"
ON_THE_FLY_COPIES=1
SKIP_VISUALS="--skip-visuals"
CLAHE_COLOR=3.5
CLAHE_LIGHT=1.5

echo "[RUNNER] Starting ablation batch at $(date)"

# B: Enhanced only — saved under prefix 'B'
python run_yield_teacher.py --auto \
  --auto-seeds "${SEEDS}" \
  --auto-modes "enhanced" \
  --auto-enhancements "${ENH}" \
  --auto-clahe-color "${CLAHE_COLOR}" \
  --auto-clahe-light "${CLAHE_LIGHT}" \
  --exp-name B ${SKIP_VISUALS}

# C: Mixed (original + enhanced) — saved under prefix 'C'
python run_yield_teacher.py --auto \
  --auto-seeds "${SEEDS}" \
  --auto-modes "mixed" \
  --auto-enhancements "${ENH}" \
  --auto-clahe-color "${CLAHE_COLOR}" \
  --auto-clahe-light "${CLAHE_LIGHT}" \
  --exp-name C ${SKIP_VISUALS}

# D: On-the-fly augmentation (creates disk-backed randomized copies) — saved under prefix 'D'
python run_yield_teacher.py --auto --on-the-fly --on-the-fly-copies ${ON_THE_FLY_COPIES} \
  --auto-seeds "${SEEDS}" \
  --auto-modes "enhanced" \
  --auto-enhancements "${ENH}" \
  --auto-clahe-color "${CLAHE_COLOR}" \
  --auto-clahe-light "${CLAHE_LIGHT}" \
  --exp-name D ${SKIP_VISUALS}

# A: Baseline (original) — saved under prefix 'A'
python run_yield_teacher.py --auto \
  --auto-seeds "${SEEDS}" \
  --auto-modes "original" \
  --auto-enhancements "${ENH}" \
  --auto-clahe-color "${CLAHE_COLOR}" \
  --auto-clahe-light "${CLAHE_LIGHT}" \
  --exp-name A ${SKIP_VISUALS}

echo "[RUNNER] All experiments submitted at $(date)"
