#!/usr/bin/env python3
"""
Aggregate all `yield_teacher_summary.json` files under `outputs/` and compute
mean ± std for requested metrics grouped by experiment prefix (A/B/C/D).

Saves `outputs/yield/ablation_aggregate.json` and prints a CSV summary.
"""
from pathlib import Path
import json
import statistics
import sys

OUT = Path("outputs")
PATTERN = "**/yield_teacher_summary.json"

metrics_names = ["mAP50", "mAP50_95", "precision", "recall"]
small_key = ("per_size", "per_size_ap", "small")

rows = {}

for p in OUT.glob(PATTERN):
    try:
        data = json.loads(p.read_text())
    except Exception:
        continue
    # experiment dir name used to infer label (we used --exp-name prefix in runner)
    exp_dir = Path(p).parent
    label = exp_dir.name.split("_")[0]
    rows.setdefault(label, []).append((p, data))

report = {}
for label, items in sorted(rows.items()):
    vals = {k: [] for k in metrics_names}
    vals["small_mAP"] = []

    for p, d in items:
        tm = d.get("test_metrics") or {}
        for k in metrics_names:
            v = tm.get(k)
            if v is None:
                # try older naming
                v = tm.get(k.lower())
            vals[k].append(float(v) if v is not None else 0.0)
        # small object mAP
        try:
            s = d.get("per_size", {}).get("per_size_ap", {}).get("small", 0.0)
            vals["small_mAP"].append(float(s))
        except Exception:
            vals["small_mAP"].append(0.0)

    summary = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            mean = 0.0
            stdev = 0.0
        elif len(arr) == 1:
            mean = arr[0]
            stdev = 0.0
        else:
            mean = statistics.mean(arr)
            stdev = statistics.stdev(arr)
        summary[k] = {"mean": mean, "std": stdev, "n": len(arr)}

    report[label] = summary

# Save and print
out_file = Path("outputs/yield/ablation_aggregate.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(report, indent=2))

print("label," + ",".join([f"{m}_mean,{m}_std" for m in metrics_names]) + ",small_mAP_mean,small_mAP_std,n")
for label, s in sorted(report.items()):
    row = [label]
    for m in metrics_names:
        row.append(f"{s[m]['mean']:.4f}")
        row.append(f"{s[m]['std']:.4f}")
    row.append(f"{s['small_mAP']['mean']:.4f}")
    row.append(f"{s['small_mAP']['std']:.4f}")
    row.append(str(s[metrics_names[0]]['n']))
    print(",".join(row))

print(f"\nSaved aggregate report to: {out_file}")
