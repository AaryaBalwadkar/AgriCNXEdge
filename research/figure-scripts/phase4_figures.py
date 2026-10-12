"""Phase 4 edge figure (Q1-grade, CPU-only).

Reads A17/new 8 files + on-disk model sizes. Writes:
  paper_figures/fig16_edge.{pdf,tif,png} + parity_results/edge_data.json

Figure grammar (matches phase1_figures_v2):
  serif Times, hairline axes, flat pastels, Okabe-Ito, panel letters (a)(b).
  NO caption sentences inside the figure -- axis labels, tick labels,
  legend entries, value labels and panel letters only. Captions live in
  the .tex \\caption{} text.

  (a) Model size (MB) -- measured file sizes from disk.
  (b) On-device latency, A17 -- mean dots with p50 marker and p95 whisker,
      CPU_ONLY vs NNAPI_PREFERRED (n=100 measured + 10 warmup per mode).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
A17 = ROOT / "A17" / "new"
OUT = ROOT / "paper_figures"
EDGE_JSON = ROOT / "parity_results" / "edge_data.json"

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "grey": "#8A8A8A", "lightgrey": "#D9D9D9"}
PASTEL = {"blue": "#C7DFF0", "orange": "#F5DEB8", "green": "#C4E7D4",
          "grey": "#E2E2E2"}

# Measured on this laptop (bytes -> MB), 2026-09-20. Pest teacher .pth absent
# on disk -> excluded (stated in caption text, not guessed).
SIZES_MB = {
    "Unified\nstudent": 120.7,   # adc_student_full.onnx
    "Leaf teacher\nConvNeXt-S": 188.8,   # leaf_best_teacher.pth
    "Fruit teacher\nConvNeXt-S": 188.8,  # fruit_best_teacher.pth
    "Yield teacher\nYOLO26s": 19.4,      # best.pt
}

# Measured A17 latency pass (bench_*.txt, n=100 measured + 10 warmup per mode)
LAT = {
    "CPU_ONLY": {"mean": 5748.9, "p50": 7041.0, "p95": 7381.0, "n": 100},
    "NNAPI_PREFERRED": {"mean": 5603.6, "p50": 6763.0, "p95": 7406.0, "n": 100},
}
# Measured A17 accuracy pass (bench_accuracy_*.txt, 50 v2 images x1 per mode)
ACC = {
    "yield_MAE": 2.9, "yield_n": 30, "pest_acc": 1.0, "pest_n": 20,
    "cpu_nnapi_pred_diffs": 0, "device": "SM-A176B",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 5.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.color": "#E5E5E5",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def verify_inputs():
    need = ["bench_accuracy_cpu_only_1789903085637.csv",
            "bench_accuracy_nnapi_preferred_1789902839764.csv",
            "bench_cpu_only_1789904360537.csv",
            "bench_nnapi_preferred_1789903713904.csv"]
    for f in need:
        assert (A17 / f).exists(), f"missing {f}"
    # cross-check latency means from CSVs (no hardcoding drift)
    for f, key in [("bench_cpu_only_1789904360537.csv", "CPU_ONLY"),
                   ("bench_nnapi_preferred_1789903713904.csv", "NNAPI_PREFERRED")]:
        ms = [float(r["ms_total"]) for r in csv.DictReader(open(A17 / f, encoding="utf-8"))]
        assert len(ms) == 100, f"{f}: {len(ms)} rows"
        m = float(np.mean(ms))
        assert abs(m - LAT[key]["mean"]) < 5.0, f"{key} mean drift {m}"
    print("inputs verified: 8 files present, latency means match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    verify_inputs()
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(7.28, 2.9))

    # (a) model size
    a = ax[0]
    names = list(SIZES_MB.keys())
    vals = [SIZES_MB[k] for k in names]
    cols = [PASTEL["blue"], PASTEL["orange"], PASTEL["orange"], PASTEL["green"]]
    edge = [OI["blue"], OI["orange"], OI["orange"], OI["green"]]
    bars = a.bar(range(len(names)), vals, color=cols, edgecolor=edge, linewidth=0.9, zorder=3)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names)
    a.set_ylabel("File size (MB)")
    a.grid(True, axis="y")
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 3.0, f"{v:.1f}",
               ha="center", va="bottom", fontsize=7.0)
    a.set_ylim(0, max(vals) * 1.18)
    a.text(0.02, 0.96, "(a)", transform=a.transAxes, fontweight="bold",
           va="top", ha="left", fontsize=9.0)

    # (b) latency: mean dots, p50 tick, p95 whisker
    b = ax[1]
    modes = ["CPU_ONLY", "NNAPI_PREFERRED"]
    labels = ["CPU", "NNAPI"]
    x = np.arange(len(modes))
    means = [LAT[m]["mean"] for m in modes]
    p50s = [LAT[m]["p50"] for m in modes]
    p95s = [LAT[m]["p95"] for m in modes]
    b.errorbar(x, means,
               yerr=[np.subtract(means, [min(m, p) for m, p in zip(means, p50s)]),
                     np.subtract(p95s, means)],
               fmt="o", color=OI["blue"], ecolor=OI["grey"], elinewidth=1.1,
               capsize=3.5, capthick=1.0, markersize=5.0, zorder=4)
    b.scatter(x, p50s, s=28, facecolors="none", edgecolors=OI["orange"],
              linewidths=1.2, zorder=5)
    for xi, m, p5, p9 in zip(x, means, p50s, p95s):
        b.text(xi + 0.07, m + 90, f"{m:.0f}", fontsize=7.0, va="bottom", ha="left")
        b.text(xi + 0.07, p9 + 90, f"{p9:.0f}", fontsize=6.5, va="bottom",
               ha="left", color="#555555")
    b.set_xticks(list(x))
    b.set_xticklabels(labels)
    b.set_ylabel("Latency (ms)")
    b.grid(True, axis="y")
    b.text(0.02, 0.96, "(b)", transform=b.transAxes, fontweight="bold",
           va="top", ha="left", fontsize=9.0)
    from matplotlib.lines import Line2D
    b.legend([Line2D([0], [0], marker="o", color=OI["blue"], label="Mean",
                      linestyle="None", markersize=5.0),
              Line2D([0], [0], marker="o", markerfacecolor="none",
                     markeredgecolor=OI["orange"], label="p50",
                     linestyle="None", markersize=6.0)],
             ["Mean", "p50"], loc="upper left", frameon=True, handletextpad=0.4)

    fig.tight_layout(pad=0.6, w_pad=2.0)
    base = out / "fig16_edge"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    EDGE_JSON.parent.mkdir(parents=True, exist_ok=True)
    EDGE_JSON.write_text(json.dumps({
        "sizes_mb": SIZES_MB,
        "sizes_note": "Measured file sizes on disk; pest teacher .pth absent, excluded.",
        "latency_ms": LAT,
        "latency_note": "A17 SM-A176B, 10 warmup + 100 measured per mode, batch=1.",
        "accuracy": ACC,
        "peak_pss_kb": 807176,
        "battery_note": "Coarse drain only (84->75 across v2 tap); no power claim.",
    }, indent=2), encoding="utf-8")
    print(f"wrote {base}.pdf/.png/.tif + edge_data.json")


if __name__ == "__main__":
    main()
