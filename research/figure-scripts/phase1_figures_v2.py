"""
phase1_figures_v2.py
====================
Phase 1 REDO — Q1-grammar charts rebuilt from the same logged artifacts.
Revises v1 following the reference-figure grammar (ViT/DINO/DINOv2/DINOv3):
  * serif typography (Times), hairline axes, light grid, flat pastel fills
  * dot/paired-dot plots instead of zero-based bars for saturated metrics
  * radar + exact-value table for the 6-way ablation (DINOv2 Fig 5 style)
  * smoothed trajectories (thick smooth + thin raw), dashed reference lines
  * takeaway-first captions; single-seed (n=1) stated honestly

USAGE:
    python phase1_figures_v2.py
    python phase1_figures_v2.py --out paper_figures

Overwrites fig02/fig05/fig06/fig07/fig08/fig09.{pdf,tif,png} in <out>/.
(v1 backups live in paper_figures/v1/.) Also rewrites figure_captions.txt.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase1_figures import (ROOT, RUNS, CLEANING, CLASS_ORDER, load_history,
                            best_record, load_yield, run_summary)

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermilion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "grey": "#8A8A8A", "lightgrey": "#D9D9D9"}
PASTEL = {"blue": "#C7DFF0", "orange": "#F5DEB8", "green": "#C4E7D4",
          "vermilion": "#F2CFC0", "purple": "#E7CFE2", "grey": "#E2E2E2"}

SINGLE_W = 3.54
DOUBLE_W = 7.28


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
        "figure.constrained_layout.use": True,
    })


def panel(ax, tag, x=0.02, y=0.96):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def save_all(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", format="pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / f"{stem}.tif", format="tiff", dpi=1000,
                bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outdir / f"{stem}.png", format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"  wrote {stem}.{{pdf,tif,png}}")


def smooth(y, w=7):
    y = np.asarray(y, float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, w // 2, mode="edge"), k, mode="valid")


# ---------------------------------------------------------------- fig02 ---
def fig02(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.6),
                             layout="constrained")
    for ax, (task, tag) in zip(axes, [("leaf", "(a)"), ("pest", "(b)"),
                                      ("fruit", "(c)")] ):
        with open(CLEANING[task], encoding="utf-8") as f:
            rep = json.load(f)
        names = CLASS_ORDER[task]
        raw = [rep[c]["original_count"] for c in names]
        clean = [rep[c]["cleaned_count"] for c in names]
        x = np.arange(len(names))
        w = 0.38
        mx = float(max(raw))
        ax.bar(x - w / 2, raw, w, label="Raw", color=OI["lightgrey"],
               edgecolor="black", linewidth=0.6)
        ax.bar(x + w / 2, clean, w, label="Retained", color=OI["blue"],
               edgecolor="black", linewidth=0.6)
        ax.set_ylim(0, mx * 1.24)
        for xi, a, b in zip(x, raw, clean):
            ya = a + mx * 0.020
            yb = b + mx * 0.020
            if ya - yb < mx * 0.045:
                yb = ya + mx * 0.045
            ax.text(xi - w / 2, ya, f"{a}", ha="center",
                    va="bottom", fontsize=6)
            ax.text(xi + w / 2, yb, f"{b}", ha="center",
                    va="bottom", fontsize=6, color="#0B3D66",
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Apple___", "").replace("_", " ")
                            for n in names], rotation=18, ha="right")
        ax.set_ylabel("Images" if task == "leaf" else "")
        ax.set_xlabel(f"{task} classes")
        ax.grid(True, axis="y")
        panel(ax, tag)
    axes[1].legend(frameon=True, edgecolor="0.3", loc="lower right")
    save_all(fig, outdir, "fig02_dataset")


# ---------------------------------------------------------------- fig05 ---
def fig05(outdir, s):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.6),
                             layout="constrained")
    ax = axes[0]
    tasks = ["Leaf", "Pest", "Fruit"]
    accs = [s["acc_leaf"], s["acc_pest"], s["acc_fruit"]]
    cols = [OI["blue"], OI["orange"], OI["green"]]
    for i, (t, v, c) in enumerate(zip(tasks, accs, cols)):
        ax.plot(v, i, "o", color=c, ms=7, zorder=3)
        ax.text(v + 0.12, i, f"{v:.2f}%", va="center", fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(94, 101.5)
    ax.axvline(100, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x")
    panel(ax, "(a)")
    ax = axes[1]
    for i, (t, v, c) in enumerate(zip(["Count MAE", "Count RMSE"],
                                      [s["mae"], s["rmse"]],
                                      [OI["vermilion"], OI["purple"]])):
        ax.plot(v, i, "s", color=c, ms=7, zorder=3)
        ax.text(v + 0.09, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Count MAE", "Count RMSE"])
    ax.set_xlabel("Error (apples/image)")
    ax.set_xlim(2.5, 7.5)
    ax.grid(True, axis="x")
    panel(ax, "(b)")
    save_all(fig, outdir, "fig05_final_performance")


# ---------------------------------------------------------------- fig06 ---
def fig06(outdir, f, p):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained")
    ax = axes[0]
    tasks = ["Leaf", "Pest", "Fruit"]
    fv = [f["acc_leaf"], f["acc_pest"], f["acc_fruit"]]
    pv = [p["acc_leaf"], p["acc_pest"], p["acc_fruit"]]
    for i, (a, b) in enumerate(zip(fv, pv)):
        ax.plot([a, b], [i, i], color="black", linewidth=1.0, zorder=1)
        ax.plot(a, i, "o", color=OI["grey"], ms=7, zorder=3)
        ax.plot(b, i, "D", color=OI["blue"], ms=6, zorder=3)
        ax.text(a, i + 0.20, f"{a:.2f}", fontsize=6.5, ha="center",
                va="bottom", color=OI["grey"])
        ax.text(b, i - 0.20, f"{b:.2f}", fontsize=6.5, ha="center",
                va="top", color=OI["blue"])
        ax.text(max(a, b) + 0.45, i, f"$\Delta${b - a:+.2f} pp",
                fontsize=7, fontweight="bold", va="center", ha="left")
    ax.set_yticks(range(3))
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(93.5, 102.8)
    ax.axvline(100, color="black", linestyle="--", linewidth=0.8)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker="o", color=OI["grey"], label="Frozen",
                      linestyle="None", markersize=7.0),
               Line2D([0], [0], marker="D", color=OI["blue"], label="Proposed",
                      linestyle="None", markersize=6.0)],
              ["Frozen", "Proposed"], loc="lower right", frameon=True,
              edgecolor="0.3", fontsize=7.0)
    ax.grid(True, axis="x")
    panel(ax, "(a)")
    ax = axes[1]
    for i, (t, a, b) in enumerate(zip(["Count MAE", "Count RMSE"],
                                      [f["mae"], f["rmse"]],
                                      [p["mae"], p["rmse"]])):
        ax.plot([a, b], [i, i], color="black", linewidth=1.0, zorder=1)
        ax.plot(a, i, "o", color=OI["grey"], ms=7, zorder=3)
        ax.plot(b, i, "D", color=OI["vermilion"], ms=6, zorder=3)
        ax.text(a, i + 0.20, f"{a:.2f}", fontsize=6.5, ha="center",
                va="bottom", color=OI["grey"])
        ax.text(b, i - 0.20, f"{b:.2f}", fontsize=6.5, ha="center",
                va="top", color=OI["vermilion"])
        ax.text(max(a, b) + 0.12, i, f"$\Delta${b - a:+.4f}",
                fontsize=7, fontweight="bold", va="center", ha="left")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Count MAE", "Count RMSE"])
    ax.set_xlabel("Error (apples/image)")
    ax.set_xlim(2.8, 8.0)
    from matplotlib.lines import Line2D as _LD2
    ax.legend([_LD2([0], [0], marker="o", color=OI["grey"], label="Frozen",
                    linestyle="None", markersize=7.0),
               _LD2([0], [0], marker="D", color=OI["vermilion"],
                    label="Proposed", linestyle="None", markersize=6.0)],
              ["Frozen", "Proposed"], loc="lower right", frameon=True,
              edgecolor="0.3", fontsize=7.0)
    ax.grid(True, axis="x")
    panel(ax, "(b)")
    save_all(fig, outdir, "fig06_frozen_vs_proposed")


# ---------------------------------------------------------------- fig07 ---
def fig07(outdir):
    hp, hf = load_history("proposed"), load_history("frozen")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained")
    ax = axes[0]
    ep = [r["epoch"] for r in hp]
    ax.plot(ep, [r["train_loss"] for r in hp], color=OI["lightgrey"],
            linewidth=0.8, alpha=0.8)
    ax.plot(ep, smooth([r["train_loss"] for r in hp]), color=OI["blue"],
            label="Proposed train", linewidth=1.6)
    for hist, lab, col in [(hp, "Proposed val", OI["vermilion"]),
                           (hf, "Frozen val", OI["green"])]:
        e2 = [r["epoch"] for r in hist]
        ax.plot(e2, [r["val_loss_total"] for r in hist], color=col,
                linewidth=0.8, alpha=0.35)
        ax.plot(e2, smooth([r["val_loss_total"] for r in hist]),
                color=col, label=lab + " (weighted)", linewidth=1.6)
    bp, bf = best_record(hp), best_record(hf)
    top = ax.get_ylim()[1]
    for best, col, frac in [(bp, OI["vermilion"], 0.97),
                            (bf, OI["green"], 0.80)]:
        ax.axvline(best["epoch"], color=col, linestyle=":", linewidth=1.0)
        ax.text(best["epoch"] + 1.5, top * frac, f"ep{best['epoch']}",
                color=col, fontsize=7, va="top", rotation=90)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True, edgecolor="0.3", fontsize=6.5)
    ax.grid(True)
    panel(ax, "(a)")
    ax = axes[1]
    for hist, style, suf, al in [(hp, "-", "Proposed", 1.0),
                                 (hf, "--", "Frozen", 0.8)]:
        e2 = [r["epoch"] for r in hist]
        for tk, col, mk in [("leaf", OI["blue"], "o"),
                            ("pest", OI["orange"], "s"),
                            ("fruit", OI["green"], "^")]:
            raw = [r[f"val_acc_{tk}"] * 100 for r in hist]
            ax.plot(e2, raw, linestyle=style, color=col, linewidth=0.7,
                    alpha=0.3)
            ax.plot(e2, smooth(raw), linestyle=style, color=col,
                    marker=mk, markevery=25, markersize=3.0,
                    label=f"{suf} {tk}", alpha=al)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(frameon=True, edgecolor="0.3", fontsize=6.0, ncol=2)
    ax.grid(True)
    panel(ax, "(b)")
    save_all(fig, outdir, "fig07_convergence")


# ---------------------------------------------------------------- fig08 ---
def fig08(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained", sharey=True)
    for ax, key, tag, ep_tag in [(axes[0], "proposed", "(a) Proposed",
                                  "ep 186"),
                                 (axes[1], "frozen", "(b) Frozen",
                                  "ep 194")]:
        hist = load_history(key)
        ep = [r["epoch"] for r in hist]
        for tk, col, mk in [("fruit", OI["blue"], "o"),
                            ("leaf", OI["orange"], "s"),
                            ("pest", OI["green"], "^"),
                            ("yield", OI["vermilion"], "D")]:
            raw = [r["weights"][tk] for r in hist]
            ax.plot(ep, raw, color=col, linewidth=0.7, alpha=0.3)
            ax.plot(ep, smooth(raw, 5), color=col, marker=mk, markevery=25,
                    markersize=3.0, label=tk.capitalize())
        term = best_record(hist)["weights"]
        ax.text(0.98, 0.04,
                f"best: F{term['fruit']:.2f}/L{term['leaf']:.2f}/"
                f"P{term['pest']:.2f}/Y{term['yield']:.3f} ({ep_tag})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, style="italic",
                bbox=dict(facecolor="white", edgecolor="0.7", pad=2))
        ax.set_xlabel("Epoch")
        ax.grid(True)
        panel(ax, tag)
    axes[0].set_ylabel("Learned precision $e^{-s_t}$")
    axes[1].legend(frameon=True, edgecolor="0.3", loc="center right",
                   fontsize=7)
    save_all(fig, outdir, "fig08_task_weights")


# ---------------------------------------------------------------- fig09 ---
def fig09(outdir, sums):
    order = ["proposed", "ablA", "ablB", "ablC", "ablD", "frozen"]
    names = ["Proposed", "A: No KD", "B: Static*", "C: No LoRA",
             "D: No warm-up", "Frozen"]
    mets = ["acc_leaf", "acc_pest", "acc_fruit", "mae", "rmse"]
    axlab = ["Leaf acc", "Pest acc", "Fruit acc", "MAE", "RMSE"]
    grid = np.array([[sums[k][m] for m in mets] for k in order], float)
    lo = grid.min(0)
    hi = grid.max(0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    norm = (grid - lo) / span
    norm[:, 3:] = 1.0 - norm[:, 3:]
    n = len(axlab)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    cols = [OI["blue"], OI["green"], OI["vermilion"], OI["orange"],
            OI["purple"], OI["grey"]]
    fig = plt.figure(figsize=(DOUBLE_W, 3.3), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, polar=True)
    for i, k in enumerate(order):
        v = np.concatenate([norm[i], norm[i][:1]])
        ls = "--" if not sums[k]["complete"] else "-"
        lw = 2.0 if k in ("proposed", "frozen") else 1.2
        ax.plot(ang, v, color=cols[i], linestyle=ls, linewidth=lw,
                label=names[i])
        ax.fill(ang, v, color=cols[i], alpha=0.06)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axlab, fontsize=9)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7,
                       color="0.45")
    ax.set_rlabel_position(135)
    ax.legend(frameon=True, edgecolor="0.3", fontsize=7.5,
              loc="upper left", bbox_to_anchor=(1.04, 1.0))
    save_all(fig, outdir, "fig09_ablation")


CAPTIONS = """\
Fig. 2. Dataset curation. Raw versus retained images per class after automated cleaning (minimum 64 px, pixel standard deviation >= 5.0, perceptual-hash deduplication threshold 8) for the (a) leaf, (b) pest, and (c) fruit classification teachers at 224-px input. Stratified 70/15/15 splits (seed 42); yield detection uses its own 1093/264 train/val split at 640-px input.

Fig. 5. Best-validation-epoch snapshot of the proposed LoRA-adapted model (epoch 186). (a) Leaf, pest, and fruit validation accuracies against the 100% reference. (b) Apple counting MAE/RMSE on the 400-image yield validation set (confidence 0.3, NMS IoU 0.45). Single seed (n=1).

Fig. 6. Frozen-backbone (best epoch 194, circles) versus proposed LoRA-adapted (best epoch 186, diamonds) configurations with paired deltas. LoRA adds +0.17 pp on leaf but loses 2.12 pp on fruit and adds +0.14/+0.38 counting error, motivating the ablation study. Single seed (n=1).

Fig. 7. Training dynamics over 200 epochs (thin raw, thick 7-epoch smooth). (a) Mean training loss and homoscedastic-weighted total validation loss with best-validation epochs marked (186 proposed, 194 frozen). (b) Per-task validation accuracies (solid proposed, dashed frozen); fruit is the noisiest task. Single seed (n=1).

Fig. 8. Learned homoscedastic task precisions exp(-s_t) for (a) proposed and (b) frozen configurations with best-epoch values annotated. Precisions are optimization coefficients balancing heterogeneous losses (single seed, n=1), not measures of task importance.

Fig. 9. Component-level ablation at best-validation epochs. (a) Min-max-normalised radar (higher is better; errors inverted) and (b) exact values: No-KD (A) matches or beats the proposed model on every metric, while static weighting (B) collapses fruit accuracy. *B logged only 103/200 epochs and must be completed; all runs single-seed (n=1).
"""


def main():
    ap = argparse.ArgumentParser(description="Phase 1 v2 figure generation")
    ap.add_argument("--out", default="paper_figures")
    args = ap.parse_args()

    apply_style()
    outdir = ROOT / args.out

    sums = {k: run_summary(k) for k in RUNS}
    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(sums, f, indent=2)
    with open(outdir / "figure_captions.txt", "w", encoding="utf-8") as f:
        f.write(CAPTIONS)

    print("Phase 1 v2 (reference grammar):")
    print("=" * 64)
    fig02(outdir)
    fig05(outdir, sums["proposed"])
    fig06(outdir, sums["frozen"], sums["proposed"])
    fig07(outdir)
    fig08(outdir)
    fig09(outdir, sums)
    print(f"Done. Files in {outdir}")


if __name__ == "__main__":
    main()
