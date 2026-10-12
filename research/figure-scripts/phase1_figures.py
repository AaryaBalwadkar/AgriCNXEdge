"""
phase1_figures.py
=================
Phase 1 — CPU-only generation of Q1-grade figures for the Agri-CNX-Edge
paper from already-logged training artifacts. No GPU, no retraining,
no torch required: inputs are plain JSON files.

USAGE:
    python phase1_figures.py
    python phase1_figures.py --out paper_figures
    python phase1_figures.py --configs proposed frozen   (subset of figures)

INPUTS (relative to this file's folder):
    Proposed Model/student/training_history.json + yield_eval.json
    Ablation Frozen Backbone/outputs/studentFrozen/training_history.json + yield_eval.json
    Ablation/outputs/ablation_{A_no_kd,B_static_weights,C_no_lora,D_no_warmup}/
        training_history.json + yield_eval.json
    PDDD Fruit 3 and PDDD Leaf 1/outputs/{leaf/seed_42/*,pest/*,fruit/seed_42/*}
        *_cleaning_report.json

OUTPUTS (in <out>/):
    fig02_dataset.{pdf,tif,png}          Fig 2: cleaning raw->clean per class
    fig05_final_performance.{pdf,tif,png} Fig 5: proposed best-epoch metrics
    fig06_frozen_vs_proposed.{pdf,tif,png} Fig 6: frozen vs proposed + deltas
    fig07_convergence.{pdf,tif,png}      Fig 7: loss + accuracy trajectories
    fig08_task_weights.{pdf,tif,png}     Fig 8: homoscedastic precisions
    fig09_ablation.{pdf,tif,png}         Fig 9: ablation comparison (B flagged)
    metrics_summary.json                 best-epoch numbers backing the paper
    figure_captions.txt                  draft self-contained captions

ELSEVIER COMPLIANCE (Computers and Electronics in Agriculture):
    * Vector PDF primary (fonts embedded Type 42) + TIFF LZW @1000dpi
      (line-art tier) at exact single (90mm/3.54in) or double (185mm/7.3in)
      column widths. PNG @600dpi included for draft manuscript only
      (Elsevier production does not accept PNG).
    * Arial family, >=7pt at final size; data lines ~1.2pt, axes 0.8pt.
    * Okabe-Ito colorblind-safe palette + hatch/marker redundancy so
      figures survive grayscale printing.
    * Bold (a)/(b)/(c) panel labels; value labels on bars; best epochs
      marked; incomplete ablation B hatched and asterisked, never hidden.
    * Bar axes start at 0 (no truncated-axis exaggeration).

ATTRIBUTION (verified by checkpoint forensics, Phase 0):
    proposed = Proposed Model/student            (LoRA adapters present)
    frozen   = Ablation Frozen Backbone/...      (no LoRA layers)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

RUNS = {
    "proposed": {
        "label": "Proposed (LoRA)",
        "hist": ROOT / "Proposed Model" / "student" / "training_history.json",
        "yield": ROOT / "Proposed Model" / "student" / "yield_eval.json",
    },
    "frozen": {
        "label": "Frozen backbone",
        "hist": (ROOT / "Ablation Frozen Backbone" / "outputs" / "studentFrozen"
                 / "training_history.json"),
        "yield": (ROOT / "Ablation Frozen Backbone" / "outputs" / "studentFrozen"
                  / "yield_eval.json"),
    },
    "ablA": {
        "label": "A: No KD",
        "hist": ROOT / "Ablation" / "outputs" / "ablation_A_no_kd" / "training_history.json",
        "yield": ROOT / "Ablation" / "outputs" / "ablation_A_no_kd" / "yield_eval.json",
    },
    "ablB": {
        "label": "B: Static weights",
        "hist": (ROOT / "Ablation" / "outputs" / "ablation_B_static_weights"
                 / "training_history.json"),
        "yield": (ROOT / "Ablation" / "outputs" / "ablation_B_static_weights"
                  / "yield_eval.json"),
    },
    "ablC": {
        "label": "C: No LoRA",
        "hist": ROOT / "Ablation" / "outputs" / "ablation_C_no_lora" / "training_history.json",
        "yield": ROOT / "Ablation" / "outputs" / "ablation_C_no_lora" / "yield_eval.json",
    },
    "ablD": {
        "label": "D: No warm-up",
        "hist": (ROOT / "Ablation" / "outputs" / "ablation_D_no_warmup"
                 / "training_history.json"),
        "yield": (ROOT / "Ablation" / "outputs" / "ablation_D_no_warmup"
                  / "yield_eval.json"),
    },
}

CLEANING = {
    "leaf": (ROOT / "PDDD Fruit 3 and PDDD Leaf 1" / "outputs" / "leaf"
             / "seed_42" / "leaf_cleaning_report.json"),
    "pest": (ROOT / "PDDD Fruit 3 and PDDD Leaf 1" / "outputs" / "pest"
             / "pest_cleaning_report.json"),
    "fruit": (ROOT / "PDDD Fruit 3 and PDDD Leaf 1" / "outputs" / "fruit"
              / "seed_42" / "fruit_cleaning_report.json"),
}

CLASS_ORDER = {
    "leaf": ["Apple_Mosaic", "Apple___Black_rot", "Alternaria", "Healthy"],
    "pest": ["xylotrechus", "aphids", "leafhoppers", "spider_mite"],
    "fruit": ["Anthracnose", "Black Rot", "Healthy"],
}

# Okabe-Ito (colorblind-safe) + grayscale-redundant hatches/markers
OI = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}
HATCHES = ["", "///", "\\\\\\", "+++", "xxx", "..."]
MARKERS = ["o", "s", "^", "D", "v", "P"]

SINGLE_W = 3.54   # 90 mm Elsevier single column
DOUBLE_W = 7.28   # 185 mm Elsevier double column


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def apply_q1_style(base_pt=8.0):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": base_pt,
        "axes.titlesize": base_pt + 0.5,
        "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 1.0,
        "ytick.labelsize": base_pt - 1.0,
        "legend.fontsize": base_pt - 1.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "pdf.fonttype": 42,   # embed TrueType (Elsevier AQC requirement)
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def panel_label(ax, tag, x=0.02, y=0.96):
    ax.text(x, y, tag, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


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


def annotate_bars(ax, rects, fmt="{:.2f}", size=6.5, dy=2):
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_history(key):
    with open(RUNS[key]["hist"], encoding="utf-8") as f:
        return json.load(f)


def best_record(hist):
    return min(hist, key=lambda r: r.get("val_loss_total", float("inf")))


def load_yield(key):
    with open(RUNS[key]["yield"], encoding="utf-8") as f:
        d = json.load(f)
    mae = d.get("count_mae", d.get("mae"))
    rmse = d.get("count_rmse", d.get("rmse"))
    n = d.get("n_images", d.get("n"))
    return {"mae": float(mae), "rmse": float(rmse), "n": int(n)}


def run_summary(key):
    hist = load_history(key)
    b = best_record(hist)
    y = load_yield(key)
    return {
        "key": key,
        "label": RUNS[key]["label"],
        "epochs_logged": len(hist),
        "complete": len(hist) >= 200,
        "best_epoch": int(b["epoch"]),
        "best_val_total": round(float(b["val_loss_total"]), 4),
        "train_at_best": round(float(b["train_loss"]), 4),
        "acc_leaf": round(float(b["val_acc_leaf"]) * 100, 2),
        "acc_pest": round(float(b["val_acc_pest"]) * 100, 2),
        "acc_fruit": round(float(b["val_acc_fruit"]) * 100, 2),
        "w_fruit": round(float(b["weights"]["fruit"]), 4),
        "w_leaf": round(float(b["weights"]["leaf"]), 4),
        "w_pest": round(float(b["weights"]["pest"]), 4),
        "w_yield": round(float(b["weights"]["yield"]), 4),
        "mae": round(y["mae"], 4),
        "rmse": round(y["rmse"], 4),
        "n_yield": y["n"],
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig02_dataset(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.6),
                             layout="constrained")
    for ax, (task, tag) in zip(axes, [("leaf", "(a)"),
                                      ("pest", "(b)"),
                                      ("fruit", "(c)")] ):
        with open(CLEANING[task], encoding="utf-8") as f:
            rep = json.load(f)
        names = CLASS_ORDER[task]
        raw = [rep[c]["original_count"] for c in names]
        clean = [rep[c]["cleaned_count"] for c in names]
        x = np.arange(len(names))
        w = 0.38
        r1 = ax.bar(x - w / 2, raw, w, label="Raw",
                    color=OI["grey"], edgecolor="black", linewidth=0.6)
        r2 = ax.bar(x + w / 2, clean, w, label="Clean",
                    color=OI["blue"], edgecolor="black", linewidth=0.6,
                    hatch="//")
        annotate_bars(ax, list(r1) + list(r2), fmt="{:.0f}", size=6)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Apple___", "").replace("_", " ")
                            for n in names], rotation=18, ha="right")
        ax.set_ylabel("Images" if task == "leaf" else "")
        ax.set_title(f"{task.capitalize()} teachers (224 px)")
        ax.grid(True, axis="y")
        panel_label(ax, tag)
    axes[2].legend(frameon=True, edgecolor="black", loc="upper left")
    fig.text(0.5, 0.0,
             "Cleaning: min 64 px, pixel-std \u2265 5.0, pHash-dedup \u2264 8. "
             "Stratified 70/15/15 split, seed 42. Yield: 1093 train / 264 val (640 px).",
             ha="center", fontsize=6.5, style="italic")
    save_all(fig, outdir, "fig02_dataset")


def fig05_final(outdir, s):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.6),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    accs = [s["acc_leaf"], s["acc_pest"], s["acc_fruit"]]
    rects = ax.bar(["Leaf", "Pest", "Fruit"], accs, width=0.55,
                   color=[OI["blue"], OI["orange"], OI["green"]],
                   edgecolor="black", linewidth=0.6,
                   hatch=["", "///", "\\\\\\"])
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h - 2.0, f"{h:.2f}%",
                ha="center", va="top", fontsize=7, color="white",
                fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification (validation, best epoch 186)")
    ax.grid(True, axis="y")
    panel_label(ax, "(a)")

    ax = axes[1]
    rects = ax.bar(["Count MAE", "Count RMSE"], [s["mae"], s["rmse"]],
                   width=0.55, color=[OI["vermilion"], OI["purple"]],
                   edgecolor="black", linewidth=0.6, hatch=["", "+++"])
    annotate_bars(ax, rects)
    ax.set_ylabel("Error (apples/image)")
    ax.set_title(f"Counting (yield val, n={s['n_yield']})")
    ax.grid(True, axis="y")
    panel_label(ax, "(b)")
    save_all(fig, outdir, "fig05_final_performance")


def fig06_compare(outdir, f, p):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained")
    ax = axes[0]
    tasks = ["Leaf", "Pest", "Fruit"]
    fv = [f["acc_leaf"], f["acc_pest"], f["acc_fruit"]]
    pv = [p["acc_leaf"], p["acc_pest"], p["acc_fruit"]]
    x = np.arange(len(tasks))
    w = 0.36
    ax.bar(x - w / 2, fv, w, label="Frozen (ep 194)", color=OI["grey"],
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, pv, w, label="Proposed (ep 186)", color=OI["blue"],
           edgecolor="black", linewidth=0.6, hatch="//")
    for r in ax.patches:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h - 2.0, f"{h:.2f}",
                ha="center", va="top", fontsize=6.5, color="white",
                fontweight="bold")
    for i, (a, b) in enumerate(zip(fv, pv)):
        d = b - a
        ax.annotate(f"$\Delta${d:+.2f} pp",
                    xy=(i, max(a, b) + 4.5), ha="center", fontsize=6.5,
                    fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification accuracy")
    ax.legend(frameon=True, edgecolor="black", loc="lower right")
    ax.grid(True, axis="y")
    panel_label(ax, "(a)")

    ax = axes[1]
    mets = ["MAE", "RMSE"]
    fv = [f["mae"], f["rmse"]]
    pv = [p["mae"], p["rmse"]]
    x = np.arange(len(mets))
    ax.bar(x - w / 2, fv, w, label="Frozen", color=OI["grey"],
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, pv, w, label="Proposed", color=OI["vermilion"],
           edgecolor="black", linewidth=0.6, hatch="//")
    for r in ax.patches:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h - 0.12, f"{h:.2f}",
                ha="center", va="top", fontsize=6.5, color="white",
                fontweight="bold")
    for i, (a, b) in enumerate(zip(fv, pv)):
        ax.annotate(f"$\Delta${b - a:+.4f}", xy=(i, max(a, b) + 0.45),
                    ha="center", fontsize=6.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Count MAE", "Count RMSE"])
    ax.set_ylabel("Error (apples/image)")
    ax.set_title("Counting error (lower is better)")
    ax.legend(frameon=True, edgecolor="black", loc="center")
    ax.grid(True, axis="y")
    panel_label(ax, "(b)")
    save_all(fig, outdir, "fig06_frozen_vs_proposed")


def fig07_convergence(outdir):
    hp, hf = load_history("proposed"), load_history("frozen")
    bp, bf = best_record(hp), best_record(hf)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained")
    ax = axes[0]
    for hist, best, lab, col, mk in [
            (hp, bp, "Proposed train", OI["blue"], "o"),
            (hp, bp, "Proposed val (weighted)", OI["vermilion"], "s"),
            (hf, bf, "Frozen val (weighted)", OI["green"], "^")]:
        ep = [r["epoch"] for r in hist]
        key = "train_loss" if lab.endswith("train") else "val_loss_total"
        ax.plot(ep, [r[key] for r in hist], label=lab, color=col,
                marker=mk, markevery=20, markersize=3.5)
    top = ax.get_ylim()[1]
    for best, col, yfrac in [(bp, OI["vermilion"], 0.98),
                             (bf, OI["green"], 0.80)]:
        ax.axvline(best["epoch"], color=col, linestyle=":", linewidth=1.0)
        ax.text(best["epoch"] + 1.5, top * yfrac, f"ep{best['epoch']}",
                color=col, fontsize=7, va="top", rotation=90)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Weighted total loss")
    ax.legend(frameon=True, edgecolor="black", fontsize=6.5)
    ax.grid(True)
    panel_label(ax, "(a)")

    ax = axes[1]
    for hist, style, suf in [(hp, "-", "Proposed"), (hf, "--", "Frozen")]:
        ep = [r["epoch"] for r in hist]
        for tk, col, mk in [("leaf", OI["blue"], "o"),
                            ("pest", OI["orange"], "s"),
                            ("fruit", OI["green"], "^")]:
            ax.plot(ep, [r[f"val_acc_{tk}"] * 100 for r in hist],
                    linestyle=style, color=col, marker=mk, markevery=25,
                    markersize=3.0, label=f"{suf} {tk}",
                    alpha=1.0 if style == "-" else 0.65)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Per-task validation accuracy")
    ax.legend(frameon=True, edgecolor="black", fontsize=6.0, ncol=2)
    ax.grid(True)
    panel_label(ax, "(b)")
    save_all(fig, outdir, "fig07_convergence")


def fig08_weights(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.7),
                             layout="constrained", sharey=True)
    for ax, key, tag, ep_tag in [(axes[0], "proposed", "(a) Proposed", "ep 186"),
                                 (axes[1], "frozen", "(b) Frozen", "ep 194")]:
        hist = load_history(key)
        ep = [r["epoch"] for r in hist]
        for tk, col, mk in [("leaf", OI["blue"], "s"),
                            ("pest", OI["orange"], "^"),
                            ("fruit", OI["green"], "o"),
                            ("yield", OI["vermilion"], "D")]:
            ax.plot(ep, [r["weights"][tk] for r in hist], label=tk.capitalize(),
                    color=col, marker=mk, markevery=25, markersize=3.0)
        term = hist[-1]["weights"]
        ax.text(0.98, 0.04,
                f"terminal: F{term['fruit']:.2f}/L{term['leaf']:.2f}/"
                f"P{term['pest']:.2f}/Y{term['yield']:.3f}\n({ep_tag})",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, style="italic",
                bbox=dict(facecolor="white", edgecolor="0.7", pad=2))
        ax.set_xlabel("Epoch")
        ax.grid(True)
        panel_label(ax, tag)
    axes[0].set_ylabel("Learned precision $e^{-s_t}$")
    axes[1].legend(frameon=True, edgecolor="black", loc="center right",
                   fontsize=7)
    fig.text(0.5, 0.0, "Precisions weight the four task losses; "
             "not a measure of task importance.", ha="center",
             fontsize=6.5, style="italic")
    save_all(fig, outdir, "fig08_task_weights")


def fig09_ablation(outdir, sums):
    order = ["proposed", "ablA", "ablB", "ablC", "ablD", "frozen"]
    labels = ["Proposed" if sums[k]["key"] == "proposed"
              else "Frozen" if sums[k]["key"] == "frozen"
              else sums[k]["label"].split(":")[0]
              for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.9),
                             layout="constrained")
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.24
    cols = [OI["blue"], OI["orange"], OI["green"]]
    for j, tk in enumerate(["acc_leaf", "acc_pest", "acc_fruit"]):
        vals = [sums[k][tk] for k in order]
        bars = ax.bar(x + (j - 1) * w, vals, w,
                      label=["Leaf", "Pest", "Fruit"][j], color=cols[j],
                      edgecolor="black", linewidth=0.6, hatch=HATCHES[j * 2])
        for b, k in zip(bars, order):
            if not sums[k]["complete"]:
                b.set_hatch("xx")
                b.set_edgecolor(OI["vermilion"])
    ax.set_xticks(x)
    ax.set_xticklabels([l + ("*" if not sums[k]["complete"] else "")
                        for l, k in zip(labels, order)], rotation=14, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification (best-epoch val)")
    ax.legend(frameon=True, edgecolor="black", fontsize=6.5)
    ax.grid(True, axis="y")
    panel_label(ax, "(a)")

    ax = axes[1]
    for j, tk in enumerate(["mae", "rmse"]):
        vals = [sums[k][tk] for k in order]
        bars = ax.bar(x + (j - 0.5) * w * 1.6, vals, w * 1.6,
                      label=["MAE", "RMSE"][j],
                      color=[OI["vermilion"], OI["purple"]][j],
                      edgecolor="black", linewidth=0.6,
                      hatch=["", "+++"][j])
        for b, k in zip(bars, order):
            if not sums[k]["complete"]:
                b.set_hatch("xx")
                b.set_edgecolor(OI["vermilion"])
    ax.set_xticks(x)
    ax.set_xticklabels([l + ("*" if not sums[k]["complete"] else "")
                        for l, k in zip(labels, order)], rotation=14, ha="right")
    ax.set_ylabel("Error (apples/image)")
    ax.set_title("Counting error (lower is better)")
    ax.legend(frameon=True, edgecolor="black", fontsize=6.5)
    ax.grid(True, axis="y")
    panel_label(ax, "(b)")
    fig.text(0.5, 0.0, "*B: Static weights — 103 epochs logged (incomplete); "
             "value is best-so-far, not final.", ha="center", fontsize=6.5,
             style="italic")
    save_all(fig, outdir, "fig09_ablation")


CAPTIONS = """\
Fig. 2. Dataset curation. Raw versus retained images per class after automated cleaning (minimum 64 px, pixel standard deviation >= 5.0, perceptual-hash deduplication threshold 8) for the (a) leaf, (b) pest, and (c) fruit classification teachers at 224-px input. Stratified 70/15/15 splits (seed 42) per task; yield detection uses its own 1093/264 train/val split at 640-px input.

Fig. 5. Best-validation-epoch performance of the proposed LoRA-adapted model (epoch 186). (a) Leaf, pest, and fruit classification accuracies on the validation split. (b) Apple counting MAE/RMSE on the 400-image yield validation set (confidence 0.3, NMS IoU 0.45).

Fig. 6. Frozen-backbone (best epoch 194) versus proposed LoRA-adapted (best epoch 186) configurations. (a) Validation accuracies with percentage-point deltas. (b) Counting errors with absolute deltas (lower is better). Checkpoint forensics confirm attribution: the proposed checkpoint contains trained LoRA adapters, the frozen checkpoint contains none.

Fig. 7. Training dynamics for proposed and frozen configurations over 200 epochs. (a) Mean training loss and homoscedastic-weighted total validation loss; dotted lines mark best-validation epochs (186 proposed, 194 frozen). (b) Per-task validation accuracies (solid: proposed; dashed: frozen).

Fig. 8. Learned homoscedastic task precisions exp(-s_t) for (a) proposed and (b) frozen configurations with terminal values annotated. Precisions are optimization coefficients balancing heterogeneous losses, not measures of task importance.

Fig. 9. Component-level ablation at best-validation epochs. (a) Classification accuracies and (b) counting errors for Proposed, A (no KD), B (static weights), C (no LoRA), D (no warm-up), and Frozen. *B logged only 103 epochs and must be completed before its row is final.
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 1 Q1 figure generation")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset drafter, e.g. --configs proposed frozen")
    args = ap.parse_args()

    apply_q1_style()
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    print("Phase 1: Q1 figure generation (CPU-only)")
    print("=" * 64)
    for key, meta in RUNS.items():
        ok_h, ok_y = meta["hist"].exists(), meta["yield"].exists()
        print(f"  {key:<9} hist={'OK' if ok_h else 'MISSING'} "
              f"yield={'OK' if ok_y else 'MISSING'}")
        if not (ok_h and ok_y):
            raise FileNotFoundError(f"missing artifact for run '{key}'")

    sums = {k: run_summary(k) for k in RUNS}
    with open(outdir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(sums, f, indent=2)
    with open(outdir / "figure_captions.txt", "w", encoding="utf-8") as f:
        f.write(CAPTIONS)

    print("\nBest-epoch summary (validation):")
    for k in ["proposed", "frozen", "ablA", "ablB", "ablC", "ablD"]:
        s = sums[k]
        flag = "" if s["complete"] else "  <-- INCOMPLETE"
        print(f"  {s['label']:<20} ep{s['best_epoch']:<4} "
              f"L{s['acc_leaf']:.2f}/P{s['acc_pest']:.2f}/F{s['acc_fruit']:.2f} "
              f"MAE{s['mae']:.4f} RMSE{s['rmse']:.4f}{flag}")

    only = set(args.configs) if args.configs else None
    print("\nRendering:")
    fig02_dataset(outdir)
    if only is None or {"proposed", "frozen"} & only:
        fig05_final(outdir, sums["proposed"])
        fig06_compare(outdir, sums["frozen"], sums["proposed"])
        fig07_convergence(outdir)
        fig08_weights(outdir)
    if only is None:
        fig09_ablation(outdir, sums)
    print(f"\nDone. Files in {outdir}")


if __name__ == "__main__":
    main()
