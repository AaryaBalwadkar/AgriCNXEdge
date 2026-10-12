"""Phase 2 Q1 figures (CPU-only, no GPU).

Rebuilds Figs 10-15 from phase2_results artifacts in paper style
(serif Times, hairline, Okabe-Ito; NO caption sentences inside figures --
axis/tick labels, cell values, panel letters and count labels only):

  fig10_leaf_cm    proposed TEST CM (4x4, diagonal from P/R/F1=1.0 + supports)
  fig11_pest_cm    proposed TEST CM (exact: 1 leafhoppers->xylotrechus error)
  fig12_fruit_cm   proposed TEST CM (exact: 9 BlackRot->Anthracnose + 1->Healthy;
                   Healthy support 0, row shown as "--")
  fig13_counting   pred-vs-GT scatter, 400 yield images + identity line
  fig14_boxes      2-panel GT-vs-pred overlay (sparse + dense)
  fig15_qualitative 6-panel overlay strip with count labels

Outputs fig{10,11,12,13,14,15}_*.{pdf,png,tif} into paper_figures/.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

ROOT = Path(__file__).resolve().parent
P2 = ROOT / "phase2_results" / "proposed"
OV = P2 / "overlays"
OUT = ROOT / "paper_figures"

LEAF_ORDER = ["Apple_Mosaic", "Apple___Black_rot", "Alternaria", "Healthy"]
LEAF_SHORT = ["Mosaic", "Black rot", "Alternaria", "Healthy"]
LEAF_SUP_T = [101, 103, 90, 329]
PEST_ORDER = ["xylotrechus", "aphids", "leafhoppers", "spider_mite"]
PEST_SUP_T = [52, 51, 82, 40]
FRUIT_ORDER = ["Anthracnose", "Black Rot", "Healthy"]
FRUIT_SUP_T = [55, 177, 0]


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
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
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


def save(fig, base):
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def cm_matrix_fig(cm, labels, base, xlabel="Predicted", ylabel="True"):
    cm = np.asarray(cm, dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1.0, cm.max()))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if np.isnan(v):
                t = "--"
                c = "#555555"
            else:
                t = f"{int(round(v))}"
                c = "white" if v > cm.max() / 2 else "black"
            ax.text(j, i, t, ha="center", va="center", fontsize=8.0, color=c)
    for spine in ax.spines.values():
        spine.set_visible(True)
    save(fig, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    # --- Fig 10: leaf TEST CM (diagonal; all P/R/F1 = 1.0) ---
    cm_leaf = np.diag(np.array(LEAF_SUP_T, dtype=float))
    cm_matrix_fig(cm_leaf, LEAF_SHORT, out / "fig10_leaf_cm")

    # --- Fig 11: pest TEST CM (exact single error) ---
    # xylotrechus P=52/53 -> one FP; leafhoppers R=81/82 -> one FN;
    # leafhoppers P=1.0 -> error lands in xylotrechus column.
    cm_pest = np.zeros((4, 4))
    cm_pest[0, 0] = 52
    cm_pest[1, 1] = 51
    cm_pest[2, 2] = 81
    cm_pest[2, 0] = 1
    cm_pest[3, 3] = 40
    cm_matrix_fig(cm_pest, PEST_ORDER, out / "fig11_pest_cm")

    # --- Fig 12: fruit TEST CM (GT-box ROI diagnostic) ---
    # Anthracnose R=1.0 (55->Anth); BR R=167/177, P(BR)=1.0, P(Anth)=55/64 ->
    # 9 BR->Anth + 1 BR->Healthy (only integer solution); Healthy support 0.
    cm_fruit = np.zeros((3, 3))
    cm_fruit[0, 0] = 55
    cm_fruit[1, 1] = 167
    cm_fruit[1, 0] = 9
    cm_fruit[1, 2] = 1
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    disp = cm_fruit.copy()
    im = ax.imshow(np.where(np.arange(3)[:, None] == 2, np.nan, disp),
                   cmap="Blues", vmin=0, vmax=max(1.0, cm_fruit.max()))
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(FRUIT_ORDER, rotation=14, ha="right")
    ax.set_yticklabels(FRUIT_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            if i == 2:
                t, c = "--", "#555555"
            else:
                t, c = f"{int(cm_fruit[i, j])}", "white" if cm_fruit[i, j] > 40 else "black"
            ax.text(j, i, t, ha="center", va="center", fontsize=8.0, color=c)
    save(fig, out / "fig12_fruit_cm")

    # --- Fig 13: pred-vs-GT scatter (400 yield images, proposed) ---
    gt, pr = [], []
    with open(P2 / "yield_per_image.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            gt.append(float(r["gt_count"]))
            pr.append(float(r["pred_count"]))
    assert len(gt) == 400, f"yield rows {len(gt)}"
    gt = np.array(gt)
    pr = np.array(pr)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(gt, pr, s=9, c="#0072B2", alpha=0.45, linewidths=0, zorder=3)
    lim = [0, max(gt.max(), pr.max()) * 1.05]
    ax.plot(lim, lim, color="#8A8A8A", linewidth=0.9, linestyle="--", zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ground-truth count")
    ax.set_ylabel("Predicted count")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    save(fig, out / "fig13_counting")

    # --- Figs 14/15: overlay panels ---
    # Fig 14: sparse + dense pair; Fig 15: 6-case strip incl. dense/occlusion.
    counts = {}
    with open(P2 / "yield_per_image.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            counts[r["image"]] = (int(r["gt_count"]), int(r["pred_count"]))
    ov_files = sorted(OV.glob("bbox_*.png"))
    assert len(ov_files) >= 6, f"overlays {len(ov_files)}"

    def panel(fig_ax, img_path, tag):
        im = Image.open(img_path).convert("RGB")
        fig_ax.imshow(im)
        fig_ax.axis("off")
        stem = img_path.stem.split("_", 2)[-1] + ".jpg"
        g, p = counts.get(stem, (-1, -1))
        fig_ax.text(0.02, 0.96, tag, transform=fig_ax.transAxes, fontsize=9.0,
                    fontweight="bold", color="white", va="top", ha="left",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.5,
                              edgecolor="none", boxstyle="round,pad=0.25"))
        fig_ax.text(0.98, 0.04, f"GT {g} / pred {p}", transform=fig_ax.transAxes,
                    fontsize=7.0, color="white", va="bottom", ha="right",
                    bbox=dict(facecolor="black", alpha=0.55, pad=1.5,
                              edgecolor="none", boxstyle="round,pad=0.25"))

    # sparse (min pred) + dense (max gt)
    by_pred = sorted(ov_files, key=lambda f: counts.get(f.stem.split("_", 2)[-1] + ".jpg", (0, 0))[1])
    by_gt = sorted(ov_files, key=lambda f: counts.get(f.stem.split("_", 2)[-1] + ".jpg", (0, 0))[0])
    fig, axs = plt.subplots(1, 2, figsize=(7.28, 2.9))
    panel(axs[0], by_pred[0], "(a)")
    panel(axs[1], by_gt[-1], "(b)")
    fig.tight_layout(pad=0.4, w_pad=0.6)
    save(fig, out / "fig14_boxes")

    # 6-case strip: sparse, mid x2, dense x2, + one more (sorted by gt)
    mid = sorted(ov_files, key=lambda f: counts.get(f.stem.split("_", 2)[-1] + ".jpg", (0, 0))[0])
    pick = [mid[0], mid[len(mid) // 3], mid[len(mid) // 2],
            mid[2 * len(mid) // 3], mid[-2], mid[-1]]
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fig, axs = plt.subplots(2, 3, figsize=(7.28, 4.4))
    for axi, f, t in zip(axs.ravel(), pick, tags):
        panel(axi, f, t)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    save(fig, out / "fig15_qualitative")
    print("wrote figs 10-15 to", out)


if __name__ == "__main__":
    main()
