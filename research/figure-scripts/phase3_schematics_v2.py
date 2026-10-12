"""
phase3_schematics_v2.py
=======================
Phase 3 REBUILD — schematics in the reference-figure grammar
(ViT Fig 1 / DINO Fig 2 / DINOv3 Fig 12):
  * ONE idea per figure (forward trace / KD loop / training timeline)
  * REAL data traced (val image apple_00784 + its 46 GT boxes)
  * pastel fills, hairline strokes, serif + real mathtext, no hatches
  * captions carry the takeaway; single-seed honesty where relevant

USAGE:
    python phase3_schematics_v2.py
    python phase3_schematics_v2.py --out paper_figures

Overwrites fig01/fig03/fig04.{pdf,tif,png} (v1 in paper_figures/v1/)
and writes phase3_captions.txt.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

ROOT = Path(__file__).resolve().parent
DOUBLE_W = 7.28

TRACE_IMG = (ROOT / "YOLO26s apple object detection model 1" / "data"
             / "yield" / "images" / "val" / "apple_00784.jpg")
TRACE_LBL = (ROOT / "YOLO26s apple object detection model 1" / "data"
             / "yield" / "labels" / "val" / "apple_00784.txt")

PASTEL = {"blue": "#DCE9F7", "orange": "#F7EBD2", "green": "#D9EDDF",
          "pink": "#F6DEDE", "grey": "#EFEFEF", "white": "#FFFFFF"}


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": False,
        "mathtext.fontset": "cm",
    })


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


def arrow(ax, x0, y0, x1, y1, color="black", lw=1.0, dashed=False,
          label=None, lsize=7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=lw, shrinkA=1.0, shrinkB=1.4,
                                linestyle=(0, (4, 2)) if dashed else "-"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 2.0, label, ha="center",
                va="bottom", fontsize=lsize, color=color, style="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))


def read_boxes():
    boxes = []
    with open(TRACE_LBL, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                boxes.append([float(v) for v in p[1:5]])
    return np.array(boxes, float)


# ---------------------------------------------------------------- fig01 ---
def fig01(outdir):
    img = np.array(Image.open(TRACE_IMG).convert("RGB"))
    H, W, _ = img.shape
    gt = read_boxes()
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # input image with patch grid
    a0 = inset_axes(ax, width="17%", height="40%", loc="lower left",
                    bbox_to_anchor=(0.005, 0.32, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0)
    a0.imshow(img)
    a0.set_xticks(np.linspace(0, W, 9))
    a0.set_yticks(np.linspace(0, H, 9))
    a0.grid(color="white", linewidth=0.5, alpha=0.7)
    a0.tick_params(length=0)
    a0.set_xticklabels([])
    a0.set_yticklabels([])
    for s in a0.spines.values():
        s.set_edgecolor("black")
        s.set_linewidth(0.8)
    ax.text(9.5, 24, "Input 512x512x3", ha="center", fontsize=7.5,
            fontweight="bold")
    ax.text(9.5, 19, "val img apple_00784", ha="center", fontsize=6.5,
            style="italic")

    # backbone stage strips
    stages = [("$f_1$ [96, 128, 128]  frozen", PASTEL["grey"], 78),
              ("$f_2$ [192, 64, 64]  frozen", PASTEL["grey"], 65),
              ("$f_3$ [384, 32, 32]  LoRA", PASTEL["blue"], 52),
              ("$f_4$ [768, 16, 16]  LoRA", PASTEL["blue"], 39)]
    for title, fc, y in stages:
        ax.add_patch(FancyBboxPatch((24, y - 5.5), 20, 11,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=fc, edgecolor="black",
                                    linewidth=0.7))
        ax.text(34, y, title, ha="center", va="center", fontsize=7)
    ax.text(34, 88, "DINOv3-ConvNeXt-Tiny", ha="center", fontsize=8,
            fontweight="bold")

    # neck + heads
    ax.add_patch(FancyBboxPatch((47, 46), 12, 11,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor=PASTEL["orange"], edgecolor="black",
                                linewidth=0.7))
    ax.text(53, 51.5, "$P_3$", ha="center", fontsize=7.5,
            fontweight="bold")
    ax.text(53, 43.8, "FPN($f_3{+}f_4$)", ha="center", fontsize=6.5,
            style="italic")
    heads = [("Leaf $\\leftarrow f_1$ : 4", 78),
             ("Pest $\\leftarrow f_2$ : 4", 65),
             ("Detect $\\leftarrow P_3$", 52),
             ("Fruit $\\leftarrow$ ROI($P_3$)", 39)]
    for title, y in heads:
        ax.add_patch(FancyBboxPatch((62, y - 5.5), 16, 11,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=PASTEL["green"],
                                    edgecolor="black", linewidth=0.7))
        ax.text(70, y, title, ha="center", va="center", fontsize=7)
    ax.text(70, 88, "Task heads", ha="center", fontsize=8,
            fontweight="bold")

    # output image with GT boxes
    a1 = inset_axes(ax, width="17%", height="40%", loc="lower left",
                    bbox_to_anchor=(0.825, 0.32, 1, 1),
                    bbox_transform=ax.transAxes, borderpad=0)
    a1.imshow(img)
    for (cx, cy, w, h) in gt:
        a1.add_patch(Rectangle((cx * W - w * W / 2, cy * H - h * H / 2),
                               w * W, h * H, linewidth=1.0,
                               edgecolor="#2CA02C", facecolor="none"))
    a1.set_xticklabels([])
    a1.set_yticklabels([])
    a1.set_xticks([])
    a1.set_yticks([])
    for s in a1.spines.values():
        s.set_edgecolor("black")
        s.set_linewidth(0.8)
    ax.text(90, 24, f"Output: {len(gt)} GT boxes", ha="center", fontsize=7.5,
            fontweight="bold")
    ax.text(90, 19, "(illustrative example)", ha="center", fontsize=6.5,
            style="italic")

    # arrows (main coords)
    arrow(ax, 20, 58, 24, 58)
    arrow(ax, 44, 78, 62, 78)
    arrow(ax, 44, 65, 62, 65)
    arrow(ax, 44, 52, 47, 51)
    arrow(ax, 44, 39, 47, 48)
    arrow(ax, 59, 51, 62, 52)
    arrow(ax, 70, 46.5, 70, 44.5)
    arrow(ax, 53, 46, 62, 40)
    ax.text(50, 10, "Teacher supervision, losses and training schedule in Fig. 4.",
            ha="center", fontsize=7, style="italic")
    save_all(fig, outdir, "fig01_architecture")


# ---------------------------------------------------------------- fig03 ---
def fig03(outdir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(4, 94, "Hierarchical feature routing", fontsize=9,
            fontweight="bold", va="top", ha="left")
    stages = [("$f_1$ [96, 128, 128]", "stages 0-1 · frozen", 76,
               PASTEL["grey"]),
              ("$f_2$ [192, 64, 64]", "stages 0-1 · frozen", 60,
               PASTEL["grey"]),
              ("$f_3$ [384, 32, 32]", "stages 2-3 · LoRA r=8", 44,
               PASTEL["blue"]),
              ("$f_4$ [768, 16, 16]", "stages 2-3 · LoRA r=8", 28,
               PASTEL["blue"])]
    for title, badge, y, fc in stages:
        ax.add_patch(FancyBboxPatch((4, y - 6), 32, 12,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=fc, edgecolor="black",
                                    linewidth=0.7))
        ax.text(20, y + 1.2, title, ha="center", va="center", fontsize=7.5)
        ax.text(20, y - 3.4, badge, ha="center", va="center", fontsize=6.5,
                style="italic", color="#333333")
    ax.add_patch(FancyBboxPatch((42, 44), 18, 13,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor=PASTEL["orange"], edgecolor="black",
                                linewidth=0.7))
    ax.text(51, 52.5, "FPN-Lite", ha="center", fontsize=7.5,
            fontweight="bold")
    ax.text(51, 47.5, "$f_3{+}f_4{\\rightarrow}P_3$ [128, 32, 32]",
            ha="center", fontsize=7)
    heads = [("LeafHead: $f_1 \\rightarrow$ 4 classes", 76),
             ("PestHead: $f_2 \\rightarrow$ 4 classes", 60),
             ("DetHead: $P_3 \\rightarrow$ [6, 32, 32]", 46),
             ("FruitHead: ROI($P_3$, 7x7) $\\rightarrow$ 3", 30)]
    for title, y in heads:
        ax.add_patch(FancyBboxPatch((66, y - 6), 30, 12,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=PASTEL["green"],
                                    edgecolor="black", linewidth=0.7))
        ax.text(81, y, title, ha="center", va="center", fontsize=7.5)
    ax.text(81, 18, "GT boxes (train) / decoded boxes (infer); conf 0.3, NMS 0.45",
            ha="center", fontsize=6.5, style="italic")
    arrow(ax, 36, 76, 66, 76)
    arrow(ax, 36, 60, 66, 60)
    arrow(ax, 36, 44, 42, 48)
    arrow(ax, 36, 28, 42, 46)
    arrow(ax, 60, 50, 66, 49)
    arrow(ax, 68.5, 46, 68.5, 42)
    arrow(ax, 51, 44, 70, 34)
    save_all(fig, outdir, "fig03_routing")


# ---------------------------------------------------------------- fig04 ---
def fig04(outdir):
    fig = plt.figure(figsize=(DOUBLE_W, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    # (a) minimal KD loop, DINO-Fig-2 style
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(2, 96, "(a) Heterogeneous KD (one classification head)",
            fontsize=8.5, fontweight="bold", va="top", ha="left")
    for cx, lab in [(25, "student"), (75, "teacher\n(CSV, T=1)")]:
        ax.add_patch(FancyBboxPatch((cx - 13, 48), 26, 14,
                                    boxstyle="round,pad=0.3,rounding_size=1.0",
                                    facecolor=PASTEL["white"],
                                    edgecolor="black", linewidth=0.8))
        ax.text(cx, 55, lab, ha="center", fontsize=7.5)
    for cx, lab in [(25, "softmax"), (75, "softmax")]:
        ax.add_patch(FancyBboxPatch((cx - 10, 68), 20, 9,
                                    boxstyle="round,pad=0.2,rounding_size=0.8",
                                    facecolor=PASTEL["grey"],
                                    edgecolor="black", linewidth=0.7))
        ax.text(cx, 72.5, lab, ha="center", fontsize=7)
    for cx, lab in [(25, "$p_1$"), (75, "$p_2$")]:
        ax.add_patch(plt.Circle((cx, 86), 5, facecolor="white",
                                edgecolor="black", linewidth=0.8))
        ax.text(cx, 86, lab, ha="center", va="center", fontsize=8)
    ax.text(50, 90, "loss: $-p_2 \\log p_1$", ha="center", fontsize=7.5)
    ax.text(50, 40, "$0.5$ Focal + $0.5$ KL; mask $-1$", ha="center",
            fontsize=7, style="italic")
    ax.add_patch(FancyBboxPatch((8, 14), 84, 16,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor=PASTEL["pink"], edgecolor="black",
                                linewidth=0.7))
    ax.text(50, 25, "Detection: GT + teacher boxes $\\rightarrow$ cell",
            ha="center", fontsize=7.5)
    ax.text(50, 19, "CIoU$\\times$7.5 + obj/cls; +$0.5\\times$teacher",
            ha="center", fontsize=7)
    arrow(ax, 25, 62, 25, 68)
    arrow(ax, 25, 77, 25, 81)
    arrow(ax, 75, 62, 75, 68)
    arrow(ax, 75, 77, 75, 81)
    # (b) training timeline, DINOv3-Fig-12 style
    bx = fig.add_subplot(gs[1])
    bx.set_xlim(0, 200)
    bx.set_ylim(0, 100)
    bx.axis("off")
    bx.text(0, 96, "(b) Training timeline (epochs)", fontsize=8.5,
            fontweight="bold", va="top", ha="left")
    bx.add_patch(FancyBboxPatch((1, 62), 4, 20, facecolor="#DCE9F7",
                                edgecolor="black", linewidth=0.7))
    bx.add_patch(FancyBboxPatch((5, 62), 194, 20, facecolor="#D9EDDF",
                                edgecolor="black", linewidth=0.7))
    bx.text(3, 72, "Ph1", ha="center", fontsize=7, fontweight="bold",
            rotation=90, va="center")
    bx.text(102, 73, "Phase 2: LoRA lr $10^{-5}$, cosine, accum $\\times$4 "
            "(eff. 32), clip 1.0, pat. 20", ha="center", fontsize=7)
    for t in [0, 5, 50, 100, 150, 200]:
        bx.text(t, 56, str(t), ha="center", fontsize=7)
    bx.text(100, 47, "Phase 1 (ep 1-5): LoRA frozen, heads+neck lr $10^{-4}$",
            ha="center", fontsize=6.5, style="italic")
    bx.plot([186, 186], [62, 86], color="#0072B2", linestyle=":",
            linewidth=1.2)
    bx.text(186, 89, "best LoRA: ep 186", ha="center", fontsize=7,
            color="#0072B2")
    bx.plot([194, 194], [62, 86], color="#555555", linestyle=":",
            linewidth=1.2)
    bx.text(194, 50, "best frozen: ep 194", ha="center", fontsize=7,
            color="#555555")
    bx.add_patch(FancyBboxPatch((100, 22), 99, 16,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor=PASTEL["grey"], edgecolor="black",
                                linewidth=0.7))
    bx.text(149.5, 30, "Export: ONNX opset 17; CMs + MAE/RMSE",
            ha="center", fontsize=6.5)
    arrow(bx, 199, 62, 199, 38)
    save_all(fig, outdir, "fig04_loss_training")


CAPTIONS = """\
Fig. 1. Student forward pass traced on a real validation image (apple_00784, 46 ground-truth boxes shown for illustration). The shared DINOv3-ConvNeXt-Tiny backbone routes fine-grained f1 to leaf, mid-scale f2 to pest, and fused P3 to detection, with ROI-Align crops of P3 feeding per-apple fruit classification. Training-only teacher supervision and losses are detailed in Fig. 4. Single-seed model, best-validation epoch 186.

Fig. 3. Hierarchical feature routing. Stage shapes and frozen/LoRA status are as implemented; detection decoding uses confidence 0.3 and NMS IoU 0.45, with ground-truth boxes at train time and decoded boxes at inference.

Fig. 4. (a) Heterogeneous knowledge distillation for one classification head, in the DINO two-tower style: teacher soft targets (T=1 CSVs) meet student logits in KL divergence alongside focal loss on hard labels; detection distills YOLO teacher boxes with a 0.5-weighted branch. (b) Two-phase training timeline with best-validation epochs and ONNX export. Single seed (n=1); early stopping patience 20.
"""


def main():
    ap = argparse.ArgumentParser(description="Phase 3 v2 schematics")
    ap.add_argument("--out", default="paper_figures")
    args = ap.parse_args()
    apply_style()
    outdir = ROOT / args.out
    for p, n in [(TRACE_IMG, "trace image"), (TRACE_LBL, "trace labels")]:
        if not p.exists():
            raise FileNotFoundError(f"missing {n}: {p}")
    print("Phase 3 v2 (reference grammar):")
    print("=" * 64)
    fig01(outdir)
    fig03(outdir)
    fig04(outdir)
    with open(outdir / "phase3_captions.txt", "w", encoding="utf-8") as f:
        f.write(CAPTIONS)
    print(f"Done. Files in {outdir}")


if __name__ == "__main__":
    main()
