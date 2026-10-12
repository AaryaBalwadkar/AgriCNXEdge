"""
phase3_schematics.py
====================
Phase 3 — Q1-grade method schematics for the Agri-CNX-Edge paper.
Every label is a verifiable code fact (sources noted per figure);
nothing is invented and no data is plotted.

USAGE:
    python phase3_schematics.py
    python phase3_schematics.py --out paper_figures

OUTPUTS (in <out>/):
    fig01_architecture.{pdf,tif,png}   Fig 1: end-to-end system
    fig03_routing.{pdf,tif,png}        Fig 3: hierarchical feature routing
    fig04_loss_training.{pdf,tif,png}  Fig 4: losses + training schedule
    phase3_captions.txt                draft self-contained captions

STYLE (matches Phase 1 + Elsevier artwork policy):
    Arial, >=7pt at final size, vector PDF (fonts embedded Type 42) +
    TIFF LZW @1000dpi at 185mm double-column width, Okabe-Ito accents,
    left-to-right data flow, dashed = training-only supervision,
    grayscale-readable (meaning carried by labels, not color).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DOUBLE_W = 7.28  # 185 mm Elsevier double column

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermilion": "#D55E00", "purple": "#CC79A7", "grey": "#666666",
      "light": "#F2F2F2", "ink": "#222222"}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": False,
    })


def new_canvas(h=5.4):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), fc="white", ec=OI["ink"],
        lw=0.9, tsize=7.5, lsize=7.0, dashed=False):
    """Title (+ optional lines) stacked top-down; caller must size h:
    h >= 5 (title only), 12.5 (1 line), 18.5 (2 lines), 30 (4 lines)."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=(0, (4, 2)) if dashed else "-"))
    cx = x + w / 2
    if lines:
        ty = y + h - 3.0
        ax.text(cx, ty, title, ha="center", va="top", fontsize=tsize,
                fontweight="bold", color=OI["ink"])
        yy = ty - 6.2
        for ln in lines:
            ax.text(cx, yy, ln, ha="center", va="top", fontsize=lsize,
                    color=OI["ink"])
            yy -= 5.8
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=OI["ink"])


def arrow(ax, x0, y0, x1, y1, color=OI["ink"], lw=1.0, dashed=False,
          label=None, lsize=6.5, dy=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=lw, shrinkA=1.0, shrinkB=1.4,
                                linestyle=(0, (4, 2)) if dashed else "-"))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center",
                va="bottom", fontsize=lsize, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))


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


# --------------------------------------------------------------------------
# Fig 1 — end-to-end architecture.
# SOURCES: student_config.py (512 input, classes 4/4/3/1, head dims),
# student_backbone.py (timm features_only, frozen [0,1], LoRA r=8/a=16
# on mlp.fc1-fc2 of stages [2,3]), student_model.py (FPN-Lite f3+f4->P3
# [128,32,32]; heads; decode conf 0.3/NMS 0.45; ROI 7x7; ONNX opset 17),
# student_loss.py (alpha .5/gamma 2, lambdas 7.5/1.0/.5, beta .5,
# homoscedastic), student_train.py (Phase1 5ep, lrs 1e-5/1e-4, accum x4),
# teacher_config.py + run_*_teacher.py (ConvNeXt-Small <- 120-class PDDD,
# 224px, seeds {42,123,456} x 8 TTA, T=1 probs), run_yield_teacher.py
# (YOLO26s, 640px, LAB-CLAHE, conf 0.20).
# --------------------------------------------------------------------------
def fig01(outdir):
    fig, ax = new_canvas(h=5.6)
    # teacher band
    box(ax, 2, 79, 96, 19,
        "TEACHERS  (training supervision only, dashed arrows)",
        ["ConvNeXt-Small: 224 px, seeds {42,123,456} x 8 TTA = 24 passes -> T=1 prob CSVs",
         "YOLO26s: 640 px, CIELAB a* CLAHE, conf 0.20 -> boxes CSV (xyxy + conf)"],
        fc=OI["light"], ec=OI["grey"], dashed=True, tsize=7.0, lsize=6.5)
    # pipeline
    box(ax, 2, 51, 12, 13, "Input", ["512 x 512 x 3"])
    box(ax, 17, 42, 23, 30, "DINOv3-ConvNeXt-Tiny",
        ["timm, features_only", "f1[96,128,128]  f2[192,64,64]",
         "f3[384,32,32]  f4[768,16,16]", "stg 0-1 frozen; stg 2-3 LoRA r8 a16"],
        fc="#E8F1FA", ec=OI["blue"], tsize=7.0, lsize=6.0)
    box(ax, 43, 44, 17, 13, "FPN-Lite (f3+f4)", ["P3 [128, 32, 32]"],
        fc="#FFF4E0", ec=OI["orange"], tsize=7.0, lsize=6.5)
    box(ax, 62, 66, 14, 8, "LeafHead (f1-4)", [], fc="#E9F5EC",
        ec=OI["green"], tsize=6.5)
    box(ax, 62, 56, 14, 8, "PestHead (f2-4)", [], fc="#E9F5EC",
        ec=OI["green"], tsize=6.5)
    box(ax, 62, 46, 14, 8, "DetHead (P3 grid)", [], fc="#E9F5EC",
        ec=OI["green"], tsize=6.5)
    box(ax, 62, 36, 14, 8, "FruitHead (ROI-3)", [], fc="#E9F5EC",
        ec=OI["green"], tsize=6.5)
    box(ax, 79, 66, 19, 8, "0.5 Focal + 0.5 KL", [], fc="#FDECEA",
        ec=OI["vermilion"], tsize=6.5)
    box(ax, 79, 56, 19, 8, "0.5 Focal + 0.5 KL", [], fc="#FDECEA",
        ec=OI["vermilion"], tsize=6.5)
    box(ax, 79, 46, 19, 8, "CIoU + obj/cls + 0.5t", [], fc="#FDECEA",
        ec=OI["vermilion"], tsize=6.5)
    box(ax, 79, 36, 19, 8, "0.5 Focal + 0.5 KL", [], fc="#FDECEA",
        ec=OI["vermilion"], tsize=6.5)
    box(ax, 63, 19, 35, 13, "Homoscedastic weighter -> L_total",
        ["per task: e^(-s) L + 0.5 s"], fc="white", ec=OI["purple"])
    box(ax, 2, 19, 38, 13, "Two-phase training",
        ["ep1-5 LoRA frozen; ep6+ LoRA lr 1e-5"], fc="white", ec=OI["ink"],
        tsize=7.0, lsize=6.5)
    box(ax, 2, 10, 96, 8,
        "Deployment: ONNX opset 17 (leaf logits, pest logits, raw_det; fruit via client-side ROI-Align)",
        [], fc=OI["light"], ec=OI["grey"], tsize=7.0)
    # data-flow arrows
    arrow(ax, 14, 57.5, 17, 57.5)                   # input -> backbone
    arrow(ax, 40, 70, 62, 70, label="f1")           # f1 tap over neck
    arrow(ax, 40, 60, 62, 60, label="f2")           # f2 tap over neck
    arrow(ax, 40, 52, 43, 52)                       # into neck (label in box title)
    arrow(ax, 60, 52, 62, 50)                       # neck -> DetHead
    arrow(ax, 73, 46, 73, 44, label="boxes", dy=0.6)  # Det -> Fruit
    arrow(ax, 51, 44, 62, 40, label="P3 crops", dy=-0.6)   # neck -> Fruit
    for ym in (70, 60, 50, 40):                     # heads -> losses
        arrow(ax, 76, ym, 79, ym)
    arrow(ax, 88.5, 36, 88.5, 32)                   # losses -> weighter
    # collector: leaf/pest/det losses join weighter around the right
    for ym in (70, 60, 50):
        ax.plot([98, 99.2], [ym, ym], color=OI["vermilion"], linewidth=0.9)
    ax.plot([99.2, 99.2], [50, 28], color=OI["vermilion"], linewidth=0.9)
    arrow(ax, 99.2, 32, 98, 28, color=OI["vermilion"])
    # teacher supervision (dashed) into loss column
    arrow(ax, 88.5, 79, 88.5, 74, color=OI["grey"], dashed=True,
          label="soft supervision (all 4 losses)", dy=1.0)
    ax.text(2, 3, "Solid = data flow      Dashed = training-only teacher supervision",
            fontsize=7.0, style="italic", va="top", ha="left")
    save_all(fig, outdir, "fig01_architecture")


# --------------------------------------------------------------------------
# Fig 3 — hierarchical feature routing close-up.
# SOURCES: student_backbone.py (stage shapes, frozen/LoRA split),
# student_model.py (FPNLite, head in_ch, decode conf/NMS, ROI params).
# --------------------------------------------------------------------------
def fig03(outdir):
    fig, ax = new_canvas(h=4.6)
    ax.text(4, 95, "DINOv3-ConvNeXt-Tiny backbone (timm, input 512x512x3)",
            fontsize=8, fontweight="bold", va="top", ha="left")
    stages = [("f1 [96,128,128]", "frozen", 71),
              ("f2 [192,64,64]", "frozen", 57),
              ("f3 [384,32,32]", "LoRA r=8", 43),
              ("f4 [768,16,16]", "LoRA r=8", 29)]
    for title, badge, y in stages:
        fc = "#EDEDED" if badge == "frozen" else "#E8F1FA"
        ec = OI["grey"] if badge == "frozen" else OI["blue"]
        box(ax, 4, y, 34, 13, title, [badge], fc=fc, ec=ec,
            tsize=7.5, lsize=7.0)
    box(ax, 44, 42, 18, 13, "FPN-Lite",
        ["f3+f4 -> P3 [128,32,32]"], fc="#FFF4E0", ec=OI["orange"])
    box(ax, 68, 72, 28, 10, "LeafHead: f1 -> 4", [],
        fc="#E9F5EC", ec=OI["green"], tsize=7.0)
    box(ax, 68, 58, 28, 10, "PestHead: f2 -> 4", [],
        fc="#E9F5EC", ec=OI["green"], tsize=7.0)
    box(ax, 68, 46, 28, 10, "DetHead: P3 -> [6,32,32]", [],
        fc="#E9F5EC", ec=OI["green"], tsize=7.0)
    box(ax, 68, 32, 28, 10, "FruitHead: ROI 7x7 -> 3", [],
        fc="#E9F5EC", ec=OI["green"], tsize=7.0)
    ax.text(82, 29, "GT boxes (train) / decoded (infer)", fontsize=6.5,
            ha="center", va="top", style="italic")
    ax.text(82, 24, "conf 0.3, NMS IoU 0.45", fontsize=6.5, ha="center",
            va="top", style="italic")
    arrow(ax, 38, 77.5, 68, 77)          # f1 -> Leaf
    arrow(ax, 38, 63.5, 68, 63)          # f2 -> Pest
    arrow(ax, 38, 49.5, 44, 49.5)        # f3 -> neck
    arrow(ax, 38, 35.5, 44, 44)          # f4 -> neck
    arrow(ax, 62, 48, 68, 51)            # neck -> Det
    arrow(ax, 82, 46, 82, 42)            # Det -> Fruit (boxes)
    arrow(ax, 53, 42, 68, 36)            # P3 -> Fruit (crops)
    ax.text(2, 6, "Stages 0-1 fully frozen; stages 2-3 train only via LoRA adapters (mlp.fc1/fc2).",
            fontsize=7.0, style="italic", va="top", ha="left")
    save_all(fig, outdir, "fig03_routing")


# --------------------------------------------------------------------------
# Fig 4 — losses + training schedule.
# SOURCES: student_loss.py (all loss terms/coefficients, weighter maths,
# ignore_index -1), student_train.py (Phase1/Phase2 LRs, cosine, accum x4,
# clip 1.0, patience 20, best-val selection, ONNX export), run_student.py
# (eff. batch 32), training_history.json (best epochs 186/194).
# --------------------------------------------------------------------------
def fig04(outdir):
    fig, ax = new_canvas(h=4.8)
    ax.text(2, 96, "(a) Per-task losses", fontsize=8.5, fontweight="bold",
            va="top", ha="left")
    ax.text(58, 96, "(b) Training schedule", fontsize=8.5, fontweight="bold",
            va="top", ha="left")
    ax.plot([57, 57], [4, 92], color=OI["grey"], linewidth=0.7)
    branches = [
        ("L_leaf = 0.5Focal(g2) + 0.5KL", ["hard labels; T=1 teacher probs; mask -1"]),
        ("L_pest = 0.5Focal(g2) + 0.5KL", ["hard labels; T=1 teacher probs; mask -1"]),
        ("L_yield = CIoU7.5 + obj1.0 + cls0.5",
         ["GT branch + 0.5 x teacher branch"]),
        ("L_fruit = 0.5Focal(g2) + 0.5KL", ["per-ROI GT-box labels; mask -1"]),
    ]
    y = 88
    for title, lines in branches:
        box(ax, 2, y - 13, 51, 13, title, lines, fc="#FDECEA",
            ec=OI["vermilion"], tsize=7.0, lsize=6.5)
        y -= 16
    box(ax, 2, 10, 51, 13, "L_total = sum [ e^(-s) L + 0.5 s ]",
        ["tasks {leaf, pest, fruit, yield}; s learned, init 0"], fc="white",
        ec=OI["purple"], tsize=7.5, lsize=6.5)
    arrow(ax, 27.5, 27, 27.5, 23)
    steps = [
        ("Phase 1: ep 1-5", ["LoRA frozen; heads+neck lr 1e-4"]),
        ("Phase 2: ep 6-200", ["LoRA 1e-5; cosine; accum x4; clip 1.0; pat 20"]),
        ("Best-val select", ["ep 186 LoRA / ep 194 frozen"]),
        ("Export + eval", ["ONNX opset 17; CMs; MAE/RMSE"]),
    ]
    y = 88
    for title, lines in steps:
        box(ax, 60, y - 13, 38, 13, title, lines, fc="white", ec=OI["ink"],
            tsize=7.5, lsize=6.5)
        if y < 88:
            arrow(ax, 79, y + 3, 79, y)
        y -= 16
    save_all(fig, outdir, "fig04_loss_training")


CAPTIONS = """\
Fig. 1. Overall Agri-CNX-Edge architecture. A 512x512 image passes once through the shared DINOv3-ConvNeXt-Tiny backbone (stages 0-1 frozen, stages 2-3 LoRA-adapted). FPN-Lite fuses f3+f4 into P3; f1 feeds leaf, f2 pest, P3 detection, and ROI-Align crops of P3 per-apple fruit classification. Training-only teachers (dashed) supply T=1 soft probabilities (classification) and boxes (detection). Task losses combine via learned homoscedastic weighting; the trained student exports to ONNX (opset 17).

Fig. 3. Hierarchical feature routing. Fine-grained f1 serves leaf texture, mid-scale f2 pest morphology, fused P3 apple detection (confidence 0.3, NMS IoU 0.45), and ROI-Align 7x7 crops of P3 per-apple fruit classification from ground-truth boxes (train) or decoded boxes (inference).

Fig. 4. (a) Per-task losses with implementation coefficients and (b) two-phase training schedule with best-validation-epoch selection and ONNX export.
"""


def main():
    ap = argparse.ArgumentParser(description="Phase 3 schematic generation")
    ap.add_argument("--out", default="paper_figures")
    args = ap.parse_args()
    apply_style()
    outdir = ROOT / args.out
    print("Phase 3: schematic generation (no data, code facts only)")
    print("=" * 64)
    fig01(outdir)
    fig03(outdir)
    fig04(outdir)
    with open(outdir / "phase3_captions.txt", "w", encoding="utf-8") as f:
        f.write(CAPTIONS)
    print(f"Done. Files in {outdir}")


if __name__ == "__main__":
    main()
