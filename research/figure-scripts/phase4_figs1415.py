"""Rebuild fig14 (green GT-vs-pred) + fig15 (photo + confidence maps).

Q1 grammar: panel letters + GT/pred count labels only (data, no
captions). Confidence maps are detector confidence (max obj x cls per
32x32 cell), Blues colormap -- an honest spatial diagnostic, not
backbone PCA features.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
GOV = ROOT / "paper_figures" / "green_ov"
OUT = ROOT / "paper_figures"
YOLO_LBL = ROOT / "YOLO26s apple object detection model 1" / "data" / "yield" / "labels" / "val"


def gt_of(stem):
    n = 0
    p = YOLO_LBL / (stem + ".txt")
    if p.exists():
        for line in open(p, encoding="utf-8", errors="ignore"):
            if len(line.split()) >= 5:
                n += 1
    return n


def pred_of(stem):
    import json
    recs = {r["image"].split(".")[0]: r["pred"] for r in
            json.load(open(GOV / "green_overlay_data.json"))}
    return recs.get(stem, -1)


def panel_photo(ax, stem, tag):
    im = Image.open(GOV / f"{stem}.png").convert("RGB")
    ax.imshow(im)
    ax.axis("off")
    ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", color="white", va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.55, pad=1.5,
                      edgecolor="none", boxstyle="round,pad=0.25"))
    ax.text(0.98, 0.04, f"GT {gt_of(stem)} / pred {pred_of(stem)}",
            transform=ax.transAxes, fontsize=7.0, color="white",
            va="bottom", ha="right",
            bbox=dict(facecolor="black", alpha=0.55, pad=1.5,
                      edgecolor="none", boxstyle="round,pad=0.25"))


def panel_conf(ax, stem, tag):
    g = np.asarray(Image.open(GOV / f"{stem}_conf.png").convert("L"),
                   dtype=float) / 255.0
    ax.imshow(g, cmap="Blues", vmin=0, vmax=1)
    ax.axis("off")
    ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", color="black", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, pad=1.5,
                      edgecolor="0.6", boxstyle="round,pad=0.25"))
    ax.text(0.98, 0.04, f"max {g.max():.2f}", transform=ax.transAxes,
            fontsize=7.0, color="black", va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.75, pad=1.5,
                      edgecolor="0.6", boxstyle="round,pad=0.25"))


def save(fig, base):
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=600)
    fig.savefig(str(base) + ".tif", dpi=1000, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

# Fig 14: 4 green GT-vs-pred (sparse -> dense)
picks14 = ["apple_03704", "apple_00283", "apple_00795", "apple_00069"]
tags14 = ["(a)", "(b)", "(c)", "(d)"]
fig, axs = plt.subplots(2, 2, figsize=(7.28, 4.6))
for axi, s, t in zip(axs.ravel(), picks14, tags14):
    panel_photo(axi, s, t)
fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
save(fig, OUT / "fig14_boxes")

# Fig 15: 3 green photos over their confidence maps
picks15 = ["apple_03704", "apple_03427", "apple_04063"]
fig, axs = plt.subplots(2, 3, figsize=(7.28, 3.6))
for j, s in enumerate(picks15):
    panel_photo(axs[0][j], s, f"({chr(97 + j)})")
    panel_conf(axs[1][j], s, f"({chr(100 + j)})")
fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
save(fig, OUT / "fig15_qualitative")
print("wrote fig14/fig15")
