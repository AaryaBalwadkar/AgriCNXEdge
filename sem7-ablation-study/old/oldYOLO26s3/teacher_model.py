"""
teacher_model.py  (FIXED — v4)
================================
FIXES OVER v3:
  - Rule 2 corrected: downsample_layers.{i} → stages.{i}.downsample  (NOT i-1)
    v3 was mapping i=1 → stages.0.downsample which doesn't exist in new timm,
    causing all inter-stage downsample weights to silently not load.
  - Auto-detects backbone variant and head classes from checkpoint itself,
    so backbone_name and pddd_num_classes in config are ignored safely.

PDDD checkpoint (convnext_small-model-99.pth):
  - Architecture: ConvNeXt-Small (stem out_channels=96 → detected as convnext_small)
  - Classes: 120 (not 40 as originally assumed)
  - All 344 keys remapped; ~340 backbone tensors should load successfully.

Download:
  https://zenodo.org/records/7890438/files/convnext_small-model-99.pth?download=1
"""

import os
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm not found. Run: pip install timm")


# ===========================================================================
#  STEP 1 — KEY REMAPPING (old timm naming → new timm naming)
# ===========================================================================

def remap_pddd_keys(old_sd: Dict) -> Dict:
    """
    Converts PDDD checkpoint keys from old timm ConvNeXt naming to new naming.

    FULL MAPPING TABLE:
    ┌────────────────────────────────────┬─────────────────────────────────────┐
    │ Old timm (PDDD checkpoint)         │ New timm (current)                  │
    ├────────────────────────────────────┼─────────────────────────────────────┤
    │ downsample_layers.0.0.*            │ stem.0.*                            │
    │ downsample_layers.0.1.*            │ stem.1.*                            │
    │ downsample_layers.{i}.0.*  (i≥1)  │ stages.{i}.downsample.0.*           │
    │ downsample_layers.{i}.1.*  (i≥1)  │ stages.{i}.downsample.1.*           │
    │ stages.{i}.{j}.dwconv.*           │ stages.{i}.blocks.{j}.conv_dw.*     │
    │ stages.{i}.{j}.pwconv1.*          │ stages.{i}.blocks.{j}.mlp.fc1.*     │
    │ stages.{i}.{j}.pwconv2.*          │ stages.{i}.blocks.{j}.mlp.fc2.*     │
    │ stages.{i}.{j}.norm.*             │ stages.{i}.blocks.{j}.norm.*        │
    │ stages.{i}.{j}.gamma              │ stages.{i}.blocks.{j}.gamma         │
    │ norm.weight / norm.bias           │ head.norm.weight / head.norm.bias   │
    │ head.weight / head.bias           │ head.fc.weight / head.fc.bias       │
    └────────────────────────────────────┴─────────────────────────────────────┘
    """
    new_sd = {}
    n_renamed = 0

    for old_key, value in old_sd.items():
        k = old_key

        # Rule 1: Stem  (downsample_layers.0.X → stem.X)
        k = re.sub(r'^downsample_layers\.0\.(\d+)(.*)', r'stem.\1\2', k)

        # Rule 2: Inter-stage downsamples  (downsample_layers.{i}.X → stages.{i}.downsample.X)
        # CRITICAL FIX: use stages.{i}, NOT stages.{i-1}
        # i=1 → stages.1.downsample (transition into stage 1, normalising 96-ch stage-0 output)
        # i=2 → stages.2.downsample (transition into stage 2, normalising 192-ch)
        # i=3 → stages.3.downsample (transition into stage 3, normalising 384-ch)
        def _ds(m):
            i    = int(m.group(1))   # keep same index — do NOT subtract 1
            x    = m.group(2)        # sub-layer index (0=norm, 1=conv)
            rest = m.group(3)        # .weight / .bias etc.
            return f'stages.{i}.downsample.{x}{rest}'

        k = re.sub(r'^downsample_layers\.([1-9]\d*)\.(\d+)(.*)', _ds, k)

        # Rule 3: Stage blocks
        def _blk(m):
            suffix = m.group(3)
            suffix = re.sub(r'^dwconv\.',  'conv_dw.',  suffix)
            suffix = re.sub(r'^pwconv1\.', 'mlp.fc1.',  suffix)
            suffix = re.sub(r'^pwconv2\.', 'mlp.fc2.',  suffix)
            return f'stages.{m.group(1)}.blocks.{m.group(2)}.{suffix}'

        k = re.sub(r'^stages\.(\d+)\.(\d+)\.(.*)', _blk, k)

        # Rule 4: Final norm before head
        k = re.sub(r'^norm\.(weight|bias)$', r'head.norm.\1', k)

        # Rule 5: Head linear layer
        if k == 'head.weight': k = 'head.fc.weight'
        if k == 'head.bias':   k = 'head.fc.bias'

        if k != old_key:
            n_renamed += 1
        new_sd[k] = value

    print(f"[INFO] Key remapping: {n_renamed}/{len(old_sd)} keys renamed, "
          f"{len(old_sd) - n_renamed} unchanged.")
    return new_sd


# ===========================================================================
#  STEP 2 — SHAPE-SAFE FILTER (skips mismatched tensors instead of crashing)
# ===========================================================================

def filter_by_shape(
    model_sd: Dict[str, torch.Tensor],
    ckpt_sd:  Dict[str, torch.Tensor],
    skip_prefixes: List[str] = None,
) -> Tuple[Dict, List, List]:
    """
    Returns only checkpoint keys whose name AND shape match the model.
    Keys in skip_prefixes are always excluded (e.g. head.fc which will be replaced).
    """
    skip_prefixes = skip_prefixes or []
    compatible, shape_mismatch, missing_in_model = {}, [], []

    for k, v in ckpt_sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            continue
        if k not in model_sd:
            missing_in_model.append(k)
            continue
        if model_sd[k].shape == v.shape:
            compatible[k] = v
        else:
            shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    return compatible, shape_mismatch, missing_in_model


# ===========================================================================
#  STEP 3 — AUTO-DETECT CHECKPOINT ARCHITECTURE
# ===========================================================================

def detect_variant_and_classes(sd: Dict[str, torch.Tensor]) -> Tuple[str, int, int]:
    """
    Returns (timm_model_name, feature_dim, num_pddd_classes) from checkpoint.
    Looks at the stem output channels to identify the ConvNeXt variant.

    ConvNeXt channel widths by variant:
      nano:   [40, 80, 160, 320]  stem out = 40
      tiny:   [96, 192, 384, 768] stem out = 96  ← but named differently
      small:  [96, 192, 384, 768] stem out = 96
      base:  [128, 256, 512,1024] stem out = 128
      large: [192, 384, 768,1536] stem out = 192
    """
    # Find stem weight after remapping
    stem_key = next((k for k in sd if k.endswith('stem.0.weight')), None)
    out_ch = sd[stem_key].shape[0] if stem_key else None

    variant_map = {96: 'convnext_small', 128: 'convnext_base',
                   192: 'convnext_large', 40: 'convnext_nano'}
    feat_map    = {96: 768, 128: 1024, 192: 1536, 40: 256}

    if out_ch in variant_map:
        variant  = variant_map[out_ch]
        feat_dim = feat_map[out_ch]
    else:
        print(f"[WARN] Unknown stem channels={out_ch}. Defaulting to convnext_small.")
        variant, feat_dim = 'convnext_small', 768

    # Detect original head size
    head_key = next((k for k in sd if k.endswith('head.fc.weight')), None)
    n_classes = sd[head_key].shape[0] if head_key else 120

    print(f"[AUTO-DETECT] Variant: {variant} | Stem out_ch: {out_ch} | "
          f"Feature dim: {feat_dim} | PDDD classes: {n_classes}")
    return variant, feat_dim, n_classes


# ===========================================================================
#  FOCAL LOSS
# ===========================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -1,
                 class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = self.ce(logits, targets)
        pt   = torch.exp(-ce)
        fl   = (1.0 - pt) ** self.gamma * ce
        mask = (targets != self.ignore_index).float()
        return (fl * mask).sum() / mask.sum().clamp(min=1.0)


# ===========================================================================
#  PDDD TEACHER MODEL
# ===========================================================================

class PDDDTeacher(nn.Module):
    """
    Self-configuring PDDD teacher.
    Auto-detects backbone variant and original class count from the checkpoint.
    backbone_name and pddd_num_classes in config are only used as fallbacks.
    """

    def __init__(self, num_classes: int, pddd_weights_path: str,
                 dropout_rate: float = 0.5, drop_path_rate: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        # ── Load raw checkpoint ──────────────────────────────────────────────
        if not os.path.exists(pddd_weights_path):
            raise FileNotFoundError(
                f"PDDD weights not found: {pddd_weights_path}\n"
                "Download: https://zenodo.org/records/7890438/files/"
                "convnext_small-model-99.pth?download=1"
            )
        print(f"[INFO] Loading PDDD checkpoint: {pddd_weights_path}")
        ckpt = torch.load(pddd_weights_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            raw = (ckpt.get("model") or ckpt.get("model_state_dict")
                   or ckpt.get("state_dict") or ckpt)
        else:
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
        raw = {k.replace("module.", ""): v for k, v in raw.items()}

        # ── Remap keys ───────────────────────────────────────────────────────
        print("[INFO] Remapping PDDD checkpoint keys (old timm → new timm)...")
        remapped = remap_pddd_keys(raw)

        # ── Auto-detect variant ──────────────────────────────────────────────
        variant, feat_dim, pddd_classes = detect_variant_and_classes(remapped)

        # ── Create timm model with original PDDD head size ───────────────────
        print(f"[INFO] Creating timm {variant} ({pddd_classes} PDDD classes, "
              f"drop_path_rate={drop_path_rate})...")
        self.model = timm.create_model(
            variant, pretrained=False,
            num_classes=pddd_classes, drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )

        # ── Filter to shape-compatible keys only (skip head.fc) ──────────────
        compatible, shape_mismatch, unexpected = filter_by_shape(
            model_sd=self.model.state_dict(),
            ckpt_sd=remapped,
            skip_prefixes=["head.fc."],
        )

        if shape_mismatch:
            print(f"[WARN] {len(shape_mismatch)} keys skipped (shape mismatch):")
            for key, cs, ms in shape_mismatch[:8]:
                print(f"  {key}: ckpt={cs}  model={ms}")
        if unexpected:
            print(f"[WARN] {len(unexpected)} keys not found in model: "
                  f"{unexpected[:4]}...")

        # ── Load backbone weights ─────────────────────────────────────────────
        missing, _ = self.model.load_state_dict(compatible, strict=False)
        bb_missing = [k for k in missing if not k.startswith("head.fc.")]
        if bb_missing:
            print(f"[WARN] Backbone keys still missing ({len(bb_missing)}): "
                  f"{bb_missing[:5]}")

        n_bb_total = len([k for k in self.model.state_dict()
                          if not k.startswith("head.fc.")])
        print(f"[INFO] Backbone tensors loaded: {len(compatible)}/{n_bb_total}")

        if len(compatible) >= n_bb_total * 0.95:
            print("[INFO] ✓ PDDD backbone loaded successfully (≥ 95%).")
        elif len(compatible) >= n_bb_total * 0.80:
            print("[WARN] Partial backbone load (80-95%). Downsample layers may be missing.")
        else:
            print("[ERROR] < 80% backbone loaded — something is wrong. "
                  "Check timm version: pip show timm")

        # ── Replace head ──────────────────────────────────────────────────────
        in_feat = self.model.head.fc.in_features
        print(f"[INFO] Replacing head: Linear({in_feat},{pddd_classes}) "
              f"→ Linear({in_feat},{num_classes})")
        self.model.head.fc = nn.Linear(in_feat, num_classes)
        nn.init.normal_(self.model.head.fc.weight, std=0.01)
        nn.init.zeros_(self.model.head.fc.bias)
        print(f"[INFO] ✓ Model ready. Backbone: {variant}. Head: {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_parameter_groups(self, lr_backbone, lr_head, weight_decay):
        bb, hd = [], []
        for name, p in self.named_parameters():
            (hd if "head.fc" in name else bb).append(p)
        return [
            {"params": bb, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
            {"params": hd, "lr": lr_head,     "weight_decay": 0.0,          "name": "head"},
        ]


# ===========================================================================
#  TEMPERATURE SCALING (post-hoc calibration)
# ===========================================================================

class TemperatureScaler(nn.Module):
    """
    Fits a single scalar temperature T on held-out (val) logits/labels by
    minimising NLL loss, following Guo et al. 2017 ("On Calibration of
    Modern Neural Networks").

    This does NOT change argmax predictions or accuracy -- it only reshapes
    the softmax probabilities so confidence matches empirical correctness.
    For a knowledge-distillation *teacher*, this matters a lot: the student
    is trained against softmax(teacher_logits / T_kd), so if the teacher's
    raw logits are already over/under-confident, that miscalibration is
    baked into every soft label the student learns from.

    Usage:
        scaler = TemperatureScaler()
        T = scaler.fit(val_logits, val_labels)   # val_logits: [N, C] tensor
        calibrated_logits = raw_logits / T
    """

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # T starts at 1.0

    @property
    def temperature(self) -> float:
        return torch.exp(self.log_temperature).item()

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 100, lr: float = 0.01) -> float:
        logits = logits.detach()
        labels = labels.detach()
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature],
                                       lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            T = torch.exp(self.log_temperature)
            loss = nll(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T_final = self.temperature
        print(f"[INFO] Temperature scaling fit: T={T_final:.4f} "
              f"({'sharpening' if T_final < 1 else 'softening'} confidence)")
        return T_final


def count_parameters(model: nn.Module) -> Dict:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": t, "frozen": f, "total": t + f,
            "trainable_M": f"{t/1e6:.2f}M", "total_M": f"{(t+f)/1e6:.2f}M"}