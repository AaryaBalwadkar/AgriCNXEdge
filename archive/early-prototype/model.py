"""
model.py — Agricultural Multi-Task Edge Model
==============================================
Backbone : ConvNeXt-Tiny (DINOv2-pretrained weights via timm)
           Loaded in 4-bit (BitsAndBytes) and 100% frozen.
Heads    : Dynamically assembled from a config dict at init time.
           • green_apple_disease  → ClassificationHead  (GAP + Linear)
           • mosaic_virus         → ClassificationHead  (GAP + Linear)
           • pest_detection       → YOLOStyleHead       (dense detection)
           • yield_counting       → YOLOStyleHead       (dense detection)

Author   : AgriEdge Project
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

# ── optional heavy deps (graceful degradation when not installed) ──────────────
try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BACKBONE  ──  Frozen ConvNeXt-Tiny  (multi-scale feature extractor)
# ══════════════════════════════════════════════════════════════════════════════

class FrozenConvNeXtBackbone(nn.Module):
    """
    Wraps timm's ConvNeXt-Tiny in ``features_only=True`` mode so that
    forward() returns a list of 4 multi-scale feature maps:
        stage-1 → (B, 96,  H/4,  W/4)
        stage-2 → (B, 192, H/8,  W/8)
        stage-3 → (B, 384, H/16, W/16)
        stage-4 → (B, 768, H/32, W/32)

    ALL parameters are frozen immediately after loading.
    Use ``load_in_4bit=True`` to quantise via BitsAndBytes (QLoRA-ready path).
    """

    # Channel widths for each stage — ConvNeXt-Tiny standard
    STAGE_CHANNELS: List[int] = [96, 192, 384, 768]

    def __init__(
        self,
        pretrained: bool = True,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        # ── Load backbone (optionally with 4-bit quantisation) ────────────────
        if load_in_4bit:
            if not _BNB_AVAILABLE:
                raise ImportError(
                    "bitsandbytes + transformers are required for 4-bit loading. "
                    "Install: pip install bitsandbytes transformers"
                )
            # NOTE: timm does not natively support BitsAndBytes quantisation.
            # The canonical path is to load the model normally, then call
            # ``bitsandbytes.nn.quantize_4bit`` on the resulting Linear layers.
            # This block performs that conversion automatically.
            backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            backbone = _quantize_linear_layers_4bit(backbone)
        else:
            backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )

        self.backbone = backbone

        # ── CRITICAL: Freeze every parameter unconditionally ──────────────────
        self._freeze_backbone()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Set requires_grad=False for every backbone parameter."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        # Double-check by entering eval mode for BN/dropout layers
        self.backbone.eval()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W)  — RGB image, H/W must be divisible by 32.
        Returns:
            features: list of 4 tensors, one per ConvNeXt stage.
        """
        # backbone stays in eval mode to keep BN running stats frozen
        with torch.no_grad():
            features = self.backbone(x)
        return features  # [f1, f2, f3, f4]

    def train(self, mode: bool = True) -> "FrozenConvNeXtBackbone":
        """Override train() so the backbone always stays in eval mode."""
        super().train(mode)
        self.backbone.eval()  # backbone NEVER leaves eval
        return self


def _quantize_linear_layers_4bit(module: nn.Module) -> nn.Module:
    """
    Recursively replace nn.Linear with bnb.nn.Linear4bit.
    Called only when load_in_4bit=True.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            q_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
            )
            # copy weights
            q_layer.weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
            )
            if child.bias is not None:
                q_layer.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, q_layer)
        else:
            _quantize_linear_layers_4bit(child)
    return module


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TASK HEADS  ──  Modular, plug-in heads
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    Global Average Pooling → MLP head for image-level classification.

    Architecture
    ────────────
    Multi-scale features are aggregated across all active stages via
    adaptive average pooling + concat, then passed through a small MLP.

    Placeholder internals: replace the MLP block with your own design.

    Args:
        in_channels_list : channel widths for each backbone stage used.
        num_classes      : number of output classes.
        dropout          : dropout probability before the final linear.
        fuse_stages      : which stage indices (0-based) to use; defaults
                           to all 4.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int,
        dropout: float = 0.3,
        fuse_stages: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.fuse_stages = fuse_stages if fuse_stages is not None else list(range(len(in_channels_list)))
        fused_dim = sum(in_channels_list[i] for i in self.fuse_stages)

        self.gap = nn.AdaptiveAvgPool2d(1)   # (B, C, 1, 1)

        # ── ▼ PLACEHOLDER: replace this block with your MLP design ───────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                              # (B, fused_dim)
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),               # logits
        )
        # ── ▲ END PLACEHOLDER ─────────────────────────────────────────────────

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of backbone feature maps [f1..f4].
        Returns:
            logits: (B, num_classes)
        """
        pooled = [self.gap(features[i]) for i in self.fuse_stages]  # each (B,C,1,1)
        fused  = torch.cat(pooled, dim=1)                            # (B, fused_dim, 1, 1)
        return self.classifier(fused)


class YOLOStyleHead(nn.Module):
    """
    Lightweight YOLO-style dense detection head.

    Operates on the deepest feature map (stage-4, stride-32) by default.
    For fine-grained tasks like yield counting, also fuses stage-3 (stride-16).

    Output per anchor per cell: [x, y, w, h, objectness, *class_scores]
    → raw_output shape: (B, num_anchors × (5 + num_classes), H_grid, W_grid)

    Placeholder internals: replace the neck/head block with your YOLO variant.

    Args:
        in_channels_list : channel widths for each backbone stage.
        num_classes      : number of detection classes (1 for single-class tasks).
        num_anchors      : anchors per grid cell.
        use_stages       : which stage indices to fuse (FPN-lite style).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = 1,
        num_anchors: int = 3,
        use_stages: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_anchors  = num_anchors
        self.num_classes  = num_classes
        self.out_channels = num_anchors * (5 + num_classes)
        self.use_stages   = use_stages if use_stages is not None else [2, 3]  # stages 3 & 4

        # Lateral projections to unify channel dims before fusion
        lateral_in = [in_channels_list[i] for i in self.use_stages]
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.SiLU(),
            )
            for c in lateral_in
        ])

        # ── ▼ PLACEHOLDER: replace this neck+head block ───────────────────────
        self.neck = nn.Sequential(
            _DepthwiseSepConv(256, 256),
            _DepthwiseSepConv(256, 256),
        )
        self.head = nn.Conv2d(256, self.out_channels, 1)
        # ── ▲ END PLACEHOLDER ─────────────────────────────────────────────────

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list of backbone feature maps [f1..f4].
        Returns:
            raw_pred: (B, num_anchors*(5+num_classes), H_grid, W_grid)
                      — pass through a YOLO decode layer for inference.
        """
        # Project each selected stage
        projected = [self.lateral_convs[k](features[self.use_stages[k]])
                     for k in range(len(self.use_stages))]

        # Simple top-down FPN-lite: upsample coarsest and add
        fused = projected[-1]
        for feat in reversed(projected[:-1]):
            fused = feat + nn.functional.interpolate(
                fused, size=feat.shape[-2:], mode="bilinear", align_corners=False
            )

        return self.head(self.neck(fused))


class _DepthwiseSepConv(nn.Module):
    """Depthwise-separable conv block (edge-friendly). Internal helper."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DYNAMIC LOSS WEIGHTING  ──  Homoscedastic Uncertainty (Kendall & Gal)
# ══════════════════════════════════════════════════════════════════════════════

class HomoscedasticUncertaintyWeighter(nn.Module):
    """
    Learns a log-variance (log σ²) per active task.
    Weighted loss = Σ_i  [ L_i / (2·σ_i²)  +  log(σ_i) ]

    The σ parameters are learned jointly with the heads, so the model
    automatically down-weights noisier tasks over training.

    Args:
        task_names: list of active task name strings.
    """

    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        self.task_names = task_names
        # Initialise log-variances at 0 (σ=1, no weighting at start)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(self, raw_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            raw_losses: {task_name: scalar_loss_tensor}
        Returns:
            total_loss    : weighted scalar for .backward()
            weight_info   : {task_name: effective_weight} for logging
        """
        total = torch.zeros(1, device=next(iter(raw_losses.values())).device)
        weight_info: Dict[str, float] = {}

        for name, loss in raw_losses.items():
            log_var    = self.log_vars[name]
            precision  = torch.exp(-log_var)              # 1/σ²
            weighted   = precision * loss + 0.5 * log_var # Kendall formula
            total      = total + weighted
            weight_info[name] = precision.item()

        return total.squeeze(), weight_info


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN MODEL  ──  AgriculturalMultiTaskModel
# ══════════════════════════════════════════════════════════════════════════════

# Registry mapping task name → head class
_HEAD_REGISTRY: Dict[str, type] = {
    "green_apple_disease": ClassificationHead,
    "mosaic_virus":        ClassificationHead,
    "pest_detection":      YOLOStyleHead,
    "yield_counting":      YOLOStyleHead,
}

# Default task-specific hyperparameters (override via head_kwargs)
_HEAD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "green_apple_disease": {"num_classes": 5,  "dropout": 0.3},
    "mosaic_virus":        {"num_classes": 2,  "dropout": 0.25},
    "pest_detection":      {"num_classes": 1,  "num_anchors": 3},
    "yield_counting":      {"num_classes": 1,  "num_anchors": 3},
}


class AgriculturalMultiTaskModel(nn.Module):
    """
    Agricultural Multi-Task Edge Model
    ====================================

    Parameters
    ----------
    active_tasks : dict
        Keys are task names (str); values are dicts of kwargs forwarded to
        each head constructor.  Pass an empty dict ``{}`` to use defaults.

        Valid task names:
            "green_apple_disease"  →  ClassificationHead
            "mosaic_virus"         →  ClassificationHead
            "pest_detection"       →  YOLOStyleHead
            "yield_counting"       →  YOLOStyleHead

    pretrained_backbone : bool
        Download ImageNet-pretrained ConvNeXt-Tiny weights from timm.

    load_backbone_4bit : bool
        Quantise backbone Linear layers to 4-bit via BitsAndBytes.
        Requires: pip install bitsandbytes transformers

    Examples
    --------
    # 1 active head (disease only):
    model = AgriculturalMultiTaskModel(
        active_tasks={"green_apple_disease": {}}
    )

    # All 4 heads with custom class counts:
    model = AgriculturalMultiTaskModel(
        active_tasks={
            "green_apple_disease": {"num_classes": 8},
            "mosaic_virus":        {"num_classes": 3},
            "pest_detection":      {"num_classes": 2, "num_anchors": 5},
            "yield_counting":      {"num_classes": 1},
        }
    )
    """

    ALL_TASKS: List[str] = list(_HEAD_REGISTRY.keys())

    def __init__(
        self,
        active_tasks: Dict[str, Dict[str, Any]],
        pretrained_backbone: bool = True,
        load_backbone_4bit: bool = False,
    ) -> None:
        super().__init__()

        # ── Validate requested tasks ──────────────────────────────────────────
        for task in active_tasks:
            if task not in _HEAD_REGISTRY:
                raise ValueError(
                    f"Unknown task '{task}'. "
                    f"Valid tasks: {list(_HEAD_REGISTRY.keys())}"
                )
        if not active_tasks:
            raise ValueError("active_tasks must contain at least one task.")

        self.active_task_names: List[str] = list(active_tasks.keys())

        # ── 1. Frozen backbone ────────────────────────────────────────────────
        self.backbone = FrozenConvNeXtBackbone(
            pretrained=pretrained_backbone,
            load_in_4bit=load_backbone_4bit,
        )
        stage_channels = FrozenConvNeXtBackbone.STAGE_CHANNELS  # [96,192,384,768]

        # ── 2. Dynamic head assembly ──────────────────────────────────────────
        heads: Dict[str, nn.Module] = {}
        for task_name, user_kwargs in active_tasks.items():
            head_cls    = _HEAD_REGISTRY[task_name]
            merged_kw   = {**_HEAD_DEFAULTS[task_name], **user_kwargs}  # user overrides defaults
            heads[task_name] = head_cls(
                in_channels_list=stage_channels,
                **merged_kw,
            )
        # nn.ModuleDict ensures all heads are properly registered
        self.heads = nn.ModuleDict(heads)

        # ── 3. Uncertainty loss weighter ──────────────────────────────────────
        self.loss_weighter = HomoscedasticUncertaintyWeighter(self.active_task_names)

        # ── 4. Log configuration ──────────────────────────────────────────────
        self._print_model_summary()

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        task_filter: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x           : (B, 3, H, W) input image tensor.
            task_filter : optional subset of active tasks to run.
                          Useful for inference when only one head is needed.
        Returns:
            outputs : {task_name: prediction_tensor}
                      Classification tasks → (B, num_classes)   logits
                      Detection tasks      → (B, A*(5+C), H, W) raw grid
        """
        # Extract frozen multi-scale features once
        features: List[torch.Tensor] = self.backbone(x)

        # Determine which heads to run
        tasks_to_run = task_filter if task_filter else self.active_task_names
        if task_filter:
            invalid = set(task_filter) - set(self.active_task_names)
            if invalid:
                raise ValueError(
                    f"task_filter contains inactive tasks: {invalid}. "
                    f"Active tasks: {self.active_task_names}"
                )

        # Route features through each active head
        outputs: Dict[str, torch.Tensor] = {}
        for task_name in tasks_to_run:
            outputs[task_name] = self.heads[task_name](features)

        return outputs

    # ── loss computation ───────────────────────────────────────────────────────

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets:     Dict[str, Any],
        criterion_map: Dict[str, nn.Module],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute dynamically weighted multi-task loss.

        Args:
            predictions  : output of forward(), {task_name: pred_tensor}
            targets      : {task_name: ground_truth}  — format is head-specific.
            criterion_map: {task_name: loss_fn}  — e.g., nn.CrossEntropyLoss()
                           Pass only the criteria for currently predicted tasks.
        Returns:
            total_loss  : scalar tensor for .backward()
            weight_info : {task_name: learned_precision} for TensorBoard/logging
        """
        raw_losses: Dict[str, torch.Tensor] = {}
        for task_name, pred in predictions.items():
            if task_name not in criterion_map:
                raise KeyError(f"No criterion provided for task '{task_name}'.")
            if task_name not in targets:
                raise KeyError(f"No target provided for task '{task_name}'.")
            raw_losses[task_name] = criterion_map[task_name](pred, targets[task_name])

        total_loss, weight_info = self.loss_weighter(raw_losses)
        return total_loss, weight_info

    # ── utilities ─────────────────────────────────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Returns ONLY the head + loss-weighter parameters (backbone excluded)."""
        params = list(self.heads.parameters()) + list(self.loss_weighter.parameters())
        return params

    def freeze_head(self, task_name: str) -> None:
        """Freeze an individual head (useful during staged training)."""
        if task_name not in self.heads:
            raise KeyError(f"Head '{task_name}' is not active.")
        for p in self.heads[task_name].parameters():
            p.requires_grad = False

    def unfreeze_head(self, task_name: str) -> None:
        """Unfreeze a previously frozen head."""
        if task_name not in self.heads:
            raise KeyError(f"Head '{task_name}' is not active.")
        for p in self.heads[task_name].parameters():
            p.requires_grad = True

    def add_head(self, task_name: str, task_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Dynamically attach a new head at runtime (e.g., when new labelled data arrives).

        Args:
            task_name  : must be in the global task registry.
            task_kwargs: head constructor kwargs; merges with defaults.
        """
        if task_name in self.heads:
            raise ValueError(f"Head '{task_name}' already exists.")
        if task_name not in _HEAD_REGISTRY:
            raise ValueError(f"Unknown task '{task_name}'.")

        merged_kw = {**_HEAD_DEFAULTS[task_name], **(task_kwargs or {})}
        stage_channels = FrozenConvNeXtBackbone.STAGE_CHANNELS
        self.heads[task_name] = _HEAD_REGISTRY[task_name](
            in_channels_list=stage_channels,
            **merged_kw,
        )
        self.active_task_names.append(task_name)
        # Re-build the loss weighter to include the new task
        self.loss_weighter = HomoscedasticUncertaintyWeighter(self.active_task_names)
        print(f"[AgriculturalMultiTaskModel] Head '{task_name}' added. "
              f"Active tasks: {self.active_task_names}")

    def _print_model_summary(self) -> None:
        backbone_params  = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.trainable_parameters())
        print("=" * 60)
        print("  AgriculturalMultiTaskModel — Build Summary")
        print("=" * 60)
        print(f"  Backbone     : ConvNeXt-Tiny  [{backbone_params:,} params | FROZEN]")
        print(f"  Active heads : {self.active_task_names}")
        for name, head in self.heads.items():
            n = sum(p.numel() for p in head.parameters())
            print(f"    ↳ {name:<28} {n:>10,} params")
        print(f"  Trainable    : {trainable_params:,} params")
        print("=" * 60)
