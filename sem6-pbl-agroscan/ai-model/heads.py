import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PRIMITIVE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class _DepthwiseSepConv(nn.Module):
    """
    Depthwise-Separable Convolution Block.

    Factorises a standard k×k convolution into two cheaper operations:
        1. Depthwise  conv  (k×k, groups=C_in)   — spatial feature mixing
        2. Pointwise  conv  (1×1)                 — channel projection

    Parameter reduction vs. standard conv:
        Standard : C_in × C_out × k²
        DS-Conv  : C_in × k² + C_in × C_out   ≈ 8-9× fewer params at k=3

    This is the fundamental primitive for all edge-deployable conv heads in
    this model.  Retained from Phase 1 without modification.

    Args:
        in_ch  : Number of input channels.
        out_ch : Number of output channels.
        stride : Spatial stride (default 1, set to 2 for downsampling).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                             padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv).

    Implements the core operator of MobileNetV2 / EfficientNet:

        Input (C_in)
            │
            ▼  Expand  1×1 Conv → BN → SiLU      C_in → C_in × t
            │
            ▼  Depthwise  3×3 DWConv → BN → SiLU  (spatial mixing, groups=C_mid)
            │
            ▼  Squeeze-and-Excite                  lightweight channel attention
            │    • AdaptiveAvgPool → 1×1 Conv → SiLU → 1×1 Conv → Sigmoid
            │    • Scales the depthwise feature map channel-wise
            │
            ▼  Project  1×1 Conv → BN             C_mid → C_out  (NO activation)
            │
            + ─── skip connection ─ (only when C_in == C_out, stride == 1)

    Inverted Bottleneck Design Rationale
    ──────────────────────────────────────
    Standard ResNets compress then expand: C → C/4 → C.  MBConv does the
    opposite: C → C·t → C_out.  Because depthwise convs are cheap (k² params
    per channel), doing spatial mixing at the wider expanded dimension is still
    more efficient than a full 3×3 conv at the narrow dimension, while capturing
    richer spatial structure.

    Squeeze-and-Excitation (SE)
    ────────────────────────────
    SE inserts a global channel recalibration step (Hu et al., 2018) at a
    negligible FLOPs cost (< 1% of the block).  On mobile hardware the SE
    module can be folded into the subsequent conv's weights after training,
    making inference overhead exactly zero.

    Args:
        in_ch          : Input channel width.
        out_ch         : Output channel width.
        expansion_ratio: Channel expansion factor for the bottleneck  (default 4).
                         EfficientNet-B0 uses t ∈ {1, 6}; we default to 4 for
                         a moderate capacity/latency trade-off.
        se_ratio       : Squeeze ratio for SE module  (default 0.25).
                         se_ch = max(1, int(in_ch × se_ratio)).
        stride         : Spatial stride of the depthwise conv  (default 1).
        dropout_rate   : Stochastic depth probability applied to the main branch
                         before the skip add  (default 0.0 = disabled).
    """

    def __init__(
        self,
        in_ch          : int,
        out_ch         : int,
        expansion_ratio: int   = 4,
        se_ratio       : float = 0.25,
        stride         : int   = 1,
        dropout_rate   : float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch  = int(in_ch * expansion_ratio)
        se_ch   = max(1, int(in_ch * se_ratio))

        # ── Phase 1: Expansion 1×1 ────────────────────────────────────────────
        # Only applied when expansion_ratio > 1 — at ratio=1 the input is
        # already at full width and no projection is needed.
        if expansion_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            )
        else:
            self.expand = nn.Identity()

        # ── Phase 2: Depthwise 3×3 ───────────────────────────────────────────
        # Spatial feature mixing within each channel independently.
        # Using _DepthwiseSepConv here would add an extra pointwise projection;
        # we want pure depthwise for the MBConv pattern, so we use a raw
        # grouped conv + separate BN + activation.
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )

        # ── Phase 3: Squeeze-and-Excitation channel attention ─────────────────
        # Pool → compress → excite → scale.
        # The two 1×1 convs act as a lightweight fully-connected attention map.
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # (B, C_mid, 1, 1)
            nn.Conv2d(mid_ch, se_ch, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_ch, mid_ch, 1, bias=True),
            nn.Sigmoid(),                         # attention ∈ (0,1) per channel
        )

        # ── Phase 4: Pointwise Projection ─────────────────────────────────────
        # No activation after projection — standard MBConv / EfficientNet design.
        # Omitting activation preserves the representational capacity of the
        # residual pathway.
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # ── Stochastic Depth (optional regularisation) ────────────────────────
        self.drop = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()

        # Skip connection is only valid when spatial resolution and channel
        # width are both preserved.
        self.use_skip = (in_ch == out_ch) and (stride == 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, C_in, H, W)
        Returns:
            out : (B, C_out, H', W')  where H' = H // stride
        """
        # Main branch
        z = self.expand(x)           # (B, C_mid, H, W)
        z = self.dw_conv(z)          # (B, C_mid, H', W')
        z = z * self.se(z)           # channel-wise attention
        z = self.project(z)          # (B, C_out, H', W')
        z = self.drop(z)             # optional stochastic depth

        # Residual add (only when shapes match)
        if self.use_skip:
            z = z + x
        return z

class EfficientNetClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels     : int,
        num_classes     : int,
        neck_ch         : int   = 256,
        expansion_ratio : int   = 4,
        se_ratio        : float = 0.25,
        dropout         : float = 0.3,
    ) -> None:
        super().__init__()
        
        # Step 1: Channel reduction
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, neck_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_ch),
            nn.SiLU(inplace=True),
        )

        # Step 2: Mobile Inverted Bottleneck feature refinement
        self.mbconv = MBConvBlock(
            in_ch          = neck_ch,
            out_ch         = neck_ch,
            expansion_ratio= expansion_ratio,
            se_ratio       = se_ratio,
            stride         = 1,           
            dropout_rate   = 0.0,         
        )

        # Step 3: Global Average Pooling -> flatten
        self.gap = nn.AdaptiveAvgPool2d(1)   

        # Step 4: Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                        
            nn.Dropout(p=dropout),
            nn.Linear(neck_ch, num_classes),     
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_reduce(x)         
        x = self.mbconv(x)                 
        x = self.gap(x)                    
        return self.classifier(x)

class LeafHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.head = EfficientNetClassificationHead(96, num_classes) 

    def forward(self, f1):
        return self.head(f1)

class PestHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.head = EfficientNetClassificationHead(192, num_classes)

    def forward(self, f2):
        return self.head(f2)

class FruitHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.head = EfficientNetClassificationHead(384, num_classes)

    def forward(self, f3):
        return self.head(f3)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CENTER-POINT (HEATMAP) DETECTION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class CenterPointDetectionHead(nn.Module):
    """
    Implementation of "Objects as Points" (CenterNet) logic.
    Instead of grid-cells, it predicts a continuous heatmap of centers, 
    allowing for sub-pixel precision and zero-overlap collisions.
    """
    def __init__(self, in_channels: int, num_classes: int = 1, neck_ch: int = 128):
        super().__init__()
        self.num_classes = num_classes

        # Refinement branch for f1 features (128x128 resolution)
        self.stem = nn.Sequential(
            _DepthwiseSepConv(in_channels, neck_ch),
            _DepthwiseSepConv(neck_ch, neck_ch)
        )

        # Head 1: Heatmap (Center Probability)
        # Predicts where apple centers are located [B, 1, 128, 128]
        self.heatmap = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_ch, num_classes, 1)
        )

        # Head 2: Width & Height
        # Regresses the box size at each center [B, 2, 128, 128]
        self.wh = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_ch, 2, 1)
        )

        # Head 3: Sub-pixel Offset
        # Recover quantization error from the 4x downsampling [B, 2, 128, 128]
        self.offset = nn.Sequential(
            nn.Conv2d(neck_ch, neck_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(neck_ch, 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Focal Loss bias initialization for the heatmap (Zhou et al. 2019)
        # Prevents initial background-heavy loss from destabilizing training.
        prior_prob = 0.01
        bias_init  = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap[-1].bias, bias_init)
        
        # Zero-initialize wh and offset heads for stable training start
        nn.init.constant_(self.wh[-1].bias, 0)
        nn.init.constant_(self.offset[-1].bias, 0)

    def forward(self, f1: Tensor) -> dict:
        # Use stride-4 features (f1) to ensure high-density apple separation
        x = self.stem(f1)
        return {
            "heatmap": self.heatmap(x),
            "wh":      self.wh(x),
            "offset":  self.offset(x)
        }

class MultiTaskHeads(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # Classification heads pull from f4 (stride 32, 768 channels)
        cls_in_ch = 768
        self.leaf_head  = EfficientNetClassificationHead(cls_in_ch, model_config.num_leaf_classes)
        self.pest_head  = EfficientNetClassificationHead(cls_in_ch, model_config.num_pest_classes)
        self.fruit_head = EfficientNetClassificationHead(cls_in_ch, model_config.num_fruit_classes)
        
        # The yield head uses f1 (stride 4, 96 channels for DINOv3 ConvNeXt-Nano)
        # We pass the channel count dynamically from model_config
        self.yield_head = CenterPointDetectionHead(
            in_channels=model_config.f1_channels,
            num_classes=model_config.num_yield_classes,
            neck_ch=128
        )

    def forward(self, features: dict):
        return {
            "leaf":      self.leaf_head(features["f4"]),
            "pest":      self.pest_head(features["f4"]),
            "fruit":     self.fruit_head(features["f4"]),
            "yield_map": self.yield_head(features["f1"]), # High-resolution detection pass
        }