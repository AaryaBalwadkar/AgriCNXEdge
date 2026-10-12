from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CANONICAL CLASS MAPS — Single source of truth for class index ordering.
# These MUST match the exact sorted(os.listdir()) order used when each
# teacher model was trained. Never reorder these lists.
# ---------------------------------------------------------------------------

# Fruit teacher: tf_efficientnetv2_b0, trained on 5 classes (alphabetical folders)
FRUIT_CLASS_MAP: dict = {
    "Anthracnose": 0,
    "Black Rot":   1,
    "Blotch":      2,
    "Healthy":     3,
    "Scab":        4,
}

# Leaf teacher: EfficientNetV2-B0 (Keras), trained on 7 classes (alphabetical folders)
LEAF_CLASS_MAP: dict = {
    "Apple_Mosaic":           0,
    "Apple___Apple_scab":     1,
    "Apple___Black_rot":      2,
    "Alternaria":             3,
    "Healthy":                4,
}

# Pest teacher: EfficientNetV2-B0, trained on 6 classes (alphabetical folders)
PEST_CLASS_MAP: dict = {
    "69 - Xylotrechus (Stemwood borer)": 0,
    "aphids":       1,
    "leafhoppers":  2,
    "melaybug":     3,
    "spider_mite":  4,
    "whitefly":     5,
}


@dataclass
class DataConfig:
    """Single source of truth for all data paths and input specifications.

    Attributes:
        image_size: Spatial resolution for all input images. 512px yields a
            16x16 feature grid at the deepest ConvNeXt stage (32x downsampling).
        teacher_csv_dir: Directory containing pre-computed teacher soft-label CSVs.
        splits_dir: Directory containing train.txt, val.txt, and test.txt files.
        images_dir: Flat directory containing all merged dataset images.
        labels_dir: Directory containing the 5 master ground-truth CSV files.
        max_boxes: Fixed number of detection anchors per image for collate-safe
            padding. Matches the yield head's num_anchors output dimension.
    """
    image_size: int = 512
    teacher_csv_dir: str = "final_dataset/teacher_outputs"
    splits_dir: str = "final_dataset/splits"
    images_dir: str = "final_dataset/images"
    labels_dir: str = "final_dataset/labels"
    max_boxes: int = 300


@dataclass
class ModelConfig:
    """Single source of truth for student model architecture dimensions.

    Attributes:
        backbone_name: Identifier for the DINOv3 ConvNeXt Tiny backbone.
        pretrained_backbone: If True, initializes with ImageNet pretrained weights.
        num_leaf_classes: Output nodes for leaf disease classification head.
        num_fruit_classes: Output nodes for fruit disease classification head.
        num_pest_classes: Output nodes for pest classification head.
        num_yield_classes: Output nodes for yield detection class scores (Apple=1).
        num_anchors: Number of detection proposal slots per image. A 16x16 spatial
            grid at 512px input produces 256 candidate anchor locations.
    """
    backbone_name: str = "dinov3_convnext_tiny"
    pretrained_backbone: bool = True
    dinov3_repo_dir: str = "./dinov3" 
    dinov3_weights: str = "./weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    num_leaf_classes: int = 5   # Must equal len(LEAF_CLASS_MAP)
    num_fruit_classes: int = 5  # Must equal len(FRUIT_CLASS_MAP)
    num_pest_classes: int = 6   # Must equal len(PEST_CLASS_MAP)
    num_yield_classes: int = 1
    num_anchors: int = 300
    
    # PARAMETER FOR CENTER-POINT LOGIC
    f1_channels: int = 96


@dataclass
class TrainConfig:
    """Single source of truth for all training, hardware, and distillation hyperparameters.

    Attributes:
        batch_size: Real samples per gradient step. 8 fits within 16GB T4 VRAM at 512px.
        gradient_accumulation_steps: Accumulate gradients over N steps before updating,
            simulating an effective batch size of 32 without OOM.
        num_workers: DataLoader subprocess workers. 2 avoids Colab CPU overhead.
        use_amp: Enables FP16 Automatic Mixed Precision to halve activation VRAM usage.
        device: Compute target. Hardcoded to cuda for Colab T4 training.
        learning_rate: Initial learning rate for the Adam optimizer.
        epochs: Total passes through the training dataset.
        lambda_leaf: Loss weight scaling the leaf classification task contribution.
        lambda_pest: Loss weight scaling the pest classification task contribution.
        lambda_fruit: Loss weight scaling the fruit classification task contribution.
        lambda_yield: Loss weight scaling the yield detection task contribution.
            Higher weight compensates for the detection task's smaller sample count.
        alpha: Blending ratio between Ground Truth (CE/Focal) and Teacher (KL) losses.
            alpha=1.0 is pure GT supervision; alpha=0.0 is pure distillation.
        beta: Scaling factor for teacher bounding box coordinate alignment (L1 loss).
        gamma: Scaling factor for teacher confidence score alignment (MSE loss).
        temperature: Softening factor applied to teacher/student logits before KL
            Divergence. Higher T smooths the distribution; loss is scaled by T^2
            to preserve gradient magnitude.
        ignore_index: Sentinel value for missing ground truth labels. Masked out
            of all classification losses to prevent gradient contamination.
        checkpoints_dir: Directory for saving latest and best model checkpoints.
    """
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_workers: int = 8
    use_amp: bool = False
    device: str = "cuda"
    learning_rate: float = 1e-3
    epochs: int = 300
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5
    temperature: float = 3.0
    ignore_index: int = -1
    checkpoints_dir: str = "checkpoints/"
    resume_from_checkpoint: str = None


@dataclass
class Config:
    """Root configuration aggregating all sub-configs into a single importable object."""
    data: DataConfig = field(default_factory=lambda: DataConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    train: TrainConfig = field(default_factory=lambda: TrainConfig())


cfg = Config()

# ---------------------------------------------------------------------------
# Sanity-check: catch map vs. class-count mismatches at import time
# ---------------------------------------------------------------------------
assert len(LEAF_CLASS_MAP)  == cfg.model.num_leaf_classes,  \
    f"LEAF_CLASS_MAP has {len(LEAF_CLASS_MAP)} entries but num_leaf_classes={cfg.model.num_leaf_classes}"
assert len(FRUIT_CLASS_MAP) == cfg.model.num_fruit_classes, \
    f"FRUIT_CLASS_MAP has {len(FRUIT_CLASS_MAP)} entries but num_fruit_classes={cfg.model.num_fruit_classes}"
assert len(PEST_CLASS_MAP)  == cfg.model.num_pest_classes,  \
    f"PEST_CLASS_MAP has {len(PEST_CLASS_MAP)} entries but num_pest_classes={cfg.model.num_pest_classes}"