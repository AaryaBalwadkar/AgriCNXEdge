# Hierarchical Multi-Task Learning with Foundation Vision Backbone and Multi-Teacher Distillation for Edge-Based Agricultural Monitoring

## 1. System Overview

The proposed system architecture is designed for offline, edge-deployable agricultural monitoring. It distills knowledge from four distinct expert models (teachers) into a single, highly efficient multi-task student backbone. The chosen student backbone is a pre-trained DINOv3 ConvNeXt Tiny, which remains frozen or is only lightly fine-tuned during training. 

By leveraging the hierarchical representations intrinsic to the ConvNeXt architecture, distinct agricultural tasks are attached to different stages of the feature pyramid. Following training, the multi-task student model undergoes post-training quantization to enable low-latency inference on mobile and edge devices in the field.

**Data Flow Diagram:**

```text
                        [ Input Image (e.g., 3x224x224) ]
                                      │
  ┌───────────────────────────────────┼────────────────────────────────────┐
  │                                   ▼                                    │
  │                  STUDENT BACKBONE (DINOv3 ConvNeXt Tiny)               │
  │                                                                        │
  │  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐         │
  │  │   Stage 1   │───────▶│   Stage 2   │───────▶│   Stage 3   │───────┐ │
  │  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘       │ │
  │         │                      │                      │              │ │
  └─────────┼──────────────────────┼──────────────────────┼──────────────┼─┘
            │                      │                      │              │
            ▼                      ▼                      ▼              ▼
     [ Leaf Disease ]       [ Pest Detect ]        [ Fruit Disease]   [ Stage 4 ]
     [     Head     ]       [     Head    ]        [     Head     ]   [Yield Head]
            │                      │                      │              │
            ▼                      ▼                      ▼              ▼
     Soft/Hard Loss         Soft/Hard Loss         Soft/Hard Loss   Soft/Hard Loss
            ▲                      ▲                      ▲              ▲
            │                      │                      │              │
       Teacher 1              Teacher 2              Teacher 3      Teacher 4
   (EfficientNetV2-B0)       (YOLOn-cls)        (MobileNetV4-Small)   (YOLOn)
    [ Leaf Disease ]       [ Pest Classification] [ Fruit Disease]   [ Yield Est. ]
            ▲                      ▲                      ▲              ▲
            │                      │                      │              │
  ┌─────────┴──────────────────────┴──────────────────────┴──────────────┴─┐
  │                           Ground Truth Labels                          │
  └────────────────────────────────────────────────────────────────────────┘
```
*Note: Post-training, the student model is quantized (e.g., INT8) for offline mobile inference, and the teacher models are discarded.*

## 2. Feature Routing Specification

The ConvNeXt Tiny backbone naturally generates a hierarchical feature pyramid across four stages. To optimize learning and resource utilization, early-stage features are routed to simpler tasks, while late-stage features are routed to complex tasks requiring global context.

*   **Stage 1 Features ($F_1$):** Extracted after the first spatial downsampling and block sequence. These high-resolution, low-level features contain fine-grained textural anomalies (e.g., localized yellowing, small spots) and are routed directly to the **Leaf Disease Head**.
*   **Stage 2 Features ($F_2$):** Extracted after the second structural stage. Balancing spatial resolution and semantic abstraction, these features are routed to the **Pest Detection Head** (classification/presence), where recognizing specific structural pest patterns is required without needing the entire global image context.
*   **Stage 3 Features ($F_3$):** Extracted after the third stage. These robust, high-level semantic features are routed to the **Fruit Disease Head**, where understanding the disease context relative to the whole fruit object is critical.
*   **Stage 4 Features ($F_4$):** The final, most abstract, and lowest-resolution feature map. These are routed to the **Yield Estimation (Detection) Head**. Complex object-level regression (bounding boxes) and categorical classification for yield counting rely deeply on this global abstraction layer.

## 3. Multi-Teacher Distillation Pipeline

To maintain high accuracy despite using a lightweight multi-task backbone, Knowledge Distillation (KD) transfers generalization capabilities from four independent expert networks to the student. 

For a given agricultural input image $x$:
1.  **Teacher 1 ($T_1$, EfficientNetV2-B0):** Processes $x$ and produces soft categorical probabilities $\hat{y}_{T1}$ representing Leaf Disease distributions.
2.  **Teacher 2 ($T_2$, YOLOn-cls):** Processes $x$ and produces soft categorical probabilities $\hat{y}_{T2}$ indicating the presence/class of Pests.
3.  **Teacher 3 ($T_3$, MobileNetV4-Small):** Processes $x$ and produces soft categorical probabilities $\hat{y}_{T3}$ for Fruit Disease classification.
4.  **Teacher 4 ($T_4$, YOLOn):** Processes $x$ and produces dense object predictions—bounding boxes $B_{T4}$ and class probabilities $P_{T4}$—for Apple Yield Estimation.

During forward propagation, the student's task heads generate corresponding predictions: $\{ \tilde{y}_{S1}, \tilde{y}_{S2}, \tilde{y}_{S3}, (\tilde{B}_{S4}, \tilde{P}_{S4}) \}$. The model's training is guided by aligning the student outputs toward the teachers' softened target distributions. This strategy acts as a strong regularizer and minimizes catastrophic interference across the disjoint tasks.

## 4. Loss Function Formulation

The total optimization objective $\mathcal{L}_{Total}$ is a weighted sum of the explicit Ground Truth (GT) multi-task losses and the Knowledge Distillation (KD) losses across all four heads.

$$ \mathcal{L}_{Total} = \sum_{i=1}^{3} \lambda_i \left( \alpha \mathcal{L}_{CE}(y_i, \tilde{y}_{Si}) + (1-\alpha) \mathcal{L}_{KL}(\hat{y}_{Ti} \parallel \tilde{y}_{Si}) \right) + \lambda_4 \mathcal{L}_{Det} $$

Where:
*   $y_i$: Ground truth one-hot labels for classification tasks $i \in \{1, 2, 3\}$.
*   $\tilde{y}_{Si}$: Student probability predictions (utilizing a softmax with temperature $\tau$ specifically during the KD component).
*   $\hat{y}_{Ti}$: Teacher probability predictions.
*   $\mathcal{L}_{CE}$: Cross-Entropy Loss for standard ground truth supervision.
*   $\mathcal{L}_{KL}$: Kullback-Leibler Divergence Loss for soft-label teacher distillation.
*   $\alpha$: Scalar hyperparameter balancing GT and KD loss ($\alpha \in [0, 1]$).
*   $\lambda_i$: Balancing weight assigned to the $i$-th specific task to ensure uniform multi-task gradients.

The Yield Estimation loss $\mathcal{L}_{Det}$ blends standard ground truth detection loss with bounding box and probability distillation from the $T_4$ teacher:

$$ \mathcal{L}_{Det} = \mathcal{L}_{YOLO\_GT}(B_{GT}, y_{GT}, \tilde{B}_{S4}, \tilde{P}_{S4}) + \beta \mathcal{L}_{DFL}(B_{T4}, \tilde{B}_{S4}) + \gamma \mathcal{L}_{KL}(\hat{P}_{T4} \parallel \tilde{P}_{S4}) $$

Where $\mathcal{L}_{YOLO\_GT}$ encapsulates standard bounding box (e.g., CIoU) and classification losses against ground truth, $\mathcal{L}_{DFL}$ represents Distribution Focal Loss (or equivalent bounding box geometric distillation), and $\beta, \gamma$ moderate the detection KD strength.

## 5. Tensor Flow

The precise tensor dimensions described below assume an input batch size of $N$ and a standard image spatial resolution ($H=224, W=224$). ConvNeXt Tiny employs structural downsampling ratios of $S_1=4, S_2=8, S_3=16, S_4=32$.

**Backbone Inputs:**
*   Input Tensor: `[N, 3, H, W]`

**Student Stage Features (ConvNeXt Tiny Architecture):**
*   $F_1$ (Stage 1 Output): `[N, 96, H/4, W/4]`
*   $F_2$ (Stage 2 Output): `[N, 192, H/8, W/8]`
*   $F_3$ (Stage 3 Output): `[N, 384, H/16, W/16]`
*   $F_4$ (Stage 4 Output): `[N, 768, H/32, W/32]`

**Head Routing & Outputs:**

1.  **Leaf Disease Head:**
    *   **Input:** $F_1$ Tensor of shape `[N, 96, H/4, W/4]`
    *   **Architecture:** Global Average Pooling (GAP) $\rightarrow$ Fully Connected Layer
    *   **Output ($\tilde{y}_{S1}$):** `[N, Num_Leaf_Classes]` (Logits/Probabilities)
2.  **Pest Detection Head:**
    *   **Input:** $F_2$ Tensor of shape `[N, 192, H/8, W/8]`
    *   **Architecture:** Global Average Pooling (GAP) $\rightarrow$ Fully Connected Layer
    *   **Output ($\tilde{y}_{S2}$):** `[N, Num_Pest_Classes]` (Logits/Probabilities)
3.  **Fruit Disease Head:**
    *   **Input:** $F_3$ Tensor of shape `[N, 384, H/16, W/16]`
    *   **Architecture:** Global Average Pooling (GAP) $\rightarrow$ Fully Connected Layer
    *   **Output ($\tilde{y}_{S3}$):** `[N, Num_Fruit_Classes]` (Logits/Probabilities)
4.  **Yield Estimation Head (Detection):**
    *   **Input:** $F_4$ Tensor of shape `[N, 768, H/32, W/32]`
    *   **Architecture:** Detection Decoder Network (utilizing $F_4$ for spatial bounding box mapping and classification). 
    *   **Output Bounding Boxes ($\tilde{B}_{S4}$):** `[N, Num_Anchors, 4]` (Formatted as offset coordinates or $cx, cy, w, h$)
    *   **Output Class Probabilities ($\tilde{P}_{S4}$):** `[N, Num_Anchors, Num_Object_Classes]` 
