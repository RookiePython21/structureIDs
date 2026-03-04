"""
model/fs_symbol.py — FS-Symbol model components for Detectron2.

Implements the two key innovations from "Few-Shot Symbol Detection in Engineering Drawings"
(Jamieson et al., 2024):

  1. Cosine similarity classifier — replaces FC layer to reduce intra-class variance.
  2. Two-stage TFA setup — freeze backbone, keep RPN+ROI trainable in Stage 2.

Usage:
    from model.fs_symbol import build_fs_symbol_model, setup_stage2

    # Stage 1
    model = build_fs_symbol_model(cfg)
    # ... train full model ...

    # Stage 2
    model = build_fs_symbol_model(cfg)
    setup_stage2(model, novel_class_ids=NOVEL_CLASS_IDS)
    # ... fine-tune on K-shot support set ...
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# ---------------------------------------------------------------------------
# Class definitions for this dataset
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    # Base classes (abundant)
    "square_structure",   # 0
    "circle_structure",   # 1
    "headwall",           # 2
    # Novel classes (rare — few-shot targets)
    "rip_rap",            # 3
    # Excluded (masked out, not detected): notes, compass, site_map, title_exclusion, legend
]
BASE_CLASS_IDS  = [0, 1, 2]   # 0-indexed within classifier
NOVEL_CLASS_IDS = [3]          # 0-indexed within classifier

# COCO category IDs (1-indexed, matching preprocess.py CATEGORY_MAP)
COCO_CATEGORY_MAP = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Cosine Similarity Classifier
# ---------------------------------------------------------------------------
class CosineSimLinear(nn.Module):
    """
    Cosine similarity classifier as used in FS-Symbol / TFA.

    Computes:  logit_i = scale * cos(w_i, x)
                       = scale * (w_i · x) / (|w_i| |x|)

    Benefits over standard FC:
      - Class weights act as prototypes in a normalized feature space.
      - Reduces intra-class variance — important for visually similar symbols.
      - Works better with limited examples (few-shot regime).

    Args:
        in_features:  Input feature dimensionality (e.g. 1024 from FC head).
        num_classes:  Number of output classes INCLUDING background.
        scale:        Temperature scale factor (paper uses 20.0).
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)

    def reset_novel_weights(self, novel_class_indices: List[int]) -> None:
        """
        Randomly reinitialize weights for novel classes.
        Called at the start of Stage 2 to remove base-class bias
        from the novel class prototypes.
        """
        with torch.no_grad():
            for idx in novel_class_indices:
                nn.init.kaiming_uniform_(
                    self.weight[idx : idx + 1], a=math.sqrt(5)
                )
        print(f"  [CosineSimLinear] Reinitialized novel class weights: {novel_class_indices}")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, num_classes={self.num_classes}, scale={self.scale}"


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------
def replace_classifier_with_cosine(model: nn.Module, scale: float = 20.0) -> nn.Module:
    """
    Replace the standard linear cls_score with CosineSimLinear.

    This must be called after build_model() but before training starts.
    The replacement preserves the box regressor (bbox_pred) unchanged.

    Args:
        model:  Detectron2 GeneralizedRCNN model.
        scale:  Cosine similarity temperature (paper: 20.0).

    Returns:
        Modified model (in-place modification, return for convenience).
    """
    box_predictor = model.roi_heads.box_predictor
    old_cls: nn.Linear = box_predictor.cls_score

    in_features  = old_cls.in_features
    num_classes  = old_cls.out_features  # includes background

    new_cls = CosineSimLinear(in_features, num_classes, scale).to(old_cls.weight.device)
    box_predictor.cls_score = new_cls

    print(
        f"[FS-Symbol] Replaced cls_score: "
        f"Linear({in_features}, {num_classes}) → "
        f"CosineSimLinear({in_features}, {num_classes}, scale={scale})"
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze backbone (ResNet-101 + FPN) for Stage 2 fine-tuning.

    Per the paper, keeping the backbone frozen preserves the diverse feature
    representations learned during base training. The RPN and ROI heads remain
    trainable — this is the key difference from vanilla TFA and is responsible
    for the +40.4 nAP improvement.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Ensure RPN is trainable (generates better proposals for novel classes)
    for param in model.proposal_generator.parameters():
        param.requires_grad = True

    # Ensure ROI heads fully trainable
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    _print_param_counts(model)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for Stage 1 full training."""
    for param in model.parameters():
        param.requires_grad = True
    print("[FS-Symbol] All parameters unfrozen (Stage 1 mode).")
    _print_param_counts(model)


def reinit_novel_class_weights(model: nn.Module, novel_class_ids: List[int]) -> None:
    """
    Randomly reinitialize the classifier weight rows for novel classes.

    Without this, the novel class weights start as the base-trained values,
    which introduces bias. Reinitializing gives them a clean start with only
    K examples to learn from.
    """
    box_predictor = model.roi_heads.box_predictor
    cls_score = box_predictor.cls_score

    if isinstance(cls_score, CosineSimLinear):
        cls_score.reset_novel_weights(novel_class_ids)
    else:
        print(
            "[FS-Symbol] WARNING: cls_score is not CosineSimLinear. "
            "Cannot reinitialize novel class weights. Did you call "
            "replace_classifier_with_cosine() first?"
        )


def setup_stage2(model: nn.Module, novel_class_ids: List[int] = None) -> None:
    """
    Prepare model for Stage 2 few-shot fine-tuning.

    Steps:
      1. Freeze backbone.
      2. Reinitialize novel class prototype weights.

    Args:
        model:           Detectron2 model loaded from Stage 1 checkpoint.
        novel_class_ids: 0-indexed class IDs for novel classes
                         (defaults to NOVEL_CLASS_IDS for this dataset).
    """
    if novel_class_ids is None:
        novel_class_ids = NOVEL_CLASS_IDS

    print("[FS-Symbol] Setting up Stage 2 (few-shot fine-tuning):")
    freeze_backbone(model)
    reinit_novel_class_weights(model, novel_class_ids)
    print(f"[FS-Symbol] Stage 2 ready. Novel class IDs: {novel_class_ids}")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_fs_symbol_model(cfg, cosine_scale: float = 20.0) -> nn.Module:
    """
    Build the FS-Symbol model from a Detectron2 config.

    Builds a standard Faster R-CNN (R-101 + FPN) then replaces the
    classification head with CosineSimLinear.

    Args:
        cfg:           Detectron2 CfgNode (already merged with fs_symbol_config.yaml).
        cosine_scale:  Temperature for cosine similarity (paper: 20.0).

    Returns:
        nn.Module ready for Stage 1 or Stage 2 training.
    """
    model = build_model(cfg)
    model = replace_classifier_with_cosine(model, scale=cosine_scale)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load a Detectron2 checkpoint into model."""
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    print(f"[FS-Symbol] Loaded checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _print_param_counts(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = total - frozen
    print(f"  [Params] Frozen: {frozen:>12,}  |  Trainable: {trainable:>12,}  |  Total: {total:>12,}")
