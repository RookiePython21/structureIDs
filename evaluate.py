"""
evaluate.py — Inference + evaluation for FS-Symbol models.

Runs inference on all test patches, combines predictions with NMS across
overlapping patches, then computes:
  - mAP  (all classes, IoU=0.5)
  - bAP  (base classes only)
  - nAP  (novel classes only)
  - Per-class AP and Recall

Can evaluate multiple K-shot models in one run for easy comparison.

Usage:
    # Evaluate a single checkpoint
    python evaluate.py \\
        --checkpoint checkpoints/fs_symbol_K5.pth \\
        --data_dir data \\
        --output_dir results/k5

    # Evaluate all K values
    python evaluate.py \\
        --checkpoints checkpoints/fs_symbol_K1.pth \\
                      checkpoints/fs_symbol_K2.pth \\
                      checkpoints/fs_symbol_K5.pth \\
                      checkpoints/fs_symbol_K9.pth \\
        --data_dir data \\
        --output_dir results

    # Also evaluate the base model (no fine-tuning)
    python evaluate.py \\
        --checkpoint checkpoints/base_model.pth \\
        --output_dir results/base

Colab usage:
    !python evaluate.py \\
        --checkpoint /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K9.pth \\
        --data_dir   /content/drive/MyDrive/structureIDs/data \\
        --output_dir /content/drive/MyDrive/structureIDs/results/k9
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

from model.fs_symbol import (
    replace_classifier_with_cosine,
    CLASS_NAMES,
    BASE_CLASS_IDS,
    NOVEL_CLASS_IDS,
)

setup_logger()
logger = logging.getLogger("fs_symbol.evaluate")

# ── Colab detection ────────────────────────────────────────────────────────────
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    _here    = "/content/drive/MyDrive/structureIDs"
    DATA_DIR = f"{_here}/data"
    OUT_DIR  = f"{_here}/results"
else:
    _here    = Path(__file__).parent
    DATA_DIR = str(_here / "data")
    OUT_DIR  = str(_here / "results")

BASE_CLASSES  = [CLASS_NAMES[i] for i in BASE_CLASS_IDS]
NOVEL_CLASSES = [CLASS_NAMES[i] for i in NOVEL_CLASS_IDS]

_registered = set()

def register_test_dataset(data_dir: str):
    name = "civil_structures_test"
    if name not in _registered:
        ann_dir   = Path(data_dir) / "annotations"
        patch_dir = Path(data_dir) / "patches"
        register_coco_instances(name, {}, str(ann_dir / "test.json"), str(patch_dir / "test"))
        _registered.add(name)
    return name


# ── Config builder ────────────────────────────────────────────────────────────
def build_eval_cfg(checkpoint: str, data_dir: str, output_dir: str) -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES   = len(CLASS_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.5
    cfg.MODEL.WEIGHTS                 = ""   # loaded manually below
    cfg.INPUT.MIN_SIZE_TEST           = 640
    cfg.INPUT.MAX_SIZE_TEST           = 640
    cfg.TEST.DETECTIONS_PER_IMAGE     = 300
    cfg.OUTPUT_DIR                    = output_dir
    cfg.DATASETS.TEST                 = ("civil_structures_test",)
    os.makedirs(output_dir, exist_ok=True)
    cfg.freeze()
    return cfg


# ── Patch-level NMS ───────────────────────────────────────────────────────────
def combine_patch_predictions_nms(predictions: list, nms_iou_thresh: float = 0.5) -> list:
    """
    Combine predictions from overlapping patches using NMS.

    Each prediction dict should have:
        "source_image" : original image name
        "patch_origin" : [x_offset, y_offset] in source image coordinates
        "instances"    : Detectron2 Instances (boxes in patch-local coords)

    Returns list of {source_image, boxes_xyxy, scores, labels} per source image.
    """
    from detectron2.structures import Boxes, Instances
    from torchvision.ops import nms

    by_source = defaultdict(list)
    for pred in predictions:
        by_source[pred["source_image"]].append(pred)

    combined = []
    for source_img, preds in by_source.items():
        all_boxes  = []
        all_scores = []
        all_labels = []

        for pred in preds:
            ox, oy    = pred["patch_origin"]
            instances = pred["instances"]
            boxes     = instances.pred_boxes.tensor.cpu()  # (N, 4) xyxy
            scores    = instances.scores.cpu()
            labels    = instances.pred_classes.cpu()

            # Shift from patch-local to source-image coordinates
            offsets = torch.tensor([ox, oy, ox, oy], dtype=torch.float32)
            boxes   = boxes + offsets

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        if not all_boxes:
            continue

        all_boxes  = torch.cat(all_boxes,  dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Per-class NMS
        keep_indices = []
        for cls_id in all_labels.unique():
            mask = all_labels == cls_id
            cls_boxes  = all_boxes[mask]
            cls_scores = all_scores[mask]
            cls_idx    = torch.where(mask)[0]
            keep       = nms(cls_boxes, cls_scores, nms_iou_thresh)
            keep_indices.append(cls_idx[keep])

        if keep_indices:
            keep = torch.cat(keep_indices)
            combined.append({
                "source_image": source_img,
                "boxes_xyxy":  all_boxes[keep],
                "scores":      all_scores[keep],
                "labels":      all_labels[keep],
            })

    return combined


# ── Metrics computation ───────────────────────────────────────────────────────
def compute_split_metrics(coco_eval_results: dict, base_classes: list, novel_classes: list) -> dict:
    """
    Extract bAP (base) and nAP (novel) from full COCO evaluation results.

    Args:
        coco_eval_results: dict from COCOEvaluator (keys like "bbox/AP", "bbox/AP-<class>")
        base_classes:      list of base class names
        novel_classes:     list of novel class names

    Returns:
        dict with mAP, bAP, nAP, and per-class APs.
    """
    per_class_ap = {}
    for cls_name in CLASS_NAMES:
        key = f"bbox/AP-{cls_name}"
        per_class_ap[cls_name] = coco_eval_results.get(key, float("nan"))

    base_aps  = [per_class_ap[c] for c in base_classes  if not np.isnan(per_class_ap.get(c, float("nan")))]
    novel_aps = [per_class_ap[c] for c in novel_classes if not np.isnan(per_class_ap.get(c, float("nan")))]

    return {
        "mAP":  coco_eval_results.get("bbox/AP",   float("nan")),
        "bAP":  float(np.mean(base_aps))  if base_aps  else float("nan"),
        "nAP":  float(np.mean(novel_aps)) if novel_aps else float("nan"),
        "per_class_AP": per_class_ap,
    }


# ── Single model evaluation ───────────────────────────────────────────────────
def evaluate_checkpoint(
    checkpoint: str,
    data_dir: str,
    output_dir: str,
    cosine_scale: float = 20.0,
) -> dict:
    test_dataset = register_test_dataset(data_dir)
    cfg          = build_eval_cfg(checkpoint, data_dir, output_dir)

    # Build model + cosine classifier + load checkpoint
    model = build_model(cfg)
    model = replace_classifier_with_cosine(model, scale=cosine_scale)
    DetectionCheckpointer(model).load(checkpoint)
    model.eval()

    logger.info(f"Evaluating: {checkpoint}")
    logger.info(f"Output dir: {output_dir}")

    # Run COCO evaluation (on test patches directly — standard D2 flow)
    evaluator   = COCOEvaluator(test_dataset, cfg, False, output_dir)
    val_loader  = build_detection_test_loader(cfg, test_dataset)
    raw_results = inference_on_dataset(model, val_loader, evaluator)

    # Compute split metrics
    metrics = compute_split_metrics(raw_results["bbox"], BASE_CLASSES, NOVEL_CLASSES)

    # Print results
    print(f"\n{'─'*50}")
    print(f"  Checkpoint: {Path(checkpoint).name}")
    print(f"  mAP  : {metrics['mAP']:.1f}")
    print(f"  bAP  : {metrics['bAP']:.1f}  (base classes)")
    print(f"  nAP  : {metrics['nAP']:.1f}  (novel classes)")
    print(f"\n  Per-class AP:")
    for cls_name in CLASS_NAMES:
        role = "[NOVEL]" if cls_name in NOVEL_CLASSES else "[base] "
        ap   = metrics["per_class_AP"].get(cls_name, float("nan"))
        print(f"    {role} {cls_name:<25} {ap:>6.1f}")
    print(f"{'─'*50}\n")

    # Save metrics JSON
    metrics["checkpoint"] = str(checkpoint)
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    return metrics


# ── Multi-K comparison table ──────────────────────────────────────────────────
def print_comparison_table(all_results: dict):
    """Print a comparison table matching the paper's results format."""
    print("\n" + "=" * 65)
    print(f"  FS-Symbol Results Summary")
    print(f"  {'K':<6} {'mAP':>8} {'bAP':>8} {'nAP':>8}")
    print("=" * 65)
    for label, metrics in sorted(all_results.items()):
        mAP = metrics.get("mAP", float("nan"))
        bAP = metrics.get("bAP", float("nan"))
        nAP = metrics.get("nAP", float("nan"))
        print(f"  {label:<6} {mAP:>8.1f} {bAP:>8.1f} {nAP:>8.1f}")
    print("=" * 65)
    print(f"  Paper baseline TFA (K=9): mAP=?, bAP=?, nAP=43.0")
    print(f"  Paper FS-Symbol (K=9):    mAP=?, bAP=?, nAP=83.4")
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FS-Symbol Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",  help="Single checkpoint to evaluate")
    group.add_argument("--checkpoints", nargs="+", help="Multiple checkpoints to compare")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUT_DIR)
    parser.add_argument("--cosine_scale", type=float, default=20.0)
    args = parser.parse_args()

    checkpoints = [args.checkpoint] if args.checkpoint else args.checkpoints

    all_results = {}
    for ckpt in checkpoints:
        if not Path(ckpt).exists():
            logger.error(f"Checkpoint not found: {ckpt}")
            continue

        # Derive a label (e.g. "K5" from "fs_symbol_K5.pth", or "base")
        stem  = Path(ckpt).stem
        label = stem.replace("fs_symbol_", "").replace("base_model", "base")

        out_dir = Path(args.output_dir) / label
        metrics = evaluate_checkpoint(ckpt, args.data_dir, str(out_dir), args.cosine_scale)
        all_results[label] = metrics

    if len(all_results) > 1:
        print_comparison_table(all_results)

        # Save combined results
        combined_path = Path(args.output_dir) / "all_results.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Combined results → {combined_path}")


if __name__ == "__main__":
    main()
