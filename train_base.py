"""
train_base.py — Stage 1: Full base-class training (FS-Symbol).

Trains the complete Faster R-CNN + ResNet-101 + FPN model on base-class
patches. Saves a checkpoint for Stage 2 few-shot fine-tuning.

Run locally:
    python train_base.py [--opts KEY VALUE ...]

Run in Google Colab:
    1. Mount Drive, clone repo, install deps (see cells below)
    2. !python train_base.py --output_dir /content/drive/MyDrive/structureIDs/checkpoints/base

Key settings (override with --opts or edit config YAML):
    MAX_ITER       = 17000  (~17 epochs over typical patch count)
    BASE_LR        = 0.01   (linear scaling: 0.01 * batch_size / 8)
    CHECKPOINT     = every 2000 iterations
    DATA           = data/annotations/train_base.json

Environment variables:
    STRUCTIDS_DATA_DIR   → root of data/ directory (default: ./data)
    STRUCTIDS_CKPT_DIR   → checkpoint output directory (default: ./checkpoints/base)
"""

import os
import sys
import logging
from pathlib import Path

# ── Google Colab detection & Drive mount ──────────────────────────────────────
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/structureIDs"
    DATA_DIR   = os.environ.get("STRUCTIDS_DATA_DIR",  f"{DRIVE_ROOT}/data")
    CKPT_DIR   = os.environ.get("STRUCTIDS_CKPT_DIR",  f"{DRIVE_ROOT}/checkpoints/base")
    # Make sure the project root is on sys.path so model/ is importable
    sys.path.insert(0, DRIVE_ROOT)
else:
    _here    = Path(__file__).parent
    DATA_DIR = os.environ.get("STRUCTIDS_DATA_DIR", str(_here / "data"))
    CKPT_DIR = os.environ.get("STRUCTIDS_CKPT_DIR", str(_here / "checkpoints" / "base"))

# ── Detectron2 imports ────────────────────────────────────────────────────────
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

from model.fs_symbol import (
    build_fs_symbol_model,
    unfreeze_all,
    CLASS_NAMES,
)

setup_logger()
logger = logging.getLogger("fs_symbol.train_base")

# ── Dataset registration ───────────────────────────────────────────────────────
def register_datasets(data_dir: str):
    """Register base training + full test datasets in Detectron2's catalog."""
    data_dir  = Path(data_dir)
    ann_dir   = data_dir / "annotations"
    patch_dir = data_dir / "patches"

    # Stage 1 uses base-class-only patches
    register_coco_instances(
        "civil_structures_train_base",
        {},
        str(ann_dir / "train_base.json"),
        str(patch_dir / "train"),
    )
    # Full test set (base + novel) for evaluation
    register_coco_instances(
        "civil_structures_test",
        {},
        str(ann_dir / "test.json"),
        str(patch_dir / "test"),
    )
    logger.info("Datasets registered: civil_structures_train_base, civil_structures_test")


# ── Evaluator-aware trainer ────────────────────────────────────────────────────
class BaseTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# ── Config builder ────────────────────────────────────────────────────────────
def build_base_cfg(data_dir: str, output_dir: str, extra_opts: list = None) -> "CfgNode":
    cfg = get_cfg()

    # Start from the D2 model zoo R-101 FPN config
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    # ── Datasets ──
    cfg.DATASETS.TRAIN = ("civil_structures_train_base",)
    cfg.DATASETS.TEST  = ("civil_structures_test",)

    # ── Model ──
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)   # 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )

    # ── Input: multiscale, no flips (text must remain readable) ──
    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 640
    cfg.INPUT.MAX_SIZE_TEST  = 1333
    cfg.INPUT.RANDOM_FLIP    = "none"

    # ── Solver (Stage 1) ──
    cfg.SOLVER.BASE_LR        = 0.01
    cfg.SOLVER.MOMENTUM       = 0.9
    cfg.SOLVER.WEIGHT_DECAY   = 0.0001
    cfg.SOLVER.WARMUP_ITERS   = 500
    cfg.SOLVER.WARMUP_FACTOR  = 1.0 / 1000
    cfg.SOLVER.IMS_PER_BATCH  = 8      # 2 images/GPU × 4 GPUs; adjust for Colab
    cfg.SOLVER.MAX_ITER       = 17000
    cfg.SOLVER.STEPS          = (10000, 14000)
    cfg.SOLVER.GAMMA          = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    # ── Dataloader ──
    cfg.DATALOADER.NUM_WORKERS = 2

    # ── Evaluation ──
    cfg.TEST.EVAL_PERIOD         = 2000
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    # ── Output ──
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Single-GPU Colab adjustment
    if IN_COLAB or torch.cuda.device_count() <= 1:
        cfg.SOLVER.IMS_PER_BATCH = 2          # 1 GPU
        cfg.SOLVER.BASE_LR       = 0.0025     # linear scaling rule: 0.01 * 2/8

    # Apply any --opts overrides
    if extra_opts:
        cfg.merge_from_list(extra_opts)

    cfg.freeze()
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FS-Symbol Stage 1: Base Training")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--output_dir", default=CKPT_DIR)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from last checkpoint in output_dir")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                        help="Detectron2 cfg overrides, e.g. SOLVER.MAX_ITER 100")
    args = parser.parse_args()

    logger.info(f"Data dir    : {args.data_dir}")
    logger.info(f"Output dir  : {args.output_dir}")
    logger.info(f"CUDA devices: {torch.cuda.device_count()}")

    register_datasets(args.data_dir)
    cfg = build_base_cfg(args.data_dir, args.output_dir, args.opts)

    # Build model + replace FC classifier with cosine similarity
    # (DefaultTrainer.build_model is called internally; we patch it below)
    trainer = BaseTrainer(cfg)
    replace_cls = lambda model: __import__("model.fs_symbol", fromlist=["replace_classifier_with_cosine"]).replace_classifier_with_cosine(model)
    trainer.model = replace_cls(trainer.model)
    unfreeze_all(trainer.model)

    # Load pretrained weights (ImageNet via D2 model zoo)
    trainer.resume_or_load(resume=args.resume)

    logger.info("Starting Stage 1 base training...")
    trainer.train()

    # Copy final checkpoint to a predictable name for Stage 2
    final_ckpt = Path(args.output_dir) / "model_final.pth"
    base_ckpt  = Path(args.output_dir).parent / "base_model.pth"
    if final_ckpt.exists():
        import shutil
        shutil.copy2(str(final_ckpt), str(base_ckpt))
        logger.info(f"Base model saved → {base_ckpt}")

    logger.info("Stage 1 complete. Use train_fewshot.py for Stage 2.")


if __name__ == "__main__":
    main()
