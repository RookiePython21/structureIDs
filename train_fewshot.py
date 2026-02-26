"""
train_fewshot.py — Stage 2: Few-shot fine-tuning (FS-Symbol).

Loads the Stage 1 base model, applies two-stage TFA:
  - Freezes backbone (preserves diverse base-class features)
  - Unfreezes RPN + ROI heads (generates better novel-class proposals)
  - Reinitializes novel class weights in cosine similarity classifier
  - Fine-tunes on K-shot balanced support set (base + novel)

Run for all K values:
    python train_fewshot.py --k_shots 1 2 3 5 9 \\
        --base_ckpt checkpoints/base_model.pth

Run for a single K:
    python train_fewshot.py --k_shots 5 \\
        --base_ckpt checkpoints/base_model.pth

Colab usage:
    !python train_fewshot.py \\
        --base_ckpt /content/drive/MyDrive/structureIDs/checkpoints/base_model.pth \\
        --data_dir  /content/drive/MyDrive/structureIDs/data \\
        --output_dir /content/drive/MyDrive/structureIDs/checkpoints
"""

import os
import sys
import copy
import logging
from pathlib import Path

# ── Google Colab detection ────────────────────────────────────────────────────
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/structureIDs"
    DATA_DIR   = os.environ.get("STRUCTIDS_DATA_DIR",  f"{DRIVE_ROOT}/data")
    BASE_CKPT  = os.environ.get("STRUCTIDS_BASE_CKPT", f"{DRIVE_ROOT}/checkpoints/base_model.pth")
    CKPT_DIR   = os.environ.get("STRUCTIDS_CKPT_DIR",  f"{DRIVE_ROOT}/checkpoints")
    sys.path.insert(0, DRIVE_ROOT)
else:
    _here     = Path(__file__).parent
    DATA_DIR  = os.environ.get("STRUCTIDS_DATA_DIR",  str(_here / "data"))
    BASE_CKPT = os.environ.get("STRUCTIDS_BASE_CKPT", str(_here / "checkpoints" / "base_model.pth"))
    CKPT_DIR  = os.environ.get("STRUCTIDS_CKPT_DIR",  str(_here / "checkpoints"))

# ── Detectron2 imports ────────────────────────────────────────────────────────
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

from model.fs_symbol import (
    build_fs_symbol_model,
    replace_classifier_with_cosine,
    setup_stage2,
    load_checkpoint,
    CLASS_NAMES,
    NOVEL_CLASS_IDS,
)

setup_logger()
logger = logging.getLogger("fs_symbol.train_fewshot")


# ── Dataset registration ───────────────────────────────────────────────────────
_registered_datasets = set()

def register_support_dataset(k: int, data_dir: str, patches_dir: str = None):
    """Register the K-shot support set as a Detectron2 dataset."""
    name     = f"civil_structures_k{k}_support"
    ann_file = Path(data_dir) / "support_sets" / f"k{k}" / "support.json"
    img_dir  = Path(data_dir) / "support_sets" / f"k{k}" / "images"

    if not ann_file.exists():
        raise FileNotFoundError(
            f"Support set not found: {ann_file}\n"
            f"Run: python scripts/generate_support_sets.py first."
        )

    if name not in _registered_datasets:
        register_coco_instances(name, {}, str(ann_file), str(img_dir))
        _registered_datasets.add(name)

    return name


def register_test_dataset(data_dir: str):
    name = "civil_structures_test"
    if name not in _registered_datasets:
        ann_dir   = Path(data_dir) / "annotations"
        patch_dir = Path(data_dir) / "patches"
        register_coco_instances(name, {}, str(ann_dir / "test.json"), str(patch_dir / "test"))
        _registered_datasets.add(name)
    return name


# ── Config builder ────────────────────────────────────────────────────────────
def build_fewshot_cfg(
    k: int,
    data_dir: str,
    base_ckpt: str,
    output_dir: str,
    extra_opts: list = None,
) -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    support_dataset = register_support_dataset(k, data_dir)
    test_dataset    = register_test_dataset(data_dir)

    # ── Datasets ──
    cfg.DATASETS.TRAIN = (support_dataset,)
    cfg.DATASETS.TEST  = (test_dataset,)

    # ── Model ── (no pretrained weights; we load Stage 1 checkpoint manually)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES   = len(CLASS_NAMES)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.WEIGHTS                 = ""   # loaded via DetectionCheckpointer

    # ── Input: fixed 640×640, no augmentation (K is tiny) ──
    cfg.INPUT.MIN_SIZE_TRAIN = (640,)
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MIN_SIZE_TEST  = 640
    cfg.INPUT.MAX_SIZE_TEST  = 640
    cfg.INPUT.RANDOM_FLIP    = "none"

    # ── Solver (Stage 2) — paper: 3200 iterations, LR=0.0005 ──
    cfg.SOLVER.BASE_LR        = 0.0005
    cfg.SOLVER.WARMUP_ITERS   = 200
    cfg.SOLVER.WARMUP_FACTOR  = 1.0 / 1000
    cfg.SOLVER.IMS_PER_BATCH  = 4
    cfg.SOLVER.MAX_ITER       = 3200
    cfg.SOLVER.STEPS          = (2400, 3000)
    cfg.SOLVER.GAMMA          = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 400

    # ── Evaluation ──
    cfg.TEST.EVAL_PERIOD          = 400
    cfg.TEST.DETECTIONS_PER_IMAGE = 300

    # ── Output ──
    k_output_dir = str(Path(output_dir) / f"fewshot_k{k}")
    cfg.OUTPUT_DIR = k_output_dir
    os.makedirs(k_output_dir, exist_ok=True)

    # Single GPU adjustment
    if IN_COLAB or torch.cuda.device_count() <= 1:
        cfg.SOLVER.IMS_PER_BATCH = 2

    if extra_opts:
        cfg.merge_from_list(extra_opts)

    cfg.freeze()
    return cfg


# ── Evaluator-aware trainer ────────────────────────────────────────────────────
class FewShotTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# ── Per-K training loop ────────────────────────────────────────────────────────
def train_one_k(k: int, base_ckpt: str, data_dir: str, output_dir: str, opts: list):
    logger.info(f"\n{'='*60}")
    logger.info(f"  Stage 2 — K={k} few-shot fine-tuning")
    logger.info(f"{'='*60}")

    cfg = build_fewshot_cfg(k, data_dir, base_ckpt, output_dir, opts)

    # Build model + replace classifier
    trainer = FewShotTrainer(cfg)
    trainer.model = replace_classifier_with_cosine(trainer.model, scale=20.0)

    # Load Stage 1 checkpoint (backbone + all weights)
    logger.info(f"Loading base checkpoint: {base_ckpt}")
    DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        base_ckpt, resume=False
    )

    # Apply Stage 2 setup: freeze backbone, reinit novel weights
    setup_stage2(trainer.model, novel_class_ids=NOVEL_CLASS_IDS)

    logger.info(f"Fine-tuning for {cfg.SOLVER.MAX_ITER} iterations on K={k} support set...")
    trainer.train()

    # Save final checkpoint with descriptive name
    final_ckpt = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    named_ckpt = Path(output_dir) / f"fs_symbol_K{k}.pth"
    if final_ckpt.exists():
        import shutil
        shutil.copy2(str(final_ckpt), str(named_ckpt))
        logger.info(f"K={k} model saved → {named_ckpt}")

    return str(named_ckpt)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="FS-Symbol Stage 2: Few-Shot Fine-Tuning")
    parser.add_argument("--k_shots",    nargs="+", type=int, default=[1, 2, 3, 5, 9])
    parser.add_argument("--base_ckpt",  default=BASE_CKPT,
                        help="Path to Stage 1 base model checkpoint")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--output_dir", default=CKPT_DIR)
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                        help="Detectron2 cfg overrides")
    args = parser.parse_args()

    if not Path(args.base_ckpt).exists():
        logger.error(
            f"Base checkpoint not found: {args.base_ckpt}\n"
            f"Run train_base.py first to generate it."
        )
        sys.exit(1)

    logger.info(f"Base checkpoint : {args.base_ckpt}")
    logger.info(f"Data dir        : {args.data_dir}")
    logger.info(f"Output dir      : {args.output_dir}")
    logger.info(f"K values        : {args.k_shots}")
    logger.info(f"CUDA devices    : {torch.cuda.device_count()}")

    results = {}
    for k in args.k_shots:
        ckpt_path = train_one_k(k, args.base_ckpt, args.data_dir, args.output_dir, args.opts)
        results[k] = ckpt_path

    logger.info("\nAll Stage 2 training complete.")
    logger.info("Checkpoints:")
    for k, path in results.items():
        logger.info(f"  K={k}: {path}")
    logger.info("\nNext step: Run evaluate.py to compute nAP / bAP / mAP.")


if __name__ == "__main__":
    main()
