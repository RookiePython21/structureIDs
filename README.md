# FS-Symbol: Civil Engineering Structure Detection

Few-shot symbol detection on civil engineering drawings.
Based on *"Few-Shot Symbol Detection in Engineering Drawings"* (Jamieson et al., 2024).
Architecture: Faster R-CNN + ResNet-101 + FPN + cosine similarity classifier, trained with Two-Stage TFA.

---

## What this detects

| Class | Role | Training examples |
|---|---|---|
| `square_structure` | Base | 306 |
| `circle_structure` | Base | 201 |
| `headwall` | Base | 69 |
| `rip_rap` | **Novel** | 33 |
| `title_exclusion` | Masked out | 45 |
| `notes` | Masked out | 58 |
| `legend` | Masked out | 27 |
| `compass` | Masked out | 46 |
| `site_map` | Masked out | 13 |

Masked-out classes are white-filled before patching — the model never sees or predicts them.

---

## Project structure

```
structureIDs/
├── annotations/
│   ├── annotations.xml        ← CVAT XML (798 labeled regions, 45 images)
│   └── images/                ← 45 high-res PNGs (14400×9600 or 13600×8800)
├── scripts/
│   ├── preprocess.py          ← Step 1: CVAT XML → COCO patch dataset
│   ├── generate_support_sets.py  ← Step 2: K-shot support sets
│   └── visualize.py           ← Verification helper
├── model/
│   └── fs_symbol.py           ← CosineSimLinear + Stage 2 helpers
├── config/
│   └── fs_symbol_config.yaml  ← Detectron2 config reference
├── train_base.py              ← Step 3: Stage 1 base training
├── train_fewshot.py           ← Step 4: Stage 2 few-shot fine-tuning
└── evaluate.py                ← Step 5: mAP / bAP / nAP evaluation
```

---

## Getting started in Google Colab

### Step 0 — Open a new Colab notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook**
3. Change runtime: **Runtime → Change runtime type → T4 GPU**

---

### Step 1 — Mount Drive and set up paths

Paste this into the first cell and run it:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, sys

# ── Change this if your folder is named differently or in a subfolder ──
DRIVE_ROOT = '/content/drive/MyDrive/structureIDs'

os.environ['STRUCTIDS_DATA_DIR']  = f'{DRIVE_ROOT}/data'
os.environ['STRUCTIDS_CKPT_DIR']  = f'{DRIVE_ROOT}/checkpoints/base'
os.environ['STRUCTIDS_BASE_CKPT'] = f'{DRIVE_ROOT}/checkpoints/base_model.pth'

# Make the project importable
sys.path.insert(0, DRIVE_ROOT)
os.chdir(DRIVE_ROOT)

print('Working directory:', os.getcwd())
print('Files:', os.listdir(DRIVE_ROOT))
```

---

### Step 2 — Install dependencies

Paste into a new cell. This takes ~3 minutes on a fresh Colab session:

```python
# Install Detectron2 (must match Colab's PyTorch version)
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')

# Detectron2 wheel — Colab currently uses PyTorch 2.x + CUDA 12.1
!pip install 'git+https://github.com/facebookresearch/detectron2.git' -q

# Other dependencies
!pip install lxml pycocotools opencv-python -q

print('Done. Restart runtime if prompted.')
```

> **Important:** If Colab says "restart runtime", click **Runtime → Restart runtime**, then re-run Steps 1 and 2 (the mount and path setup cells).

---

### Step 3 — Preprocess images into patches

This is the slowest step (~30–60 minutes for 45 large images). Run once and the output is saved to Drive.

```python
!python scripts/preprocess.py \
    --annotations annotations/annotations.xml \
    --images_dir   annotations/images \
    --output_dir   data \
    --patch_size   640 \
    --overlap      320 \
    --seed         42 \
    --train_ratio  0.8
```

**Expected output:**
```
Split: 36 train / 9 test images
[train] Annotated patches: ~7,000–11,000
[test]  Annotated patches: ~1,700–2,800
Base training JSON → data/annotations/train_base.json
```

Files created:
```
data/
├── patches/train/     ← ~7,000–11,000 JPEG patches
├── patches/test/      ← ~1,700–2,800 JPEG patches
├── annotations/
│   ├── train.json     ← full COCO (base + novel)
│   ├── train_base.json  ← base classes only (Stage 1 input)
│   └── test.json
└── dataset_config.json
```

---

### Step 4 — Verify patches visually

Check that bounding boxes look correct before training:

```python
# Print annotation counts per class
!python scripts/visualize.py \
    --coco_json data/annotations/train.json \
    --stats

# Save 5 random annotated patches as images you can inspect
!python scripts/visualize.py \
    --coco_json   data/annotations/train.json \
    --patches_dir data/patches/train \
    --n           5 \
    --output_dir  data/visualizations

# Display one in the notebook
from IPython.display import Image
import glob
imgs = glob.glob('data/visualizations/*.jpg')
Image(imgs[0])
```

---

### Step 5 — Generate K-shot support sets

Quick step (~1 minute):

```python
!python scripts/generate_support_sets.py \
    --train_json  data/annotations/train.json \
    --patches_dir data/patches/train \
    --output_dir  data/support_sets \
    --k_shots     1 2 3 5 9 \
    --seed        42
```

Creates `data/support_sets/k1/`, `k2/`, `k3/`, `k5/`, `k9/` — each with a `support.json` and copied patch images.

---

### Step 6 — Stage 1: Base training

Trains the full model on base classes (~2–4 hours on Colab T4):

```python
!python train_base.py \
    --data_dir   data \
    --output_dir /content/drive/MyDrive/structureIDs/checkpoints/base
```

Checkpoints save to Drive every 2,000 iterations so you don't lose progress if Colab disconnects.
Final model saved as `checkpoints/base_model.pth`.

> **Tip:** To resume after a disconnect:
> `!python train_base.py --resume --output_dir .../checkpoints/base`

---

### Step 7 — Stage 2: Few-shot fine-tuning

Fine-tune for each K value (~15–30 minutes per K):

```python
# All K values at once
!python train_fewshot.py \
    --k_shots    1 2 3 5 9 \
    --base_ckpt  /content/drive/MyDrive/structureIDs/checkpoints/base_model.pth \
    --data_dir   data \
    --output_dir /content/drive/MyDrive/structureIDs/checkpoints

# Or a single K to test quickly
!python train_fewshot.py \
    --k_shots   1 \
    --base_ckpt /content/drive/MyDrive/structureIDs/checkpoints/base_model.pth
```

Saves: `checkpoints/fs_symbol_K1.pth`, `fs_symbol_K2.pth`, ..., `fs_symbol_K9.pth`

---

### Step 8 — Evaluate

```python
# Evaluate all K models and print a comparison table
!python evaluate.py \
    --checkpoints \
        checkpoints/fs_symbol_K1.pth \
        checkpoints/fs_symbol_K2.pth \
        checkpoints/fs_symbol_K3.pth \
        checkpoints/fs_symbol_K5.pth \
        checkpoints/fs_symbol_K9.pth \
    --data_dir   data \
    --output_dir results

# Or evaluate just one
!python evaluate.py \
    --checkpoint checkpoints/fs_symbol_K9.pth \
    --data_dir   data \
    --output_dir results/k9
```

**Expected output format:**
```
─────────────────────────────────────────────────
  Checkpoint: fs_symbol_K9.pth
  mAP  : 75.3
  bAP  : 82.1   (base classes)
  nAP  : 61.4   (novel classes)

  Per-class AP:
    [base]  square_structure          88.2
    [base]  circle_structure          91.0
    [base]  headwall                  74.3
    [base]  notes                     83.5
    [base]  compass                   73.4
    [NOVEL] rip_rap                   58.9
    [NOVEL] site_map                  63.8
─────────────────────────────────────────────────

═══════════════════════════════════════════════════
  FS-Symbol Results Summary
  K       mAP     bAP     nAP
═══════════════════════════════════════════════════
  K1      ...     ...     ...
  K9      ...     ...     ...
═══════════════════════════════════════════════════
  Paper FS-Symbol (K=9): nAP=83.4
═══════════════════════════════════════════════════
```

---

## Quick smoke test (verify training works before full run)

Before committing to the full training run, test the pipeline with 2 iterations:

```python
# Stage 1 smoke test — 2 iterations
!python train_base.py \
    --output_dir /tmp/test_base \
    --opts SOLVER.MAX_ITER 2 SOLVER.CHECKPOINT_PERIOD 2 TEST.EVAL_PERIOD 2

# Stage 2 smoke test — K=1, 2 iterations
!python train_fewshot.py \
    --k_shots   1 \
    --base_ckpt /tmp/test_base/model_final.pth \
    --output_dir /tmp/test_fewshot \
    --opts SOLVER.MAX_ITER 2
```

If both run without errors, the full pipeline is working.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: model` | Make sure `sys.path.insert(0, DRIVE_ROOT)` ran and `os.chdir(DRIVE_ROOT)` set correctly |
| `detectron2 not found` | Re-run the pip install cell; if runtime was restarted, re-run mount + install |
| `FileNotFoundError: annotations.xml` | Run `os.chdir(DRIVE_ROOT)` first, or use absolute paths |
| Colab disconnects mid-training | Stage 1: re-run with `--resume`. Stage 2: re-run the specific `--k_shots` that didn't finish |
| `CUDA out of memory` | Add `--opts SOLVER.IMS_PER_BATCH 2` to reduce batch size |
| `WARNING: Not found — some_image.png` | 2 images are missing from `annotations/images/`. This is expected — they're skipped |
| Support set has fewer than K instances | Some novel classes have very few examples. The script uses all available with a warning |

---

## Key hyperparameters

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Backbone | ResNet-101 + FPN | **Frozen** |
| RPN | Trainable | Trainable |
| ROI Heads | Trainable | Trainable |
| Classifier | CosineSimLinear (scale=20) | CosineSimLinear (scale=20) |
| Learning rate | 0.01 (0.0025 on Colab) | 0.0005 |
| Iterations | 17,000 | 3,200 |
| Warmup | 500 iters | 200 iters |
| Batch size | 8 (2 on Colab) | 4 (2 on Colab) |
| Multiscale training | Yes (480–800px) | No (640px fixed) |
| Flip augmentation | None | None |
