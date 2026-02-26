# FS-Symbol: Civil Engineering Structure Detection

Few-shot symbol detection on civil engineering drawings.
Based on *"Few-Shot Symbol Detection in Engineering Drawings"* (Jamieson et al., 2024).
Architecture: Faster R-CNN + ResNet-101 + FPN + cosine similarity classifier, trained with Two-Stage TFA.

**Repo:** https://github.com/RookiePython21/structureIDs

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
│   ├── annotations.xml        ← CVAT XML (798 labeled regions, 45 images) — tracked in git
│   └── images/                ← 45 high-res PNGs (~538MB) — stored on Google Drive only
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

### Step 1 — Clone the repo and pull images from Drive

Paste this into the first cell and run it:

```python
# Clone the repo (code + annotations XML)
!git clone https://github.com/RookiePython21/structureIDs.git
%cd structureIDs

import os, sys
sys.path.insert(0, '/content/structureIDs')

# Mount Drive — images are too large for GitHub so they live here
from google.colab import drive
drive.mount('/content/drive')

# Copy images from Drive into the cloned repo
import shutil
src = '/content/drive/MyDrive/structureIDs/annotations/images'
dst = 'annotations/images'
os.makedirs(dst, exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
print('Images copied:', len(os.listdir(dst)), 'files')

# Point checkpoints and data at Drive so they persist across sessions
os.environ['STRUCTIDS_DATA_DIR']  = '/content/drive/MyDrive/structureIDs/data'
os.environ['STRUCTIDS_CKPT_DIR']  = '/content/drive/MyDrive/structureIDs/checkpoints/base'
os.environ['STRUCTIDS_BASE_CKPT'] = '/content/drive/MyDrive/structureIDs/checkpoints/base_model.pth'
```

> **Why this split?** Code lives on GitHub (easy to update with `git pull`). Images live on Drive (too large for GitHub at 538MB). Patches and checkpoints also write to Drive so they survive Colab disconnects.

---

### Step 2 — Install dependencies

```python
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')

!pip install 'git+https://github.com/facebookresearch/detectron2.git' -q
!pip install lxml pycocotools opencv-python -q

print('Done. Restart runtime if prompted.')
```

> **If Colab prompts a restart:** click **Runtime → Restart runtime**, then re-run Step 1 and Step 2 before continuing.

---

### Step 3 — Preprocess images into patches

Run once (~30–60 minutes). Output saves to Drive so you never need to repeat it.

```python
!python scripts/preprocess.py \
    --annotations annotations/annotations.xml \
    --images_dir   annotations/images \
    --output_dir   /content/drive/MyDrive/structureIDs/data \
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

Files created on Drive:
```
structureIDs/data/
├── patches/train/       ← ~7,000–11,000 JPEG patches
├── patches/test/        ← ~1,700–2,800 JPEG patches
├── annotations/
│   ├── train.json       ← full COCO (base + novel)
│   ├── train_base.json  ← base classes only (Stage 1 input)
│   └── test.json
└── dataset_config.json
```

---

### Step 4 — Verify patches visually

```python
# Print annotation counts per class
!python scripts/visualize.py \
    --coco_json /content/drive/MyDrive/structureIDs/data/annotations/train.json \
    --stats

# Save 5 random annotated patches as images
!python scripts/visualize.py \
    --coco_json   /content/drive/MyDrive/structureIDs/data/annotations/train.json \
    --patches_dir /content/drive/MyDrive/structureIDs/data/patches/train \
    --n           5 \
    --output_dir  /content/drive/MyDrive/structureIDs/data/visualizations

# Display one in the notebook
from IPython.display import Image
import glob
imgs = glob.glob('/content/drive/MyDrive/structureIDs/data/visualizations/*.jpg')
Image(imgs[0])
```

---

### Step 5 — Generate K-shot support sets

Quick step (~1 minute):

```python
!python scripts/generate_support_sets.py \
    --train_json  /content/drive/MyDrive/structureIDs/data/annotations/train.json \
    --patches_dir /content/drive/MyDrive/structureIDs/data/patches/train \
    --output_dir  /content/drive/MyDrive/structureIDs/data/support_sets \
    --k_shots     1 2 3 5 9 \
    --seed        42
```

---

### Step 6 — Stage 1: Base training

Trains the full model on base classes (~2–4 hours on Colab T4). Checkpoints save to Drive every 2,000 iterations.

```python
!python train_base.py \
    --data_dir   /content/drive/MyDrive/structureIDs/data \
    --output_dir /content/drive/MyDrive/structureIDs/checkpoints/base
```

> **Resume after a disconnect:**
> `!python train_base.py --resume --output_dir /content/drive/MyDrive/structureIDs/checkpoints/base`

---

### Step 7 — Stage 2: Few-shot fine-tuning

~15–30 minutes per K value:

```python
# All K values at once
!python train_fewshot.py \
    --k_shots    1 2 3 5 9 \
    --base_ckpt  /content/drive/MyDrive/structureIDs/checkpoints/base_model.pth \
    --data_dir   /content/drive/MyDrive/structureIDs/data \
    --output_dir /content/drive/MyDrive/structureIDs/checkpoints

# Or a single K to test quickly
!python train_fewshot.py \
    --k_shots   1 \
    --base_ckpt /content/drive/MyDrive/structureIDs/checkpoints/base_model.pth \
    --data_dir  /content/drive/MyDrive/structureIDs/data \
    --output_dir /content/drive/MyDrive/structureIDs/checkpoints
```

Saves: `checkpoints/fs_symbol_K1.pth`, `fs_symbol_K2.pth`, ..., `fs_symbol_K9.pth`

---

### Step 8 — Evaluate

```python
# Evaluate all K models and print a comparison table
!python evaluate.py \
    --checkpoints \
        /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K1.pth \
        /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K2.pth \
        /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K3.pth \
        /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K5.pth \
        /content/drive/MyDrive/structureIDs/checkpoints/fs_symbol_K9.pth \
    --data_dir   /content/drive/MyDrive/structureIDs/data \
    --output_dir /content/drive/MyDrive/structureIDs/results
```

**Expected output format:**
```
─────────────────────────────────────────────────
  Checkpoint: fs_symbol_K9.pth
  mAP  : 75.3
  bAP  : 82.1   (base classes)
  nAP  : 61.4   (novel classes)

  Per-class AP:
    [base]  square_structure     88.2
    [base]  circle_structure     91.0
    [base]  headwall             74.3
    [NOVEL] rip_rap              58.9
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

```python
# Stage 1 smoke test — 2 iterations
!python train_base.py \
    --data_dir   /content/drive/MyDrive/structureIDs/data \
    --output_dir /tmp/test_base \
    --opts SOLVER.MAX_ITER 2 SOLVER.CHECKPOINT_PERIOD 2 TEST.EVAL_PERIOD 2

# Stage 2 smoke test — K=1, 2 iterations
!python train_fewshot.py \
    --k_shots    1 \
    --base_ckpt  /tmp/test_base/model_final.pth \
    --data_dir   /content/drive/MyDrive/structureIDs/data \
    --output_dir /tmp/test_fewshot \
    --opts SOLVER.MAX_ITER 2
```

---

## Keeping code up to date

When you change code locally and push to GitHub:
```bash
# Local machine
git add -A && git commit -m "your message" && git push
```

Then in Colab pull the latest before training:
```python
!git pull
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: model` | Make sure `sys.path.insert(0, '/content/structureIDs')` ran |
| `detectron2 not found` | Re-run the pip install cell; if runtime restarted, re-run Steps 1 & 2 first |
| `FileNotFoundError: annotations/images` | The images copy from Drive step (Step 1) didn't finish — re-run it |
| `FileNotFoundError: annotations.xml` | Run `%cd /content/structureIDs` to fix the working directory |
| Colab disconnects mid-training | Stage 1: re-run with `--resume`. Stage 2: re-run the specific `--k_shots` value |
| `CUDA out of memory` | Add `--opts SOLVER.IMS_PER_BATCH 2` to reduce batch size |
| `WARNING: Not found — some_image.png` | 2 images are missing from the dataset — expected, they are skipped |
| Support set has fewer than K instances | `rip_rap` has only 33 examples total. Script uses all available with a warning |

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
