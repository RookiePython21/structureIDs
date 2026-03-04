"""
scripts/augment.py — Offline synthetic data augmentation for FS-Symbol patches.

Takes the output of preprocess.py and generates augmented copies of every patch.
Augmentations are applied per-image with correct bbox transforms so the COCO
annotation JSON stays valid.

Augmentations
─────────────
  Geometric   : horizontal flip, vertical flip, 90 / 180 / 270° rotation, zoom
  Photometric : brightness + contrast jitter, Gaussian blur, salt-and-pepper noise
  Structural  : elastic distortion  (optional — requires scipy)
  Copy-paste  : paste symbol crops from the library into patches, boosting rare
                (novel) class instances

Why each helps for civil-engineering drawing detection
───────────────────────────────────────────────────────
  • Flips / rotations  — drawings have no canonical orientation; scanners vary.
  • Zoom               — symbols appear at different scales across plan sets.
  • Brightness/contrast— scan quality varies enormously (old blueprints, PDFs).
  • Blur               — low-resolution or compressed scans.
  • Salt-and-pepper    — document scanning noise / compression artefacts.
  • Elastic distortion — paper warping from scanning or folding.
  • Copy-paste         — directly multiplies instances of rare / novel classes;
                         most impactful technique for few-shot detection.

Usage
─────
    # Generate augmented patches (stored separately):
    python scripts/augment.py \\
        --patches_dir      data/patches/train \\
        --ann_json         data/annotations/train.json \\
        --output_dir       data/patches/train_aug \\
        --out_json         data/annotations/train_aug.json \\
        --aug_factor       3 \\
        --copy_paste_factor 5 \\
        --seed             42

    # Produce a merged original+augmented training JSON for direct use:
    python scripts/augment.py ... \\
        --merge_json data/annotations/train_combined.json
"""

import json
import math
import os
import random
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Internal bbox helpers
# All augmentation functions use (x1, y1, x2, y2, cat_id) xyxy tuples.
# ---------------------------------------------------------------------------

def _coco_ann_to_box(ann):
    """COCO [x, y, w, h] → (x1, y1, x2, y2, cat_id)."""
    x, y, w, h = ann["bbox"]
    return (float(x), float(y), float(x + w), float(y + h), ann["category_id"])


def _box_to_coco_ann(x1, y1, x2, y2, cat_id, image_id, ann_id):
    w, h = x2 - x1, y2 - y1
    return {
        "id":          ann_id,
        "image_id":    image_id,
        "category_id": cat_id,
        "bbox":        [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
        "area":        round(w * h, 2),
        "iscrowd":     0,
    }


def _clip(x1, y1, x2, y2, S):
    return max(0.0, x1), max(0.0, y1), min(float(S), x2), min(float(S), y2)


def _valid(x1, y1, x2, y2, min_px=4):
    return (x2 - x1) >= min_px and (y2 - y1) >= min_px


# ---------------------------------------------------------------------------
# Geometric augmentations
# All assume square patches (S × S).
# ---------------------------------------------------------------------------

def aug_hflip(image, boxes, S):
    """Horizontal flip."""
    return (
        cv2.flip(image, 1),
        [(S - x2, y1, S - x1, y2, c) for x1, y1, x2, y2, c in boxes],
    )


def aug_vflip(image, boxes, S):
    """Vertical flip."""
    return (
        cv2.flip(image, 0),
        [(x1, S - y2, x2, S - y1, c) for x1, y1, x2, y2, c in boxes],
    )


def aug_rot90(image, boxes, S, k):
    """
    Rotate 90° CCW k times (k = 1, 2, 3).
    CCW 90° transform: (x, y) → (y, S - x).
    """
    img_out = np.rot90(image, k)
    new_boxes = []
    for (x1, y1, x2, y2, cat) in boxes:
        cur = (x1, y1, x2, y2)
        for _ in range(k % 4):
            cx1, cy1, cx2, cy2 = cur
            # CCW 90°: new bbox = (y1, S-x2, y2, S-x1)
            cur = (cy1, S - cx2, cy2, S - cx1)
        nx1, ny1, nx2, ny2 = _clip(*cur, S)
        if _valid(nx1, ny1, nx2, ny2):
            new_boxes.append((nx1, ny1, nx2, ny2, cat))
    return img_out, new_boxes


def aug_zoom(image, boxes, S, scale_range=(0.6, 0.9)):
    """
    Randomly crop a sub-region (scale × S) and resize back to S×S.
    Mimics the model seeing the same scene at a larger visual scale.
    Only boxes whose centre falls inside the crop are retained.
    """
    scale  = random.uniform(*scale_range)
    crop_s = max(1, int(S * scale))
    max_off = S - crop_s
    ox = random.randint(0, max_off)
    oy = random.randint(0, max_off)

    crop    = image[oy : oy + crop_s, ox : ox + crop_s]
    img_out = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
    ratio   = S / crop_s

    new_boxes = []
    for (x1, y1, x2, y2, cat) in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not (ox <= cx < ox + crop_s and oy <= cy < oy + crop_s):
            continue  # centre outside crop — discard (mirrors preprocess.py)
        nx1 = (x1 - ox) * ratio
        ny1 = (y1 - oy) * ratio
        nx2 = (x2 - ox) * ratio
        ny2 = (y2 - oy) * ratio
        nx1, ny1, nx2, ny2 = _clip(nx1, ny1, nx2, ny2, S)
        if _valid(nx1, ny1, nx2, ny2):
            new_boxes.append((nx1, ny1, nx2, ny2, cat))

    return img_out, new_boxes


# ---------------------------------------------------------------------------
# Photometric augmentations (bboxes pass through unchanged)
# ---------------------------------------------------------------------------

def aug_brightness_contrast(image, boxes, S):
    """Random brightness (+/- 40) and contrast (0.6 – 1.4×)."""
    alpha = random.uniform(0.6, 1.4)
    beta  = random.randint(-40, 40)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta), boxes


def aug_gaussian_blur(image, boxes, S):
    """Gaussian blur with kernel 3 or 5."""
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(image, (ksize, ksize), 0), boxes


def aug_salt_pepper(image, boxes, S, amount=0.008):
    """Salt-and-pepper noise to simulate scan artefacts."""
    img_out = image.copy()
    n = int(amount * S * S)
    # Salt
    ys = np.random.randint(0, S, n)
    xs = np.random.randint(0, S, n)
    img_out[ys, xs] = 255
    # Pepper
    ys = np.random.randint(0, S, n)
    xs = np.random.randint(0, S, n)
    img_out[ys, xs] = 0
    return img_out, boxes


# ---------------------------------------------------------------------------
# Elastic distortion (requires scipy)
# ---------------------------------------------------------------------------

def aug_elastic(image, boxes, S, alpha=35, sigma=5):
    """
    Smooth random displacement field applied to pixels and bbox corners.
    Simulates paper warping from scanning or folding.
    Requires scipy; skipped silently if not available.
    """
    if not HAS_SCIPY:
        return image, boxes

    rng = np.random.RandomState(random.randint(0, 99999))
    dx = gaussian_filter(rng.randn(S, S), sigma) * alpha
    dy = gaussian_filter(rng.randn(S, S), sigma) * alpha

    xs, ys = np.meshgrid(np.arange(S), np.arange(S))
    src_x  = np.clip(xs + dx, 0, S - 1)
    src_y  = np.clip(ys + dy, 0, S - 1)

    img_out = np.zeros_like(image)
    for ch in range(image.shape[2]):
        img_out[:, :, ch] = map_coordinates(
            image[:, :, ch],
            [src_y.ravel(), src_x.ravel()],
            order=1,
        ).reshape(S, S)

    # Transform bbox corners through the displacement field
    new_boxes = []
    for (x1, y1, x2, y2, cat) in boxes:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        new_corners = []
        for (cx, cy) in corners:
            ix = int(np.clip(round(cx), 0, S - 1))
            iy = int(np.clip(round(cy), 0, S - 1))
            new_corners.append((
                float(np.clip(cx + dx[iy, ix], 0, S)),
                float(np.clip(cy + dy[iy, ix], 0, S)),
            ))
        nxs = [p[0] for p in new_corners]
        nys = [p[1] for p in new_corners]
        nx1, ny1, nx2, ny2 = _clip(min(nxs), min(nys), max(nxs), max(nys), S)
        if _valid(nx1, ny1, nx2, ny2):
            new_boxes.append((nx1, ny1, nx2, ny2, cat))

    return img_out.astype(np.uint8), new_boxes


# ---------------------------------------------------------------------------
# Copy-paste augmentation
# ---------------------------------------------------------------------------

def build_crop_library(patches_dir: Path, coco: dict, padding: int = 4):
    """
    Extract every annotated symbol as an image crop.
    Returns {cat_id: [(crop_bgr, crop_w, crop_h), ...]}.
    """
    img_map  = {img["id"]: img for img in coco["images"]}
    library  = defaultdict(list)

    for ann in coco["annotations"]:
        img_info = img_map.get(ann["image_id"])
        if img_info is None:
            continue
        img_path = patches_dir / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        H, W  = image.shape[:2]
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue

        crop = image[y1:y2, x1:x2].copy()
        library[ann["category_id"]].append((crop, x2 - x1, y2 - y1))

    return library


def copy_paste_augment(
    image, boxes, S, crop_library, cat_ids, n_paste=2, max_iou=0.25
):
    """
    Paste up to n_paste crops from `crop_library[cat_ids]` into the image.
    Placement is rejected if it overlaps existing boxes above max_iou.
    """
    candidates = [
        (cat_id, info)
        for cat_id in cat_ids
        if cat_id in crop_library
        for info in crop_library[cat_id]
    ]
    if not candidates:
        return image.copy(), list(boxes)

    img_out   = image.copy()
    new_boxes = list(boxes)
    pasted    = 0
    attempts  = 0

    while pasted < n_paste and attempts < n_paste * 12:
        attempts += 1
        cat_id, (crop, cw, ch) = random.choice(candidates)

        # Optional small random scale of the crop (0.8 – 1.2×)
        scale = random.uniform(0.8, 1.2)
        cw2   = max(4, int(cw * scale))
        ch2   = max(4, int(ch * scale))
        if cw2 >= S or ch2 >= S:
            continue
        crop_r = cv2.resize(crop, (cw2, ch2), interpolation=cv2.INTER_LINEAR)

        px = random.randint(0, S - cw2)
        py = random.randint(0, S - ch2)

        nx1, ny1, nx2, ny2 = float(px), float(py), float(px + cw2), float(py + ch2)

        # IoU check against existing boxes
        crowded = False
        for (ex1, ey1, ex2, ey2, _) in new_boxes:
            ix1 = max(nx1, ex1); iy1 = max(ny1, ey1)
            ix2 = min(nx2, ex2); iy2 = min(ny2, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = cw2 * ch2 + (ex2 - ex1) * (ey2 - ey1) - inter
                if union > 0 and inter / union > max_iou:
                    crowded = True
                    break
        if crowded:
            continue

        img_out[py : py + ch2, px : px + cw2] = crop_r
        new_boxes.append((nx1, ny1, nx2, ny2, cat_id))
        pasted += 1

    return img_out, new_boxes


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

_GEO_AUGS   = ["hflip", "vflip", "rot90_1", "rot90_2", "rot90_3", "zoom"]
_PHOTO_AUGS = ["brightness_contrast", "gaussian_blur", "salt_pepper"]
_STRUCT_AUGS = ["elastic"] if HAS_SCIPY else []


def _apply(aug_name, image, boxes, S):
    """Dispatch a single named augmentation."""
    dispatch = {
        "hflip":               aug_hflip,
        "vflip":               aug_vflip,
        "rot90_1":             lambda i, b, s: aug_rot90(i, b, s, 1),
        "rot90_2":             lambda i, b, s: aug_rot90(i, b, s, 2),
        "rot90_3":             lambda i, b, s: aug_rot90(i, b, s, 3),
        "zoom":                aug_zoom,
        "brightness_contrast": aug_brightness_contrast,
        "gaussian_blur":       aug_gaussian_blur,
        "salt_pepper":         aug_salt_pepper,
        "elastic":             aug_elastic,
    }
    fn = dispatch.get(aug_name)
    return fn(image, boxes, S) if fn else (image, boxes)


def pick_and_apply(image, boxes, S):
    """
    Pick a random combination of augmentations:
      - One geometric  (always)
      - One photometric (70% chance)
      - One structural  (30% chance, scipy permitting)
    Returns (aug_image, aug_boxes, short_name_str).
    """
    chosen = [random.choice(_GEO_AUGS)]
    if random.random() < 0.70:
        chosen.append(random.choice(_PHOTO_AUGS))
    if _STRUCT_AUGS and random.random() < 0.30:
        chosen.append(random.choice(_STRUCT_AUGS))

    img_out, boxes_out = image.copy(), list(boxes)
    for aug in chosen:
        img_out, boxes_out = _apply(aug, img_out, boxes_out, S)

    return img_out, boxes_out, "+".join(chosen)


# ---------------------------------------------------------------------------
# Short name helper for filenames
# ---------------------------------------------------------------------------

_SHORT = {
    "hflip": "hf", "vflip": "vf",
    "rot90_1": "r1", "rot90_2": "r2", "rot90_3": "r3",
    "zoom": "zm",
    "brightness_contrast": "bc", "gaussian_blur": "gb", "salt_pepper": "sp",
    "elastic": "el",
}


def _short_name(aug_str):
    return "-".join(_SHORT.get(p, p) for p in aug_str.split("+"))


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def augment_dataset(
    patches_dir: str,
    ann_json: str,
    output_dir: str,
    out_json: str,
    aug_factor: int = 3,
    copy_paste_factor: int = 5,
    seed: int = 42,
    merge_json: str = None,
):
    random.seed(seed)
    np.random.seed(seed)

    patches_dir = Path(patches_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_json) as f:
        coco = json.load(f)

    S          = coco.get("info", {}).get("patch_size", 640)
    cat_map    = {c["id"]: c for c in coco["categories"]}
    novel_ids  = {c["id"] for c in coco["categories"] if c.get("role") == "novel"}
    base_ids   = {c["id"] for c in coco["categories"] if c.get("role") == "base"}
    all_cat_ids = set(cat_map.keys())

    print(f"Patch size    : {S}")
    print(f"Base classes  : {[cat_map[i]['name'] for i in sorted(base_ids)]}")
    print(f"Novel classes : {[cat_map[i]['name'] for i in sorted(novel_ids)]}")
    print(f"Elastic       : {'enabled' if HAS_SCIPY else 'disabled (pip install scipy)'}")

    # Build crop library for copy-paste
    print("\nBuilding copy-paste crop library...")
    crop_library = build_crop_library(patches_dir, coco)
    for cat_id, crops in sorted(crop_library.items()):
        print(f"  {cat_map[cat_id]['name']:<25} {len(crops)} crops")

    # Index annotations by image id
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    out_images      = []
    out_annotations = []
    out_img_id      = 1
    out_ann_id      = 1
    stats           = defaultdict(int)

    print(f"\nGenerating augmentations (aug_factor={aug_factor}, "
          f"copy_paste_factor={copy_paste_factor})...")

    for img_info in coco["images"]:
        img_path = patches_dir / img_info["file_name"]
        if not img_path.exists():
            print(f"  WARNING: missing {img_info['file_name']}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        stem  = Path(img_info["file_name"]).stem
        boxes = [_coco_ann_to_box(a) for a in img_to_anns[img_info["id"]]]
        novel_count = sum(1 for _, _, _, _, c in boxes if c in novel_ids)

        # ── Standard augmentations ────────────────────────────────────────
        for i in range(aug_factor):
            aug_img, aug_boxes, aug_name = pick_and_apply(image, boxes, S)
            if not aug_boxes:
                continue

            short  = _short_name(aug_name)
            fname  = f"{stem}_a{i}_{short}.jpg"
            cv2.imwrite(str(output_dir / fname), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            out_images.append({
                "id":           out_img_id,
                "file_name":    fname,
                "width":        S,
                "height":       S,
                "source_image": img_info.get("source_image", ""),
                "patch_origin": img_info.get("patch_origin", [0, 0]),
                "augmented":    True,
                "aug_type":     aug_name,
            })
            for (x1, y1, x2, y2, cat_id) in aug_boxes:
                out_annotations.append(
                    _box_to_coco_ann(x1, y1, x2, y2, cat_id, out_img_id, out_ann_id)
                )
                stats[cat_map[cat_id]["name"]] += 1
                out_ann_id += 1
            out_img_id += 1

        # ── Copy-paste augmentations ───────────────────────────────────────
        # Patches that already contain novel instances get more copies so the
        # model sees novel symbols in more spatial contexts.
        n_cp = copy_paste_factor if novel_count > 0 else max(1, copy_paste_factor // 3)
        # Prefer pasting novel classes; fall back to all classes if none exist.
        paste_cats = list(novel_ids) if novel_ids & set(crop_library) else list(all_cat_ids)

        for i in range(n_cp):
            # Apply a random geometric aug first, then paste
            aug_img, aug_boxes, aug_name = pick_and_apply(image, boxes, S)
            aug_img, aug_boxes = copy_paste_augment(
                aug_img, aug_boxes, S,
                crop_library, paste_cats,
                n_paste=random.randint(1, 3),
            )
            if not aug_boxes:
                continue

            short = _short_name(aug_name)
            fname = f"{stem}_cp{i}_{short}.jpg"
            cv2.imwrite(str(output_dir / fname), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            out_images.append({
                "id":           out_img_id,
                "file_name":    fname,
                "width":        S,
                "height":       S,
                "source_image": img_info.get("source_image", ""),
                "patch_origin": img_info.get("patch_origin", [0, 0]),
                "augmented":    True,
                "aug_type":     f"copy_paste+{aug_name}",
            })
            for (x1, y1, x2, y2, cat_id) in aug_boxes:
                out_annotations.append(
                    _box_to_coco_ann(x1, y1, x2, y2, cat_id, out_img_id, out_ann_id)
                )
                stats[cat_map[cat_id]["name"]] += 1
                out_ann_id += 1
            out_img_id += 1

    # ── Save augmented COCO JSON ──────────────────────────────────────────
    out_coco = {
        "info": {
            **coco.get("info", {}),
            "augmented":          True,
            "aug_factor":         aug_factor,
            "copy_paste_factor":  copy_paste_factor,
        },
        "categories":  coco["categories"],
        "images":      out_images,
        "annotations": out_annotations,
    }

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out_coco, f)

    print(f"\nAugmented dataset:")
    print(f"  Images      : {len(out_images)}")
    print(f"  Annotations : {len(out_annotations)}")
    print(f"  Per-class breakdown:")
    for cls, cnt in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"    {cls:<25} {cnt}")
    print(f"  Saved → {out_json_path}")

    # ── Optional merge of original + augmented ────────────────────────────
    if merge_json:
        orig_imgs  = coco["images"]
        orig_anns  = coco["annotations"]

        # Offset augmented IDs to avoid collision with originals
        orig_max_img_id = max((img["id"] for img in orig_imgs), default=0)
        orig_max_ann_id = max((ann["id"] for ann in orig_anns), default=0)

        merged_imgs = list(orig_imgs)
        merged_anns = list(orig_anns)

        for img in out_images:
            img_copy       = dict(img)
            img_copy["id"] = img["id"] + orig_max_img_id
            merged_imgs.append(img_copy)

        for ann in out_annotations:
            ann_copy              = dict(ann)
            ann_copy["id"]        = ann["id"] + orig_max_ann_id
            ann_copy["image_id"]  = ann["image_id"] + orig_max_img_id
            merged_anns.append(ann_copy)

        merged_coco = {
            "info": {**coco.get("info", {}), "merged": True},
            "categories":  coco["categories"],
            "images":      merged_imgs,
            "annotations": merged_anns,
        }

        merge_path = Path(merge_json)
        merge_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merge_path, "w") as f:
            json.dump(merged_coco, f)

        print(f"\nMerged (original + augmented):")
        print(f"  Images      : {len(merged_imgs)}")
        print(f"  Annotations : {len(merged_anns)}")
        print(f"  Saved → {merge_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline augmentation for FS-Symbol COCO patch dataset"
    )
    parser.add_argument("--patches_dir",        default="data/patches/train",
                        help="Directory of input patch images")
    parser.add_argument("--ann_json",           default="data/annotations/train.json",
                        help="Input COCO JSON produced by preprocess.py")
    parser.add_argument("--output_dir",         default="data/patches/train_aug",
                        help="Where augmented patch images are written")
    parser.add_argument("--out_json",           default="data/annotations/train_aug.json",
                        help="Output COCO JSON for the augmented set")
    parser.add_argument("--aug_factor",         type=int,   default=3,
                        help="Standard augmented copies per original patch")
    parser.add_argument("--copy_paste_factor",  type=int,   default=5,
                        help="Copy-paste copies per patch (higher for novel-class patches)")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--merge_json",         default=None,
                        help="If set, write merged original+augmented COCO JSON here")
    args = parser.parse_args()

    augment_dataset(
        patches_dir=args.patches_dir,
        ann_json=args.ann_json,
        output_dir=args.output_dir,
        out_json=args.out_json,
        aug_factor=args.aug_factor,
        copy_paste_factor=args.copy_paste_factor,
        seed=args.seed,
        merge_json=args.merge_json,
    )


if __name__ == "__main__":
    main()
