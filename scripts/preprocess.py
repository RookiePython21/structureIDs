"""
scripts/preprocess.py — CVAT XML → COCO-format patch dataset.

Implements the data preprocessing pipeline from FS-Symbol:
  1. Parse CVAT XML annotations.
  2. Mask out title_exclusion (and optionally legend) regions with white fill.
  3. Convert all shape types to axis-aligned bounding boxes.
  4. Tile each image into 640×640 patches with 320px overlap.
  5. Assign annotations to patches (center-point criterion, per paper).
  6. Save annotated patches as JPEG + COCO JSON.

Outputs:
    data/
    ├── patches/
    │   ├── train/   ← annotated patch images
    │   └── test/
    └── annotations/
        ├── train.json        ← full COCO (base + novel)
        ├── train_base.json   ← base-class only (Stage 1 input)
        └── test.json

Usage:
    python scripts/preprocess.py \\
        --annotations annotations/annotations.xml \\
        --images_dir   annotations/images \\
        --output_dir   data \\
        --patch_size   640 \\
        --overlap      320 \\
        --seed         42 \\
        --train_ratio  0.8
"""

import os
import re
import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from lxml import etree

# ---------------------------------------------------------------------------
# Class configuration (must match model/fs_symbol.py and COCO category IDs)
# ---------------------------------------------------------------------------
EXCLUSION_CLASSES = {"title_exclusion", "legend", "compass", "site_map", "notes"}

BASE_CLASSES  = ["square_structure", "circle_structure", "headwall"]
NOVEL_CLASSES = ["rip_rap"]
ALL_DETECTION_CLASSES = BASE_CLASSES + NOVEL_CLASSES

# 1-indexed COCO category IDs
CATEGORY_MAP = {cls: i + 1 for i, cls in enumerate(ALL_DETECTION_CLASSES)}


# ---------------------------------------------------------------------------
# CVAT XML parsing
# ---------------------------------------------------------------------------
def _box_from_rotated(xtl, ytl, xbr, ybr, rotation_deg):
    """Convert a rotated box to an axis-aligned bounding box."""
    cx = (xtl + xbr) / 2
    cy = (ytl + ybr) / 2
    w  = xbr - xtl
    h  = ybr - ytl
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    rotated = [(cx + cos_a * px - sin_a * py, cy + sin_a * px + cos_a * py) for px, py in half_pts]
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_cvat_xml(xml_path: str):
    """
    Parse CVAT XML annotations.

    Returns:
        List of dicts:
            {
              "id": int,
              "name": str,
              "width": int,
              "height": int,
              "annotations": [{"label": str, "bbox_xyxy": [x1,y1,x2,y2]}, ...]
            }
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    images = []
    for img_elem in root.findall("image"):
        img_info = {
            "id":     int(img_elem.get("id")),
            "name":   img_elem.get("name"),
            "width":  int(img_elem.get("width")),
            "height": int(img_elem.get("height")),
            "annotations": [],
        }

        for ann_elem in img_elem:
            label = ann_elem.get("label")
            if label is None:
                continue

            bbox = None

            if ann_elem.tag == "box":
                xtl = float(ann_elem.get("xtl"))
                ytl = float(ann_elem.get("ytl"))
                xbr = float(ann_elem.get("xbr"))
                ybr = float(ann_elem.get("ybr"))
                rotation = float(ann_elem.get("rotation", 0.0))
                if abs(rotation) > 1.0:
                    bbox = _box_from_rotated(xtl, ytl, xbr, ybr, rotation)
                else:
                    bbox = [xtl, ytl, xbr, ybr]

            elif ann_elem.tag == "ellipse":
                cx = float(ann_elem.get("cx"))
                cy = float(ann_elem.get("cy"))
                rx = float(ann_elem.get("rx"))
                ry = float(ann_elem.get("ry"))
                bbox = [cx - rx, cy - ry, cx + rx, cy + ry]

            elif ann_elem.tag == "polygon":
                pts_str = ann_elem.get("points", "")
                pts = [
                    tuple(float(v) for v in pt.split(","))
                    for pt in pts_str.split(";")
                    if pt.strip()
                ]
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]

            if bbox is not None:
                img_info["annotations"].append({"label": label, "bbox_xyxy": bbox})

        images.append(img_info)

    return images


# ---------------------------------------------------------------------------
# Masking & patching
# ---------------------------------------------------------------------------
def apply_exclusion_masks(image: np.ndarray, annotations: list, img_w: int, img_h: int) -> np.ndarray:
    """
    White-fill all exclusion regions (title blocks, legends) on the image.
    Mirrors the border/title-block removal step in the paper.
    """
    masked = image.copy()
    for ann in annotations:
        if ann["label"] in EXCLUSION_CLASSES:
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            x1 = max(0, int(math.floor(x1)))
            y1 = max(0, int(math.floor(y1)))
            x2 = min(img_w, int(math.ceil(x2)))
            y2 = min(img_h, int(math.ceil(y2)))
            masked[y1:y2, x1:x2] = 255
    return masked


def generate_patch_coords(img_h: int, img_w: int, patch_size: int, overlap: int):
    """
    Generate (x1, y1, x2, y2) patch coordinates covering the entire image.
    Last patch in each row/column may be smaller (padded when extracted).
    """
    stride = patch_size - overlap
    coords = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)
            coords.append((x, y, x2, y2))
            if x2 == img_w:
                break
            x += stride
        if y2 == img_h:
            break
        y += stride
    return coords


def extract_patch(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, patch_size: int) -> np.ndarray:
    """Extract patch; pad with white if smaller than patch_size."""
    crop = image[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == patch_size and w == patch_size:
        return crop
    padded = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    padded[:h, :w] = crop
    return padded


def assign_annotation_to_patch(ann_bbox, px1, py1, px2, py2):
    """
    Assign annotation to patch using center-point criterion (per paper).
    Returns bbox in patch-local coordinates [x1,y1,x2,y2], or None.
    Annotations whose center falls outside the patch are discarded.
    """
    ax1, ay1, ax2, ay2 = ann_bbox
    ann_cx = (ax1 + ax2) / 2
    ann_cy = (ay1 + ay2) / 2

    if not (px1 <= ann_cx < px2 and py1 <= ann_cy < py2):
        return None

    # Clip to patch boundary and convert to local coordinates
    lx1 = max(ax1, px1) - px1
    ly1 = max(ay1, py1) - py1
    lx2 = min(ax2, px2) - px1
    ly2 = min(ay2, py2) - py1

    if lx2 <= lx1 or ly2 <= ly1:
        return None

    return [lx1, ly1, lx2, ly2]


# ---------------------------------------------------------------------------
# Filename normalisation (handles spaces / special chars from CVAT export)
# ---------------------------------------------------------------------------
def _normalize(name: str) -> str:
    """
    Normalize a filename for fuzzy matching.
    'C-101 (1)_001.png' and 'C-101_1_001.png' both become 'c-101_1_001.png'.
    """
    stem, ext = os.path.splitext(name)
    stem = stem.replace(" ", "_")           # spaces → underscores
    stem = re.sub(r"[^\w\-]", "_", stem)    # special chars → underscores
    stem = re.sub(r"_+", "_", stem)         # collapse runs of underscores
    stem = stem.strip("_")
    return (stem + ext).lower()


def build_filename_lookup(images_dir: Path) -> dict:
    """
    Build a dict mapping normalized name → actual Path for every file in images_dir.
    Lets us match XML filenames (with spaces) to Drive filenames (with underscores).
    """
    lookup = {}
    for p in images_dir.iterdir():
        if p.is_file():
            lookup[_normalize(p.name)] = p
    return lookup


def resolve_image_path(xml_name: str, images_dir: Path, lookup: dict):
    """
    Return the actual Path for xml_name, or None if not found.
    Tries exact match first, then normalized match.
    """
    exact = images_dir / xml_name
    if exact.exists():
        return exact
    return lookup.get(_normalize(xml_name))


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def process_dataset(
    annotations_xml: str,
    images_dir: str,
    output_dir: str,
    patch_size: int = 640,
    overlap: int = 320,
    seed: int = 42,
    train_ratio: float = 0.8,
):
    random.seed(seed)
    np.random.seed(seed)

    output_dir  = Path(output_dir)
    images_dir  = Path(images_dir)
    patches_dir = output_dir / "patches"
    ann_out_dir = output_dir / "annotations"
    patches_dir.mkdir(parents=True, exist_ok=True)
    ann_out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Parse XML --------------------------------------------------------
    # Build normalized filename lookup once (handles spaces / special chars)
    filename_lookup = build_filename_lookup(images_dir)
    print(f"Found {len(filename_lookup)} image files in {images_dir}")

    print("Parsing CVAT XML...")
    images = parse_cvat_xml(annotations_xml)
    print(f"  Found {len(images)} images")

    class_counts = defaultdict(int)
    for img in images:
        for ann in img["annotations"]:
            class_counts[ann["label"]] += 1

    print("\nAnnotation counts per class:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        role = "BASE" if cls in BASE_CLASSES else ("NOVEL" if cls in NOVEL_CLASSES else "EXCL")
        print(f"  [{role:<5}] {cls:<25} {cnt}")

    # ---- Train/test split -------------------------------------------------
    random.shuffle(images)
    n_train     = int(len(images) * train_ratio)
    train_imgs  = images[:n_train]
    test_imgs   = images[n_train:]
    print(f"\nSplit: {len(train_imgs)} train / {len(test_imgs)} test images")

    # ---- Process each split -----------------------------------------------
    coco_data_by_split = {}

    for split_name, split_images in [("train", train_imgs), ("test", test_imgs)]:
        split_patches_dir = patches_dir / split_name
        split_patches_dir.mkdir(parents=True, exist_ok=True)

        coco_images      = []
        coco_annotations = []
        ann_id           = 1
        img_id           = 1
        patch_stats      = defaultdict(int)
        skipped_images   = 0

        print(f"\nProcessing '{split_name}' split...")

        for img_info in split_images:
            img_path = resolve_image_path(img_info["name"], images_dir, filename_lookup)
            if img_path is None:
                print(f"  WARNING: Not found — {img_info['name']}")
                skipped_images += 1
                continue

            print(f"  → {img_info['name']}", flush=True)
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  WARNING: Could not read image — {img_path}")
                skipped_images += 1
                continue

            img_h, img_w = image.shape[:2]

            # Apply exclusion masks (title blocks, legends)
            image = apply_exclusion_masks(image, img_info["annotations"], img_w, img_h)

            # Filter to detectable annotations only
            detect_anns = [
                ann for ann in img_info["annotations"]
                if ann["label"] in CATEGORY_MAP
            ]

            # Generate patch grid
            patch_coords = generate_patch_coords(img_h, img_w, patch_size, overlap)
            patches_saved = 0

            for px1, py1, px2, py2 in patch_coords:
                # Collect annotations whose center falls in this patch
                patch_anns = []
                for ann in detect_anns:
                    local_bbox = assign_annotation_to_patch(
                        ann["bbox_xyxy"], px1, py1, px2, py2
                    )
                    if local_bbox is not None:
                        patch_anns.append({"label": ann["label"], "bbox_xyxy": local_bbox})

                # Skip empty patches (no annotations)
                if not patch_anns:
                    continue

                # Save patch image
                patch = extract_patch(image, px1, py1, px2, py2, patch_size)
                stem = Path(img_info["name"]).stem.replace(" ", "_")
                patch_filename = f"{stem}_{px1}_{py1}.jpg"
                patch_path = split_patches_dir / patch_filename
                cv2.imwrite(str(patch_path), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Build COCO records
                coco_images.append({
                    "id":           img_id,
                    "file_name":    patch_filename,
                    "width":        patch_size,
                    "height":       patch_size,
                    "source_image": img_info["name"],
                    "patch_origin": [px1, py1],
                })

                for ann in patch_anns:
                    x1, y1, x2, y2 = ann["bbox_xyxy"]
                    w, h = x2 - x1, y2 - y1
                    coco_annotations.append({
                        "id":          ann_id,
                        "image_id":    img_id,
                        "category_id": CATEGORY_MAP[ann["label"]],
                        "bbox":        [x1, y1, w, h],  # COCO: [x, y, w, h]
                        "area":        w * h,
                        "iscrowd":     0,
                    })
                    patch_stats[ann["label"]] += 1
                    ann_id += 1

                img_id   += 1
                patches_saved += 1

        # Build and save COCO JSON
        categories = [
            {
                "id":            cat_id,
                "name":          cls_name,
                "supercategory": "structure",
                "role":          "base" if cls_name in BASE_CLASSES else "novel",
            }
            for cls_name, cat_id in CATEGORY_MAP.items()
        ]

        coco_json = {
            "info": {
                "description": "Civil Engineering FS-Symbol Dataset",
                "version":     "1.0",
                "split":       split_name,
                "patch_size":  patch_size,
                "overlap":     overlap,
            },
            "categories":  categories,
            "images":      coco_images,
            "annotations": coco_annotations,
        }

        coco_path = ann_out_dir / f"{split_name}.json"
        with open(coco_path, "w") as f:
            json.dump(coco_json, f)

        coco_data_by_split[split_name] = coco_json

        print(f"\n  [{split_name}] Results:")
        print(f"    Annotated patches : {len(coco_images)}")
        print(f"    Total annotations : {len(coco_annotations)}")
        if skipped_images:
            print(f"    Skipped images    : {skipped_images}")
        print(f"    Per-class breakdown:")
        for cls in ALL_DETECTION_CLASSES:
            cnt = patch_stats.get(cls, 0)
            role = "BASE" if cls in BASE_CLASSES else "NOVEL"
            print(f"      [{role:<5}] {cls:<25} {cnt}")
        print(f"    Saved → {coco_path}")

    # ---- Base-only training set (Stage 1 input) ---------------------------
    train_data     = coco_data_by_split.get("train", {})
    base_cat_ids   = {CATEGORY_MAP[c] for c in BASE_CLASSES if c in CATEGORY_MAP}
    base_img_ids   = {a["image_id"] for a in train_data.get("annotations", []) if a["category_id"] in base_cat_ids}
    base_images    = [img for img in train_data.get("images", [])      if img["id"] in base_img_ids]
    base_anns      = [ann for ann in train_data.get("annotations", []) if ann["category_id"] in base_cat_ids]
    base_cats      = [c   for c   in train_data.get("categories", [])  if c["id"] in base_cat_ids]

    base_coco = {
        "info":        {**train_data.get("info", {}), "split": "train_base"},
        "categories":  base_cats,
        "images":      base_images,
        "annotations": base_anns,
    }

    base_path = ann_out_dir / "train_base.json"
    with open(base_path, "w") as f:
        json.dump(base_coco, f)

    print(f"\nBase training JSON → {base_path}")
    print(f"  Patches: {len(base_images)}  |  Annotations: {len(base_anns)}")

    # ---- Dataset config ---------------------------------------------------
    config = {
        "patch_size":       patch_size,
        "overlap":          overlap,
        "seed":             seed,
        "train_ratio":      train_ratio,
        "base_classes":     BASE_CLASSES,
        "novel_classes":    NOVEL_CLASSES,
        "exclusion_classes": list(EXCLUSION_CLASSES),
        "category_map":     CATEGORY_MAP,
        "splits": {
            "train_images": len(train_imgs),
            "test_images":  len(test_imgs),
        },
    }
    config_path = output_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDataset config → {config_path}")
    print("\nPreprocessing complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CVAT XML annotations into COCO patch dataset (FS-Symbol)"
    )
    parser.add_argument("--annotations",  default="annotations/annotations.xml")
    parser.add_argument("--images_dir",   default="annotations/images")
    parser.add_argument("--output_dir",   default="data")
    parser.add_argument("--patch_size",   type=int,   default=640)
    parser.add_argument("--overlap",      type=int,   default=320)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--train_ratio",  type=float, default=0.8)
    args = parser.parse_args()

    process_dataset(
        annotations_xml=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap=args.overlap,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
