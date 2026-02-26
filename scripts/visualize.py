"""
scripts/visualize.py — Visualize patch annotations for verification.

Draws bounding boxes + class labels on random patches from the COCO dataset.
Use this after preprocessing to confirm patches and annotations look correct
(Step 2 of the verification checklist).

Usage:
    # Show 5 random training patches in a window
    python scripts/visualize.py --coco_json data/annotations/train.json \\
                                 --patches_dir data/patches/train \\
                                 --n 5

    # Save to a directory instead of displaying
    python scripts/visualize.py --coco_json data/annotations/train.json \\
                                 --patches_dir data/patches/train \\
                                 --output_dir data/visualizations \\
                                 --n 5 --split base

    # Visualize a K-shot support set
    python scripts/visualize.py --coco_json data/support_sets/k5/support.json \\
                                 --patches_dir data/support_sets/k5/images \\
                                 --n 10
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Color palette per class (BGR format for OpenCV)
CLASS_COLORS = {
    "square_structure": (80,  120, 255),   # coral
    "circle_structure": (255, 100,  50),   # blue
    "headwall":         (50,  200,  80),   # green
    "notes":            (200,  80, 200),   # purple
    "compass":          (50,  200, 200),   # yellow-green
    "rip_rap":          (0,   165, 255),   # orange  ← novel
    "site_map":         (0,   100, 255),   # red     ← novel
}
DEFAULT_COLOR = (180, 180, 180)


def draw_annotations(patch: np.ndarray, annotations: list, cat_id_to_name: dict) -> np.ndarray:
    """Draw bounding boxes with class labels on a patch."""
    vis = patch.copy()
    for ann in annotations:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x2, y2 = x + w, y + h
        cls_name = cat_id_to_name.get(ann["category_id"], f"cls_{ann['category_id']}")
        color    = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        role     = " [N]" if cls_name in ("rip_rap", "site_map") else ""

        cv2.rectangle(vis, (x, y), (x2, y2), color, 2)

        label = f"{cls_name}{role}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x, y - lh - 4), (x + lw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def visualize_patches(
    coco_json: str,
    patches_dir: str,
    n: int = 5,
    output_dir: str = None,
    split: str = "all",
    seed: int = 0,
):
    with open(coco_json) as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    patches_dir    = Path(patches_dir)

    # Index annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Optional: filter to base-only or novel-only patches
    base_cat_ids  = {c["id"] for c in data["categories"] if c.get("role") == "base"}
    novel_cat_ids = {c["id"] for c in data["categories"] if c.get("role") == "novel"}

    def has_split(img_id):
        if split == "all":
            return True
        img_anns = anns_by_img[img_id]
        cats = {a["category_id"] for a in img_anns}
        if split == "base":
            return bool(cats & base_cat_ids)
        if split == "novel":
            return bool(cats & novel_cat_ids)
        return True

    eligible = [img for img in data["images"] if has_split(img["id"])]

    rng = random.Random(seed)
    selected = rng.sample(eligible, min(n, len(eligible)))

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(selected)} patches from {coco_json}")
    print(f"Legend: [N] = novel class | boxes colored per class\n")

    for img_info in selected:
        patch_path = patches_dir / img_info["file_name"]
        if not patch_path.exists():
            print(f"  Patch not found: {patch_path}")
            continue

        patch = cv2.imread(str(patch_path))
        if patch is None:
            print(f"  Could not read: {patch_path}")
            continue

        annotations = anns_by_img[img_info["id"]]
        vis = draw_annotations(patch, annotations, cat_id_to_name)

        # Add info overlay
        source = img_info.get("source_image", "")
        origin = img_info.get("patch_origin", [0, 0])
        info   = f"{Path(source).stem} | origin=({origin[0]},{origin[1]}) | anns={len(annotations)}"
        cv2.putText(vis, info, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, info, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Print annotation summary to terminal
        cls_counts = defaultdict(int)
        for ann in annotations:
            cls_counts[cat_id_to_name.get(ann["category_id"], "?")] += 1
        print(f"  {img_info['file_name']}")
        for cls, cnt in cls_counts.items():
            role = "[NOVEL]" if cls in ("rip_rap", "site_map") else "[base] "
            print(f"    {role} {cls}: {cnt}")

        if output_dir:
            out_file = Path(output_dir) / f"vis_{img_info['file_name']}"
            cv2.imwrite(str(out_file), vis)
            print(f"    Saved → {out_file}")
        else:
            cv2.imshow(img_info["file_name"], vis)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    if output_dir:
        print(f"\nAll visualizations saved to: {output_dir}")
    print("Done.")


def print_dataset_stats(coco_json: str):
    """Print a per-class annotation summary for a COCO dataset."""
    with open(coco_json) as f:
        data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_id_to_role = {c["id"]: c.get("role", "?") for c in data["categories"]}
    counts = defaultdict(int)
    for ann in data["annotations"]:
        counts[ann["category_id"]] += 1

    print(f"\nDataset: {coco_json}")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Per-class:")
    for cat_id, name in sorted(cat_id_to_name.items()):
        role = cat_id_to_role[cat_id]
        print(f"    [{role:<5}] id={cat_id} {name:<25} {counts[cat_id]:>4}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize COCO patch dataset annotations"
    )
    parser.add_argument("--coco_json",   default="data/annotations/train.json")
    parser.add_argument("--patches_dir", default="data/patches/train")
    parser.add_argument("--n",           type=int, default=5, help="Number of patches to show")
    parser.add_argument("--output_dir",  default=None, help="Save images here instead of displaying")
    parser.add_argument("--split",       default="all", choices=["all", "base", "novel"])
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--stats",       action="store_true", help="Print dataset stats and exit")
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.coco_json)
        return

    visualize_patches(
        coco_json=args.coco_json,
        patches_dir=args.patches_dir,
        n=args.n,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
