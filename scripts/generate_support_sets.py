"""
scripts/generate_support_sets.py — Generate K-shot support sets for Stage 2.

For each K in {1, 2, 3, 5, 9}:
  - Samples K instances per class from the training patch COCO dataset.
  - Base and novel classes are both included (balanced support set per paper).
  - Copies the relevant patch images into a support set directory.
  - Saves a COCO JSON for each K.

The random seed is fixed (seed + K per experiment) to ensure reproducibility
across runs — matching the paper's fixed-K experimental protocol.

Usage:
    python scripts/generate_support_sets.py \\
        --train_json  data/annotations/train.json \\
        --patches_dir data/patches/train \\
        --output_dir  data/support_sets \\
        --k_shots     1 2 3 5 9 \\
        --seed        42

Output structure:
    data/support_sets/
    ├── k1/
    │   ├── images/       ← copied patch images
    │   ├── support.json  ← COCO JSON for this K
    │   └── summary.json  ← human-readable per-class instance list
    ├── k2/  ...
    └── k9/
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

BASE_CLASSES  = ["square_structure", "circle_structure", "headwall"]
NOVEL_CLASSES = ["rip_rap"]
ALL_CLASSES   = BASE_CLASSES + NOVEL_CLASSES


def load_coco(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_instance_index(coco_data: dict) -> dict:
    """
    Build a per-class index of (img_info, annotation) tuples.

    Returns:
        {class_name: [(img_info_dict, ann_dict), ...], ...}
    """
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    img_id_to_info = {img["id"]: img for img in coco_data["images"]}

    instances = defaultdict(list)
    for ann in coco_data["annotations"]:
        cls_name = cat_id_to_name.get(ann["category_id"])
        if cls_name and cls_name in ALL_CLASSES:
            img_info = img_id_to_info.get(ann["image_id"])
            if img_info:
                instances[cls_name].append((img_info, ann))

    return instances


def sample_k_shots(instances: dict, k: int, rng: random.Random) -> dict:
    """
    Sample exactly K instances per class.
    If a class has fewer than K instances, uses all available (with warning).
    """
    sampled = {}
    for cls_name in ALL_CLASSES:
        inst_list = instances.get(cls_name, [])
        if not inst_list:
            print(f"  WARNING: No instances for class '{cls_name}' in training set.")
            sampled[cls_name] = []
        elif len(inst_list) < k:
            print(
                f"  WARNING: '{cls_name}' has only {len(inst_list)} instances "
                f"(requested K={k}). Using all."
            )
            sampled[cls_name] = list(inst_list)
        else:
            sampled[cls_name] = rng.sample(inst_list, k)
    return sampled


def build_support_coco(sampled: dict, all_categories: list) -> dict:
    """Build a COCO JSON dict from sampled (img_info, annotation) pairs."""
    images_seen = {}
    annotations = []
    ann_id = 1

    for cls_name in ALL_CLASSES:
        for img_info, ann in sampled.get(cls_name, []):
            img_id = img_info["id"]
            if img_id not in images_seen:
                images_seen[img_id] = img_info
            annotations.append({**ann, "id": ann_id})
            ann_id += 1

    return {
        "info":        {"description": "FS-Symbol K-Shot Support Set"},
        "categories":  all_categories,
        "images":      list(images_seen.values()),
        "annotations": annotations,
    }


def generate_support_sets(
    train_json: str,
    patches_dir: str,
    output_dir: str,
    k_shots: tuple = (1, 2, 3, 5, 9),
    seed: int = 42,
):
    patches_dir = Path(patches_dir)
    output_dir  = Path(output_dir)

    coco_data = load_coco(train_json)
    instances = build_instance_index(coco_data)

    print("Instance counts in training set:")
    for cls in ALL_CLASSES:
        n    = len(instances.get(cls, []))
        role = "BASE" if cls in BASE_CLASSES else "NOVEL"
        print(f"  [{role:<5}] {cls:<25} {n:>4}")

    for k in k_shots:
        print(f"\nGenerating K={k} support set...")
        k_dir        = output_dir / f"k{k}"
        k_images_dir = k_dir / "images"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_images_dir.mkdir(exist_ok=True)

        # Fixed seed per K (reproducible across runs; seed varies per K
        # so that K=1 is a strict subset of K=2, etc.)
        rng     = random.Random(seed + k)
        sampled = sample_k_shots(instances, k, rng)

        # Build COCO JSON
        support_coco = build_support_coco(sampled, coco_data["categories"])

        # Copy patch images
        copied = set()
        missing = []
        for img_info in support_coco["images"]:
            fname = img_info["file_name"]
            if fname in copied:
                continue
            src = patches_dir / fname
            dst = k_images_dir / fname
            if src.exists():
                shutil.copy2(src, dst)
                copied.add(fname)
            else:
                missing.append(fname)

        if missing:
            print(f"  WARNING: {len(missing)} patch images not found in {patches_dir}")

        # Save COCO JSON
        coco_path = k_dir / "support.json"
        with open(coco_path, "w") as f:
            json.dump(support_coco, f)

        # Save human-readable summary
        summary = {
            "k":    k,
            "seed": seed + k,
            "total_instances": sum(len(v) for v in sampled.values()),
            "per_class": {
                cls: [
                    {
                        "patch":   img["file_name"],
                        "bbox_xywh": ann["bbox"],
                        "category_id": ann["category_id"],
                    }
                    for img, ann in sampled.get(cls, [])
                ]
                for cls in ALL_CLASSES
            },
        }
        with open(k_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        n_patches = len(support_coco["images"])
        n_anns    = len(support_coco["annotations"])
        print(f"  K={k}: {n_patches} unique patches, {n_anns} total annotations")
        print(f"  Per-class samples:")
        for cls in ALL_CLASSES:
            n    = len(sampled.get(cls, []))
            role = "BASE" if cls in BASE_CLASSES else "NOVEL"
            print(f"    [{role:<5}] {cls:<25} {n}")
        print(f"  → {coco_path}")

    print(f"\nAll support sets saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate K-shot support sets from COCO training dataset"
    )
    parser.add_argument("--train_json",  default="data/annotations/train.json")
    parser.add_argument("--patches_dir", default="data/patches/train")
    parser.add_argument("--output_dir",  default="data/support_sets")
    parser.add_argument("--k_shots",     nargs="+", type=int, default=[1, 2, 3, 5, 9])
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    generate_support_sets(
        train_json=args.train_json,
        patches_dir=args.patches_dir,
        output_dir=args.output_dir,
        k_shots=args.k_shots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
