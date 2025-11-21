#!/usr/bin/env python3
"""
Export RefCOCO dataset from HuggingFace cache to local data directory.

This script downloads RefCOCO and saves it to data/refcoco/ for easy viewing
and offline access.

Usage:
    python scripts/export_refcoco.py --split val --max_samples 1000
    python scripts/export_refcoco.py --split test --max_samples 500
    python scripts/export_refcoco.py --all  # Export all splits
"""
import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.refcoco_dataset import RefCOCODataset
from src.common.paths import ProjectPaths


def export_split(
    split: str,
    output_dir: Path,
    max_samples: int = None,
    variant: str = "refcoco"
):
    """
    Export a RefCOCO split to local directory.

    Args:
        split: Dataset split ("val", "test", "testA", "testB")
        output_dir: Base output directory (e.g., data/refcoco)
        max_samples: Maximum samples to export (None = all)
        variant: RefCOCO variant
    """
    print(f"\n{'=' * 70}")
    print(f"EXPORTING {split.upper()} SPLIT")
    print(f"{'=' * 70}")

    # Load dataset
    dataset = RefCOCODataset(
        split=split,
        refcoco_split=variant,
        max_samples=max_samples
    )

    # Create split directory
    split_dir = output_dir / split
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {
        "split": split,
        "variant": variant,
        "num_samples": len(dataset),
        "samples": []
    }

    print(f"\nExporting {len(dataset)} samples to {split_dir}/")
    print("Saving images and metadata...")

    # Export each sample
    for idx in tqdm(range(len(dataset)), desc=f"Exporting {split}"):
        sample = dataset[idx]

        # Save image
        image = sample["image"]
        image_filename = f"{split}_{idx:05d}.jpg"
        image_path = images_dir / image_filename
        image.save(image_path, quality=95)

        # Add to metadata
        metadata["samples"].append({
            "id": idx,
            "image_path": f"images/{image_filename}",
            "width": sample["width"],
            "height": sample["height"],
            "phrase": sample["phrase"],
            "bbox_norm": {
                "x_min": sample["bbox_norm"][0],
                "y_min": sample["bbox_norm"][1],
                "x_max": sample["bbox_norm"][2],
                "y_max": sample["bbox_norm"][3]
            }
        })

    # Save metadata JSON
    metadata_path = split_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Exported {len(dataset)} samples")
    print(f"  Images: {images_dir}/")
    print(f"  Metadata: {metadata_path}")

    # Calculate size
    total_size = sum(f.stat().st_size for f in images_dir.glob("*.jpg"))
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

    return len(dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Export RefCOCO dataset to local directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "testA", "testB"],
        help="Dataset split to export"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to export (default: all)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="refcoco",
        choices=["refcoco", "refcoco+", "refcocog"],
        help="RefCOCO variant"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all splits (val, test)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/refcoco"),
        help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Export all main splits
        splits = ["val", "test"]
        total = 0
        for split in splits:
            count = export_split(split, args.output_dir, None, args.variant)
            total += count

        print(f"\n{'=' * 70}")
        print(f"EXPORT COMPLETE!")
        print(f"{'=' * 70}")
        print(f"Total samples exported: {total}")
        print(f"Location: {args.output_dir}/")
        print(f"\nDirectory structure:")
        print(f"  {args.output_dir}/")
        print(f"  ├── val/")
        print(f"  │   ├── images/       ({splits[0]} images)")
        print(f"  │   └── metadata.json")
        print(f"  └── test/")
        print(f"      ├── images/       ({splits[1]} images)")
        print(f"      └── metadata.json")

    elif args.split:
        export_split(args.split, args.output_dir, args.max_samples, args.variant)

        print(f"\n{'=' * 70}")
        print(f"EXPORT COMPLETE!")
        print(f"{'=' * 70}")
        print(f"\nYou can now browse the data in: {args.output_dir}/{args.split}/")
        print(f"  - View images: {args.output_dir}/{args.split}/images/")
        print(f"  - View metadata: {args.output_dir}/{args.split}/metadata.json")

    else:
        parser.error("Either --split or --all must be specified")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
