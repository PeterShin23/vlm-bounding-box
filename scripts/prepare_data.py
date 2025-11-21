#!/usr/bin/env python3
"""
RefCOCO data preparation and verification script.

This script:
1. Loads RefCOCO dataset from HuggingFace
2. Shows dataset statistics
3. Creates visualizations of sample (image, phrase, bbox) tuples
4. Verifies data integrity

Usage:
    python scripts/prepare_data.py --visualize
    python scripts/prepare_data.py --split train --num_samples 10
"""
import argparse
from pathlib import Path
import sys
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.refcoco_dataset import RefCOCODataset
from src.common.paths import ProjectPaths
from src.common.viz import draw_bbox_on_image, create_grid
from src.data.box_utils import denormalize_bbox
from PIL import Image, ImageDraw, ImageFont


def print_dataset_info(dataset_name: str = "lmms-lab/RefCOCO", variant: str = "refcoco"):
    """Print information about the RefCOCO dataset."""
    print("\n" + "=" * 70)
    print("REFCOCO DATASET INFORMATION")
    print("=" * 70)
    print(f"\nDataset: {dataset_name}")
    print(f"Variant: {variant}")
    print("\nRefCOCO is a referring expression dataset where:")
    print("  - Images come from MS COCO")
    print("  - Each sample has a referring expression (phrase)")
    print("  - Each phrase refers to one specific object/region")
    print("  - Task: predict bounding box for the described region")
    print("\nIMPORTANT - Available splits:")
    print("  - val: ~8,800 samples (use as train split)")
    print("  - test: ~5,000 samples (final evaluation)")
    print("  - testA: ~2,000 samples (people subset)")
    print("  - testB: ~1,800 samples (objects subset)")
    print("\n  NOTE: No dedicated 'train' split - use 'val' for training!")
    print("\nVariants:")
    print("  - refcoco: Original dataset")
    print("  - refcoco+: No location words allowed in expressions")
    print("  - refcocog: More complex, longer expressions")
    print("=" * 70)


def load_and_inspect_dataset(
    split: str = "train",
    variant: str = "refcoco",
    num_samples: int = 5
):
    """
    Load dataset and inspect samples.

    Args:
        split: Dataset split to load
        variant: RefCOCO variant
        num_samples: Number of samples to inspect
    """
    print(f"\n{'=' * 70}")
    print(f"Loading {split} split...")
    print(f"{'=' * 70}")

    # Load dataset
    dataset = RefCOCODataset(
        split=split,
        refcoco_split=variant,
        max_samples=None  # Load full dataset for inspection
    )

    print(f"\n✓ Successfully loaded {len(dataset)} samples")

    # Inspect first few samples
    print(f"\nInspecting first {num_samples} samples:")
    print("-" * 70)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i + 1}:")
        print(f"  Image size: {sample['width']} x {sample['height']}")
        print(f"  Phrase: \"{sample['phrase']}\"")
        print(f"  BBox (normalized): {sample['bbox_norm']}")
        print(f"  BBox (JSON): {sample['bbox_json']}")

    return dataset


def visualize_samples(
    dataset: RefCOCODataset,
    num_samples: int = 9,
    output_path: Path = None
):
    """
    Create visualization grid of samples with bounding boxes and phrases.

    Args:
        dataset: RefCOCO dataset
        num_samples: Number of samples to visualize
        output_path: Path to save visualization
    """
    print(f"\nCreating visualization of {num_samples} samples...")

    # Select random samples
    random.seed(42)  # For reproducibility
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Create visualizations
    vis_images = []
    for idx in indices:
        sample = dataset[idx]

        # Get image
        image = sample["image"]

        # Resize for visualization (keep aspect ratio)
        max_size = 400
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Draw bbox
        vis = draw_bbox_on_image(
            image,
            sample["bbox_norm"],
            color="green",
            width=3,
            label=None  # We'll add phrase as text below
        )

        # Add phrase text at bottom
        draw = ImageDraw.Draw(vis)

        # Wrap phrase if too long
        phrase = sample["phrase"]
        if len(phrase) > 50:
            phrase = phrase[:47] + "..."

        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()

        # Draw phrase at bottom with background
        text_y = vis.height - 25
        text_bbox = draw.textbbox((5, text_y), phrase, font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text((5, text_y), phrase, fill="white", font=font)

        vis_images.append(vis)

    # Create grid
    grid = create_grid(vis_images, grid_width=3, padding=10)

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path)
        print(f"✓ Visualization saved to {output_path}")

    return grid


def test_data_loading(variant: str = "refcoco"):
    """Test that all splits can be loaded."""
    print(f"\n{'=' * 70}")
    print("TESTING DATA LOADING")
    print(f"{'=' * 70}")

    splits = ["val", "test", "testA", "testB"]

    for split in splits:
        try:
            print(f"\nTesting {split} split...")
            dataset = RefCOCODataset(
                split=split,
                refcoco_split=variant,
                max_samples=10  # Just test with 10 samples
            )
            sample = dataset[0]

            # Verify all required fields
            required_fields = ["image", "width", "height", "phrase", "bbox_norm", "bbox_json"]
            for field in required_fields:
                assert field in sample, f"Missing field: {field}"

            print(f"  ✓ {split}: {len(dataset)} samples loaded successfully")

        except Exception as e:
            print(f"  ✗ {split}: Failed to load - {e}")


def print_budget_recommendations():
    """Print recommendations for budget-aware training."""
    print("\n" + "=" * 70)
    print("BUDGET-AWARE TRAINING RECOMMENDATIONS")
    print("=" * 70)
    print("\nFor your $10 Runpod budget, recommended configurations:")
    print("\nIMPORTANT: Use 'val' split for training (8,800 samples)")
    print("\n1. LOCAL DEBUG (MPS - FREE):")
    print("   - split: val")
    print("   - max_samples: 500-1000")
    print("   - Purpose: Quick iteration, debug code")
    print("   - Time: ~10-20 minutes")
    print("\n2. RUNPOD CONSERVATIVE (~$3-5):")
    print("   - GPU: L4 or A10G (~$0.30-0.50/hr)")
    print("   - split: val")
    print("   - max_samples: 5000")
    print("   - num_epochs: 2")
    print("   - Time: ~3-6 hours")
    print("\n3. RUNPOD MODERATE (~$6-9):")
    print("   - GPU: RTX 3090 (~$0.50-0.70/hr)")
    print("   - split: val (all 8,800 samples)")
    print("   - num_epochs: 2-3")
    print("   - max_steps: 5000 (early stopping)")
    print("   - Time: ~8-12 hours")
    print("\nUse 'test' split for final evaluation!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and inspect RefCOCO dataset"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="refcoco",
        choices=["refcoco", "refcoco+", "refcocog"],
        help="RefCOCO variant to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "testA", "testB"],
        help="Dataset split to inspect (use 'val' for training data)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to inspect"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of samples"
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=9,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test loading all splits"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information"
    )

    args = parser.parse_args()

    # Setup paths
    paths = ProjectPaths()
    paths.create_directories()

    # Show info if requested
    if args.info:
        print_dataset_info(variant=args.variant)
        return

    # Test loading if requested
    if args.test:
        test_data_loading(variant=args.variant)
        print_budget_recommendations()
        return

    # Load and inspect dataset
    dataset = load_and_inspect_dataset(
        split=args.split,
        variant=args.variant,
        num_samples=args.num_samples
    )

    # Create visualizations if requested
    if args.visualize:
        vis_dir = Path("outputs/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)

        output_path = vis_dir / f"refcoco_{args.variant}_{args.split}_samples.png"

        visualize_samples(
            dataset,
            num_samples=args.num_vis,
            output_path=output_path
        )

    # Print budget recommendations
    print_budget_recommendations()

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review visualizations in outputs/visualizations/")
    print("  2. For local debug training:")
    print("     python scripts/train.py --debug --max_train_samples 500")
    print("  3. For Runpod training:")
    print("     See MIGRATION_REFCOCO.md for setup instructions")
    print("=" * 70)


if __name__ == "__main__":
    main()
