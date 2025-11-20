#!/usr/bin/env python3
"""
Data preparation script for saliency dataset.

This script:
1. Downloads/locates the saliency dataset (DUTS recommended)
2. Processes images and masks to compute bounding boxes
3. Creates train/val/test splits
4. Saves splits to JSON files for training

Usage:
    python scripts/prepare_data.py --data_dir data/raw/duts-tr --output_dir data/processed

For download instructions:
    python scripts/prepare_data.py --download_help
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.paths import ProjectPaths
from src.data.saliency_dataset import RawSaliencyDataset
from src.data.split_utils import create_splits, save_all_splits, DataSample
from src.common.viz import draw_bbox_on_image, create_grid, mask_to_rgb
from PIL import Image
import numpy as np
from tqdm import tqdm


def download_instructions():
    """Print instructions for downloading the dataset."""
    print("\n" + "=" * 70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("\n** RECOMMENDED: DUTS Dataset **")
    print("\nThe DUTS dataset is the preferred choice for this project:")
    print("  - 10,553 high-quality training images")
    print("  - Pixel-level saliency annotations")
    print("  - Challenging scenarios, actively maintained")
    print("  - Well-established benchmark (CVPR 2017)")
    print("\n1. Download DUTS-TR (training set):")
    print("   Website: http://saliencydetection.net/duts/")
    print("   Direct: http://saliencydetection.net/duts/download/DUTS-TR.zip (~800MB)")
    print("   Save to: data/downloads/DUTS-TR.zip")
    print("\n2. Extract the zip file:")
    print("   unzip data/downloads/DUTS-TR.zip -d /tmp/")
    print("\n3. Organize the files:")
    print("   mkdir -p data/raw/duts-tr")
    print("   mv /tmp/DUTS-TR/DUTS-TR-Image data/raw/duts-tr/images")
    print("   mv /tmp/DUTS-TR/DUTS-TR-Mask data/raw/duts-tr/masks")
    print("   rm -rf /tmp/DUTS-TR")
    print("\n4. Run data preparation:")
    print("   python scripts/prepare_data.py \\")
    print("       --data_dir data/raw/duts-tr \\")
    print("       --train_size 2000 \\")
    print("       --val_size 500 \\")
    print("       --test_size 500 \\")
    print("       --visualize")
    print("\nCITATION:")
    print("  Wang et al. 'Learning to Detect Salient Objects with")
    print("  Image-level Supervision', CVPR 2017")
    print("  See CITATIONS.md for full BibTeX and copyright info")
    print("\n" + "-" * 70)
    print("\nAlternative Datasets:")
    print("\nOption 2: MSRA-B Dataset (~2,500 images)")
    print("  Note: Some official download links may be broken")
    print("  Search for 'MSRA-B saliency dataset' for mirrors")
    print("\nOption 3: DUT-OMRON Dataset (~5,000 images)")
    print("  Download: http://saliencydetection.net/dut-omron/")
    print("\n" + "-" * 70)
    print("\nExpected directory structure after setup:")
    print("  data/raw/duts-tr/")
    print("    images/")
    print("      ILSVRC2012_test_00000003.jpg")
    print("      ILSVRC2012_test_00000004.jpg")
    print("      ...")
    print("    masks/")
    print("      ILSVRC2012_test_00000003.png")
    print("      ILSVRC2012_test_00000004.png")
    print("      ...")
    print("=" * 70)


def visualize_samples(
    samples: list[DataSample],
    num_samples: int = 9,
    output_path: Path = None
):
    """
    Visualize a few samples with their bounding boxes.

    Args:
        samples: List of DataSample objects
        num_samples: Number of samples to visualize
        output_path: Path to save visualization
    """
    print(f"\nCreating visualization of {num_samples} samples...")

    # Select random samples
    import random
    selected = random.sample(samples, min(num_samples, len(samples)))

    # Create visualizations
    vis_images = []
    for sample in selected[:num_samples]:
        # Load image
        image = Image.open(sample.image_path).convert("RGB")

        # Resize for visualization (keep aspect ratio)
        max_size = 300
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Draw bbox
        vis = draw_bbox_on_image(
            image,
            sample.bbox_norm,
            color="green",
            width=2,
            label=f"Main Subject"
        )

        vis_images.append(vis)

    # Create grid
    grid = create_grid(vis_images, grid_width=3, padding=5)

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path)
        print(f"Visualization saved to {output_path}")

    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Prepare saliency dataset for training"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to dataset directory containing 'images/' and 'masks/' folders"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="Path to images directory (alternative to --data_dir)"
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        help="Path to masks directory (alternative to --data_dir)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed splits"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=2000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=500,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Number of test samples"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum total samples to process (for debugging)"
    )
    parser.add_argument(
        "--image_suffix",
        type=str,
        default=".jpg",
        help="Image file suffix"
    )
    parser.add_argument(
        "--mask_suffix",
        type=str,
        default=".png",
        help="Mask file suffix"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of sample data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    parser.add_argument(
        "--download_help",
        action="store_true",
        help="Show dataset download instructions"
    )

    args = parser.parse_args()

    # Show download instructions if requested
    if args.download_help:
        download_instructions()
        return

    # Determine image and mask directories
    if args.data_dir:
        image_dir = args.data_dir / "images"
        mask_dir = args.data_dir / "masks"
    elif args.image_dir and args.mask_dir:
        image_dir = args.image_dir
        mask_dir = args.mask_dir
    else:
        print("Error: Must provide either --data_dir or both --image_dir and --mask_dir")
        print("\nFor dataset download instructions, run:")
        print("  python scripts/prepare_data.py --download_help")
        sys.exit(1)

    # Check directories exist
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        print("\nFor dataset download instructions, run:")
        print("  python scripts/prepare_data.py --download_help")
        sys.exit(1)

    if not mask_dir.exists():
        print(f"Error: Mask directory not found: {mask_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    print(f"Image directory: {image_dir}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target splits: {args.train_size} train / {args.val_size} val / {args.test_size} test")

    # Load raw dataset
    print("\nLoading raw dataset...")
    raw_dataset = RawSaliencyDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        max_samples=args.max_samples
    )

    print(f"Found {len(raw_dataset)} valid image-mask pairs")

    # Process all samples
    print("\nProcessing samples (computing bounding boxes from masks)...")
    all_samples = []
    for i in tqdm(range(len(raw_dataset))):
        sample = raw_dataset[i]
        all_samples.append(sample)

    print(f"Processed {len(all_samples)} samples")

    # Create splits
    print("\nCreating train/val/test splits...")
    splits = create_splits(
        all_samples,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # Save splits
    print(f"\nSaving splits to {args.output_dir}...")
    save_all_splits(splits, args.output_dir)

    # Visualize samples if requested
    if args.visualize:
        vis_dir = Path("outputs/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Visualize train samples
        visualize_samples(
            splits["train"],
            num_samples=9,
            output_path=vis_dir / "train_samples.png"
        )

        # Visualize val samples
        visualize_samples(
            splits["val"],
            num_samples=9,
            output_path=vis_dir / "val_samples.png"
        )

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print(f"\nSplit files saved to: {args.output_dir}")
    print("\nYou can now run training with:")
    print("  python scripts/train.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
