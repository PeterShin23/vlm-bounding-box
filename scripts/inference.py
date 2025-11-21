#!/usr/bin/env python3
"""
Inference script for RefCOCO phrase grounding - predict bounding boxes from referring expressions.

Usage:
    python scripts/inference.py \
        --image path/to/image.jpg \
        --phrase "the red car on the left" \
        --checkpoint outputs/checkpoints/final
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.inference import load_model_and_predict


def main():
    parser = argparse.ArgumentParser(
        description="Run RefCOCO phrase grounding inference on a single image"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--phrase",
        type=str,
        required=True,
        help="Referring expression (e.g., 'the red car on the left', 'person wearing blue shirt')"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint (if None, uses base model)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device to use"
    )

    args = parser.parse_args()

    # Check image exists
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    # Run prediction
    result = load_model_and_predict(
        image_path=args.image,
        phrase=args.phrase,
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=args.device,
        visualize=True
    )

    # Save visualization if requested
    if args.output and result["visualization"] is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["visualization"].save(args.output)
        print(f"\nVisualization saved to: {args.output}")


if __name__ == "__main__":
    main()
