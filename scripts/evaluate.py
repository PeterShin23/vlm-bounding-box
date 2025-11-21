#!/usr/bin/env python3
"""
Evaluation script for RefCOCO phrase grounding models.

Usage:
    # Evaluate on test split (recommended)
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final --split test

    # Evaluate on val split
    python scripts/evaluate.py --checkpoint outputs/checkpoints/final --split val
"""
import argparse
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.device import get_device, get_device_name
from src.data.refcoco_dataset import create_refcoco_dataloader
from src.pipeline.model_qwen3 import load_qwen3_vl_with_lora, load_lora_weights
from src.pipeline.eval import Evaluator


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Custom collate function for batching."""
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "testA", "testB"],
        default="test",
        help="Which split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to training config file"
    )
    parser.add_argument(
        "--data_config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data config file"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate (for debugging)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (1 recommended for generation)"
    )

    args = parser.parse_args()

    # Load configs
    print("Loading configurations...")
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Get device
    device = get_device()
    print(f"Using device: {get_device_name()}")

    # Load RefCOCO dataset
    print(f"\nLoading RefCOCO {args.split} split...")
    if args.split == "val":
        print("NOTE: This is the TRAINING split (RefCOCO uses 'val' for training)")

    dataloader = create_refcoco_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        max_samples=args.max_samples,
        collate_fn=collate_fn
    )

    print(f"Loaded {len(dataloader.dataset)} samples")

    # Load model
    print("\nLoading model...")
    model, processor = load_qwen3_vl_with_lora(
        model_name=train_config["model"]["name"],
        use_quantization=False,  # Don't quantize for evaluation
        lora_r=train_config["lora"]["r"],
        lora_alpha=train_config["lora"]["alpha"],
        lora_dropout=train_config["lora"]["dropout"],
        target_modules=train_config["lora"].get("target_modules"),
        device=str(device)
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_lora_weights(model, str(args.checkpoint))

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(
        model=model,
        processor=processor,
        device=str(device),
        iou_thresholds=train_config["evaluation"]["iou_thresholds"],
        max_new_tokens=train_config["evaluation"]["max_new_tokens"]
    )

    # Determine output path
    if args.output_path is None:
        args.output_path = Path("outputs/evaluations") / f"{args.checkpoint.name}_{args.split}.json"

    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        dataloader=dataloader,
        max_samples=args.max_samples,
        save_predictions=train_config["evaluation"]["save_predictions"],
        output_path=args.output_path
    )

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
