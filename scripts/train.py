#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen3-VL on RefCOCO phrase grounding.

Usage:
    # Local debug (MPS)
    python scripts/train.py --debug --max_train_samples 500

    # Runpod training (CUDA)
    python scripts/train.py --prefer_cuda --max_train_samples 5000 --num_epochs 2
"""
import argparse
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.device import get_device, get_device_name
from src.common.paths import ProjectPaths
from src.data.refcoco_dataset import create_refcoco_dataloader
from src.pipeline.model_qwen3 import load_qwen3_vl_with_lora
from src.pipeline.training import Trainer


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """Custom collate function for batching."""
    # Group items by key
    collated = {}
    for key in batch[0].keys():
        collated[key] = [item[key] for item in batch]
    return collated


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on RefCOCO phrase grounding")
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
        "--output_dir",
        type=Path,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (use small subset of data, 1 epoch)"
    )
    parser.add_argument(
        "--prefer_cuda",
        action="store_true",
        help="Prefer CUDA over MPS (for Runpod training)"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        help="Maximum training samples (for budget-aware training)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum training steps (early stopping)"
    )

    args = parser.parse_args()

    # Load configs
    print("Loading configurations...")
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Override configs from CLI args
    if args.output_dir:
        train_config["logging"]["output_dir"] = str(args.output_dir)
    if args.num_epochs:
        train_config["training"]["num_epochs"] = args.num_epochs
    if args.max_steps:
        train_config["training"]["max_steps"] = args.max_steps

    # Determine max samples
    max_train_samples = args.max_train_samples or data_config["sampling"].get("max_train_samples")
    max_val_samples = data_config["sampling"].get("max_val_samples")

    # Debug mode: reduce data and epochs
    if args.debug:
        print("\n*** DEBUG MODE ENABLED ***")
        max_train_samples = 100
        max_val_samples = 50
        train_config["training"]["num_epochs"] = 1
        data_config["dataloader"]["batch_size"] = 2

    # Setup paths
    paths = ProjectPaths()
    paths.create_directories()

    # Get device (with CUDA preference for Runpod)
    device = get_device(prefer_cuda=args.prefer_cuda)
    print(f"\nUsing device: {get_device_name(prefer_cuda=args.prefer_cuda)}")

    # Load RefCOCO datasets
    # IMPORTANT: RefCOCO uses "val" split for TRAINING and "test" split for EVALUATION
    print("\nLoading RefCOCO dataset...")
    print(f"NOTE: Using 'val' split for TRAINING (RefCOCO has no 'train' split)")

    # Get device string for dataloader optimization
    device_str = device.type

    train_loader = create_refcoco_dataloader(
        split="val",  # TRAINING data
        batch_size=data_config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=data_config["dataloader"].get("num_workers"),  # None = auto-select
        max_samples=max_train_samples,
        collate_fn=collate_fn,
        device=device_str  # Pass device for optimization
    )

    val_loader = create_refcoco_dataloader(
        split="test",  # VALIDATION/EVALUATION data
        batch_size=data_config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=data_config["dataloader"].get("num_workers"),  # None = auto-select
        max_samples=max_val_samples,
        collate_fn=collate_fn,
        device=device_str  # Pass device for optimization
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Load model
    print("\nLoading model...")
    model, processor = load_qwen3_vl_with_lora(
        model_name=train_config["model"]["name"],
        use_quantization=train_config["model"]["use_quantization"],
        quantization_bits=train_config["model"]["quantization_bits"],
        lora_r=train_config["lora"]["r"],
        lora_alpha=train_config["lora"]["alpha"],
        lora_dropout=train_config["lora"]["dropout"],
        target_modules=train_config["lora"].get("target_modules"),
        device=str(device)
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        processor=processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=train_config["training"]["learning_rate"],
        num_epochs=train_config["training"]["num_epochs"],
        warmup_steps=train_config["training"]["warmup_steps"],
        gradient_accumulation_steps=train_config["training"]["gradient_accumulation_steps"],
        max_grad_norm=train_config["training"]["max_grad_norm"],
        device=str(device),
        output_dir=Path(train_config["logging"]["output_dir"]),
        log_dir=Path(train_config["logging"]["log_dir"]),
        logging_steps=train_config["logging"]["logging_steps"],
        eval_steps=train_config["logging"].get("eval_steps"),
        save_steps=train_config["logging"].get("save_steps"),
    )

    # Start training
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
