#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen3-VL on main subject detection.

Usage:
    python scripts/train.py --config configs/training.yaml
"""
import argparse
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.device import get_device, get_device_name
from src.common.paths import ProjectPaths
from src.data.saliency_dataset import SaliencyMainSubjectDataset, create_dataloader
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
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on bounding box detection")
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
        "--processed_data_dir",
        type=Path,
        help="Override processed data directory from config"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (use small subset of data)"
    )

    args = parser.parse_args()

    # Load configs
    print("Loading configurations...")
    train_config = load_config(args.config)
    data_config = load_config(args.data_config)

    # Override configs from CLI args
    if args.processed_data_dir:
        data_config["data"]["processed_data_dir"] = str(args.processed_data_dir)
    if args.output_dir:
        train_config["logging"]["output_dir"] = str(args.output_dir)

    # Debug mode: reduce data and epochs
    if args.debug:
        print("\n*** DEBUG MODE ENABLED ***")
        data_config["processing"]["max_samples"] = 100
        train_config["training"]["num_epochs"] = 1
        data_config["dataloader"]["batch_size"] = 2

    # Setup paths
    paths = ProjectPaths()
    paths.create_directories()

    processed_dir = Path(data_config["data"]["processed_data_dir"])

    # Check if processed data exists
    if not (processed_dir / "train.json").exists():
        print(f"\nError: Processed data not found at {processed_dir}")
        print("Please run data preparation first:")
        print("  python scripts/prepare_data.py --data_dir /path/to/dataset")
        sys.exit(1)

    # Get device
    device = get_device()
    print(f"\nUsing device: {get_device_name()}")

    # Load datasets
    print("\nLoading datasets...")

    train_dataset = SaliencyMainSubjectDataset(
        split_json_path=processed_dir / "train.json",
        resize=data_config["processing"].get("resize"),
        augment=data_config["processing"].get("augment_train", False),
        max_samples=data_config["processing"].get("max_samples")
    )

    val_dataset = SaliencyMainSubjectDataset(
        split_json_path=processed_dir / "val.json",
        resize=data_config["processing"].get("resize"),
        augment=False,
        max_samples=data_config["processing"].get("max_samples")
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=data_config["dataloader"].get("num_workers", 0),
        collate_fn=collate_fn
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=data_config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=data_config["dataloader"].get("num_workers", 0),
        collate_fn=collate_fn
    )

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
