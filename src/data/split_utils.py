"""
Utilities for splitting datasets into train/val/test sets.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataSample:
    """Represents a single data sample with image path and metadata."""
    image_path: str
    mask_path: str
    width: int
    height: int
    bbox_norm: Tuple[float, float, float, float]
    bbox_json: str


def create_splits(
    samples: List[DataSample],
    train_size: int = 2000,
    val_size: int = 500,
    test_size: int = 500,
    seed: int = 42
) -> Dict[str, List[DataSample]]:
    """
    Split samples into train/val/test sets.

    Args:
        samples: List of DataSample objects
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        seed: Random seed for reproducibility

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing sample lists
    """
    # Set random seed
    random.seed(seed)

    # Shuffle samples
    samples_shuffled = samples.copy()
    random.shuffle(samples_shuffled)

    # Calculate total needed
    total_needed = train_size + val_size + test_size

    # Check if we have enough samples
    if len(samples_shuffled) < total_needed:
        print(f"Warning: Only {len(samples_shuffled)} samples available, "
              f"but {total_needed} requested. Using all available samples.")

        # Adjust sizes proportionally
        available = len(samples_shuffled)
        ratio = available / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = available - train_size - val_size

    # Create splits
    train_samples = samples_shuffled[:train_size]
    val_samples = samples_shuffled[train_size:train_size + val_size]
    test_samples = samples_shuffled[train_size + val_size:train_size + val_size + test_size]

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }


def save_split_to_json(samples: List[DataSample], output_path: Path):
    """
    Save a list of samples to a JSON file.

    Args:
        samples: List of DataSample objects
        output_path: Path to output JSON file
    """
    # Convert dataclasses to dicts
    samples_dict = [asdict(sample) for sample in samples]

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(samples_dict, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def load_split_from_json(input_path: Path) -> List[DataSample]:
    """
    Load samples from a JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        List of DataSample objects
    """
    with open(input_path, 'r') as f:
        samples_dict = json.load(f)

    # Convert dicts to dataclasses
    samples = [DataSample(**sample) for sample in samples_dict]

    print(f"Loaded {len(samples)} samples from {input_path}")
    return samples


def save_all_splits(
    splits: Dict[str, List[DataSample]],
    output_dir: Path
):
    """
    Save train/val/test splits to JSON files.

    Args:
        splits: Dictionary with 'train', 'val', 'test' keys
        output_dir: Directory to save split files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        output_path = output_dir / f"{split_name}.json"
        save_split_to_json(samples, output_path)

    # Also save split statistics
    stats = {
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
        "total": sum(len(samples) for samples in splits.values())
    }

    stats_path = output_dir / "split_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSplit statistics saved to {stats_path}")
    print(f"Train: {stats['train_size']}")
    print(f"Val: {stats['val_size']}")
    print(f"Test: {stats['test_size']}")
    print(f"Total: {stats['total']}")


def load_all_splits(input_dir: Path) -> Dict[str, List[DataSample]]:
    """
    Load train/val/test splits from JSON files.

    Args:
        input_dir: Directory containing split files

    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_path = input_dir / f"{split_name}.json"
        if split_path.exists():
            splits[split_name] = load_split_from_json(split_path)
        else:
            print(f"Warning: {split_path} not found, skipping {split_name} split")

    return splits
