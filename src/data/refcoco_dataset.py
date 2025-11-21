"""
PyTorch dataset for RefCOCO referring expression grounding.

Loads data from HuggingFace and formats for phrase-conditioned bounding box prediction.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Optional
from datasets import load_dataset

from .box_utils import coco_to_xyxy, normalize_bbox, bbox_to_json


class RefCOCODataset(Dataset):
    """
    Dataset for RefCOCO referring expression grounding.

    Each sample contains:
    - image: PIL Image
    - width, height: Image dimensions
    - phrase: Referring expression (text description)
    - bbox_norm: Normalized ground truth bounding box (x_min, y_min, x_max, y_max)
    - bbox_json: JSON string representation of the normalized box
    """

    def __init__(
        self,
        split: str = "train",
        dataset_name: str = "lmms-lab/RefCOCO",
        max_samples: Optional[int] = None,
        refcoco_split: str = "refcoco",  # "refcoco", "refcoco+", or "refcocog"
    ):
        """
        Initialize RefCOCO dataset.

        Args:
            split: Dataset split ("train", "validation", or "test")
            dataset_name: HuggingFace dataset name
            max_samples: Maximum number of samples to use (for budget-aware training)
            refcoco_split: Which RefCOCO variant to use
        """
        self.split = split
        self.max_samples = max_samples

        print(f"\nLoading RefCOCO dataset: {dataset_name}")
        print(f"  Split: {split}")
        print(f"  Variant: {refcoco_split}")
        if max_samples:
            print(f"  Max samples: {max_samples}")

        # Load dataset from HuggingFace
        # Note: lmms-lab/RefCOCO uses a single 'default' config
        try:
            # Try loading with just the dataset name and split
            self.dataset = load_dataset(
                dataset_name,
                split=split
            )
        except (ValueError, KeyError, OSError) as e:
            print(f"Error loading dataset with split: {e}")
            print("Attempting to load full dataset...")
            # Fallback: load full dataset then select split
            try:
                full_dataset = load_dataset(dataset_name)
                self.dataset = full_dataset[split]
            except (ValueError, KeyError, OSError) as e2:
                print(f"Error loading full dataset: {e2}")
                raise ValueError(
                    f"Could not load {dataset_name}. Please check the dataset name and your internet connection."
                ) from e2

        # Limit samples if requested
        if max_samples and max_samples < len(self.dataset):
            # Use deterministic sampling for reproducibility
            indices = list(range(min(max_samples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - image: PIL Image
            - width: int (image width)
            - height: int (image height)
            - phrase: str (referring expression)
            - bbox_norm: tuple of (x_min, y_min, x_max, y_max) in [0, 1]
            - bbox_json: str (JSON representation)
        """
        sample = self.dataset[idx]

        # Extract image
        # Note: Field names may vary depending on HF dataset structure
        # Common field names: "image", "img", "image_data"
        if "image" in sample:
            image = sample["image"]
        elif "img" in sample:
            image = sample["img"]
        else:
            raise KeyError(f"Could not find image field in sample. Available keys: {sample.keys()}")

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get image dimensions
        width, height = image.size

        # Extract referring expression
        # For lmms-lab/RefCOCO, the phrase is in "question" field
        if "question" in sample:
            phrase = sample["question"]
        elif "sentence" in sample:
            phrase = sample["sentence"]
        elif "expression" in sample:
            phrase = sample["expression"]
        elif "text" in sample:
            phrase = sample["text"]
        elif "phrase" in sample:
            phrase = sample["phrase"]
        else:
            raise KeyError(f"Could not find phrase field in sample. Available keys: {sample.keys()}")

        # Extract bounding box in COCO format
        # Common field names: "bbox", "box", "bounding_box"
        if "bbox" in sample:
            bbox_coco = sample["bbox"]
        elif "box" in sample:
            bbox_coco = sample["box"]
        elif "bounding_box" in sample:
            bbox_coco = sample["bounding_box"]
        else:
            raise KeyError(f"Could not find bbox field in sample. Available keys: {sample.keys()}")

        # Convert COCO format [x, y, w, h] to [x_min, y_min, x_max, y_max]
        bbox_px = coco_to_xyxy(bbox_coco)

        # Normalize coordinates
        bbox_norm = normalize_bbox(bbox_px, width, height)

        # Convert to JSON string
        bbox_json = bbox_to_json(bbox_norm)

        return {
            "image": image,
            "width": width,
            "height": height,
            "phrase": phrase,
            "bbox_norm": bbox_norm,
            "bbox_json": bbox_json,
        }


def create_refcoco_dataloader(
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None,
    collate_fn=None,
    device: Optional[str] = None,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for RefCOCO dataset with device-aware defaults.

    Args:
        split: Dataset split
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (None = auto-select based on device:
                     0 for MPS, 2 for CUDA/CPU)
        max_samples: Maximum samples for budget-aware training
        collate_fn: Optional custom collate function
        device: Device type ("mps", "cuda", "cpu") for optimization defaults.
                If None, will use sensible defaults.
        **dataset_kwargs: Additional arguments for RefCOCODataset

    Returns:
        DataLoader
    """
    dataset = RefCOCODataset(
        split=split,
        max_samples=max_samples,
        **dataset_kwargs
    )

    # Auto-select num_workers based on device if not specified
    if num_workers is None:
        # MPS requires num_workers=0 due to multiprocessing issues
        # CUDA/CPU can benefit from parallel data loading
        if device == "mps":
            num_workers = 0
        else:
            num_workers = 2  # Safe default for CUDA/CPU

    # Auto-select pin_memory based on device
    # pin_memory improves CUDA transfer speed but MPS doesn't support it
    use_pin_memory = (device == "cuda")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
