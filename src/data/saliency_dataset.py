"""
PyTorch dataset for saliency-based main subject bounding box detection.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as T

from .box_utils import mask_to_largest_bbox, normalize_bbox, bbox_to_json
from .split_utils import DataSample, load_split_from_json


class SaliencyMainSubjectDataset(Dataset):
    """
    Dataset for training VLM to predict bounding boxes around main subjects.

    Each sample contains:
    - image: PIL Image
    - width, height: Image dimensions
    - gt_box_norm: Normalized ground truth bounding box (x_min, y_min, x_max, y_max)
    - gt_box_json: JSON string representation of the box
    """

    def __init__(
        self,
        split_json_path: Optional[Path] = None,
        samples: Optional[List[DataSample]] = None,
        image_root: Optional[Path] = None,
        mask_root: Optional[Path] = None,
        resize: Optional[int] = None,
        augment: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset.

        Args:
            split_json_path: Path to JSON file containing split data
            samples: List of DataSample objects (alternative to split_json_path)
            image_root: Root directory for images (if paths in samples are relative)
            mask_root: Root directory for masks (if paths in samples are relative)
            resize: If provided, resize images to this size (shorter edge)
            augment: Whether to apply data augmentation
            max_samples: Maximum number of samples to use (for debugging)
        """
        # Load samples from JSON or use provided samples
        if split_json_path is not None:
            self.samples = load_split_from_json(split_json_path)
        elif samples is not None:
            self.samples = samples
        else:
            raise ValueError("Must provide either split_json_path or samples")

        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        self.image_root = Path(image_root) if image_root else None
        self.mask_root = Path(mask_root) if mask_root else None
        self.resize = resize
        self.augment = augment

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup image transforms."""
        transform_list = []

        # Basic transforms (always applied)
        if self.resize:
            transform_list.append(T.Resize(self.resize))

        # Augmentation (only for training)
        if self.augment:
            transform_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                # Could add more augmentations here
            ])

        # Note: We don't convert to tensor or normalize here
        # because Qwen3-VL's processor will handle that
        self.transform = T.Compose(transform_list) if transform_list else None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - image: PIL Image
            - width: int (original or resized width)
            - height: int (original or resized height)
            - gt_box_norm: tuple of (x_min, y_min, x_max, y_max) in [0, 1]
            - gt_box_json: str (JSON representation)
        """
        sample = self.samples[idx]

        # Load image
        image_path = Path(sample.image_path)
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Get current size (may have changed due to transforms)
        width, height = image.size

        # Get ground truth box (already normalized in the sample)
        gt_box_norm = sample.bbox_norm
        gt_box_json = sample.bbox_json

        return {
            "image": image,
            "width": width,
            "height": height,
            "gt_box_norm": gt_box_norm,
            "gt_box_json": gt_box_json,
        }


class RawSaliencyDataset(Dataset):
    """
    Dataset for raw saliency data (images + masks) that computes bounding boxes on the fly.

    This is useful for the data preparation stage before creating splits.
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        max_samples: Optional[int] = None
    ):
        """
        Initialize raw dataset.

        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing saliency masks
            image_suffix: File extension for images
            mask_suffix: File extension for masks
            max_samples: Maximum number of samples to load
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

        # Find all images
        self.image_paths = sorted(self.image_dir.glob(f"*{image_suffix}"))

        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]

        # Verify corresponding masks exist
        self._verify_masks()

    def _verify_masks(self):
        """Verify that all images have corresponding masks."""
        valid_paths = []
        for img_path in self.image_paths:
            # Get mask path
            mask_name = img_path.stem + self.mask_suffix
            mask_path = self.mask_dir / mask_name

            if mask_path.exists():
                valid_paths.append(img_path)
            else:
                print(f"Warning: No mask found for {img_path.name}, skipping")

        self.image_paths = valid_paths
        print(f"Found {len(self.image_paths)} valid image-mask pairs")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> DataSample:
        """
        Get a single sample and compute its bounding box.

        Returns:
            DataSample with computed bounding box
        """
        # Get image path
        image_path = self.image_paths[idx]

        # Get mask path
        mask_name = image_path.stem + self.mask_suffix
        mask_path = self.mask_dir / mask_name

        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        # Compute bounding box from mask
        bbox_px = mask_to_largest_bbox(mask_array)

        # Normalize bbox
        bbox_norm = normalize_bbox(bbox_px, width, height)

        # Convert to JSON
        bbox_json = bbox_to_json(bbox_norm)

        # Create DataSample
        # Store paths as strings (relative to dataset root)
        return DataSample(
            image_path=str(image_path),
            mask_path=str(mask_path),
            width=width,
            height=height,
            bbox_norm=bbox_norm,
            bbox_json=bbox_json
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (use 0 for MPS compatibility)
        collate_fn: Optional custom collate function

    Returns:
        DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # MPS doesn't support pin_memory
    )
