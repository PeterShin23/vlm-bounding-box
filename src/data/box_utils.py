"""
Bounding box utilities for converting masks to boxes, normalization, and IoU computation.
"""
import json
import numpy as np
import torch
from typing import Tuple, Optional
from scipy import ndimage


def mask_to_largest_bbox(mask: np.ndarray | torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Compute the tight bounding box around the largest connected foreground component.

    Args:
        mask: Binary mask (H, W) where foreground is > 0

    Returns:
        Tuple of (x_min_px, y_min_px, x_max_px, y_max_px) in pixel coordinates
        Returns (0, 0, 0, 0) if no foreground pixels found
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Ensure binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Check if mask is empty
    if not np.any(binary_mask):
        return (0, 0, 0, 0)

    # Find connected components
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return (0, 0, 0, 0)

    # Find the largest component
    if num_features == 1:
        largest_component = binary_mask
    else:
        # Count pixels in each component
        component_sizes = np.bincount(labeled_mask.ravel())
        # Ignore background (label 0)
        component_sizes[0] = 0
        largest_label = component_sizes.argmax()
        largest_component = (labeled_mask == largest_label)

    # Find bounding box of largest component
    rows = np.any(largest_component, axis=1)
    cols = np.any(largest_component, axis=0)

    if not np.any(rows) or not np.any(cols):
        return (0, 0, 0, 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Note: y_max and x_max are inclusive, but we want exclusive upper bounds
    # for typical bbox representation
    return (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)


def normalize_bbox(
    bbox_px: Tuple[int, int, int, int],
    width: int,
    height: int
) -> Tuple[float, float, float, float]:
    """
    Normalize pixel coordinates to [0, 1] range.

    Args:
        bbox_px: Bounding box in pixel coordinates (x_min, y_min, x_max, y_max)
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Normalized bounding box (x_min, y_min, x_max, y_max) in [0, 1]
    """
    x_min_px, y_min_px, x_max_px, y_max_px = bbox_px

    # Avoid division by zero
    if width == 0 or height == 0:
        return (0.0, 0.0, 0.0, 0.0)

    x_min = x_min_px / width
    y_min = y_min_px / height
    x_max = x_max_px / width
    y_max = y_max_px / height

    # Clamp to [0, 1]
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    return (x_min, y_min, x_max, y_max)


def denormalize_bbox(
    bbox_norm: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    """
    Convert normalized coordinates to pixel coordinates.

    Args:
        bbox_norm: Normalized bounding box (x_min, y_min, x_max, y_max) in [0, 1]
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Bounding box in pixel coordinates (x_min, y_min, x_max, y_max)
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_norm

    x_min_px = int(x_min_norm * width)
    y_min_px = int(y_min_norm * height)
    x_max_px = int(x_max_norm * width)
    y_max_px = int(y_max_norm * height)

    # Clamp to image bounds
    x_min_px = max(0, min(width, x_min_px))
    y_min_px = max(0, min(height, y_min_px))
    x_max_px = max(0, min(width, x_max_px))
    y_max_px = max(0, min(height, y_max_px))

    return (x_min_px, y_min_px, x_max_px, y_max_px)


def bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: First bounding box (x_min, y_min, x_max, y_max)
        box_b: Second bounding box (x_min, y_min, x_max, y_max)

    Returns:
        IoU score in [0, 1]
    """
    # Extract coordinates
    x_min_a, y_min_a, x_max_a, y_max_a = box_a
    x_min_b, y_min_b, x_max_b, y_max_b = box_b

    # Compute intersection
    x_min_i = max(x_min_a, x_min_b)
    y_min_i = max(y_min_a, y_min_b)
    x_max_i = min(x_max_a, x_max_b)
    y_max_i = min(y_max_a, y_max_b)

    # Check if there's an intersection
    if x_max_i <= x_min_i or y_max_i <= y_min_i:
        return 0.0

    # Compute areas
    intersection_area = (x_max_i - x_min_i) * (y_max_i - y_min_i)
    area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)
    area_b = (x_max_b - x_min_b) * (y_max_b - y_min_b)

    # Compute union
    union_area = area_a + area_b - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return float(iou)


def bbox_to_json(
    bbox_norm: Tuple[float, float, float, float],
    precision: int = 4
) -> str:
    """
    Serialize a normalized bounding box to a canonical JSON string.

    Args:
        bbox_norm: Normalized bounding box (x_min, y_min, x_max, y_max)
        precision: Number of decimal places for float formatting

    Returns:
        JSON string with sorted keys and controlled float precision
    """
    x_min, y_min, x_max, y_max = bbox_norm

    # Round to specified precision
    bbox_dict = {
        "x_min": round(x_min, precision),
        "y_min": round(y_min, precision),
        "x_max": round(x_max, precision),
        "y_max": round(y_max, precision),
    }

    # Serialize with sorted keys for consistency
    json_str = json.dumps(bbox_dict, sort_keys=True, separators=(',', ':'))
    return json_str


def json_to_bbox(json_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse a JSON string to extract normalized bounding box coordinates.

    Args:
        json_str: JSON string containing x_min, y_min, x_max, y_max

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) or None if parsing fails
    """
    try:
        # Parse JSON
        bbox_dict = json.loads(json_str)

        # Extract coordinates
        x_min = float(bbox_dict["x_min"])
        y_min = float(bbox_dict["y_min"])
        x_max = float(bbox_dict["x_max"])
        y_max = float(bbox_dict["y_max"])

        # Validate coordinates are in valid range
        if not (0 <= x_min <= 1 and 0 <= y_min <= 1 and
                0 <= x_max <= 1 and 0 <= y_max <= 1):
            return None

        # Validate box is valid (max > min)
        if x_max <= x_min or y_max <= y_min:
            return None

        return (x_min, y_min, x_max, y_max)

    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain additional content.

    This is useful for parsing model outputs that may include extra text
    before or after the JSON.

    Args:
        text: Text that may contain a JSON object

    Returns:
        Extracted JSON string or None if no valid JSON found
    """
    # Try to find JSON object in the text
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    json_str = text[start_idx:end_idx + 1]

    # Validate it's actually valid JSON
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        return None
