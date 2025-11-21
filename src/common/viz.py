"""
Visualization utilities for drawing bounding boxes on images.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Tuple, Optional


def draw_bbox_on_image(
    image: Image.Image,
    bbox_norm: Tuple[float, float, float, float],
    color: str = "red",
    width: int = 3,
    label: Optional[str] = None
) -> Image.Image:
    """
    Draw a bounding box on an image.

    Args:
        image: PIL Image to draw on
        bbox_norm: Normalized bounding box (x_min, y_min, x_max, y_max) in [0,1]
        color: Box color (default: red)
        width: Line width in pixels
        label: Optional text label to display above the box

    Returns:
        New PIL Image with bounding box drawn
    """
    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Convert normalized coords to pixel coords
    img_width, img_height = image.size
    x_min = bbox_norm[0] * img_width
    y_min = bbox_norm[1] * img_height
    x_max = bbox_norm[2] * img_width
    y_max = bbox_norm[3] * img_height

    # Draw rectangle
    draw.rectangle(
        [(x_min, y_min), (x_max, y_max)],
        outline=color,
        width=width
    )

    # Draw label if provided
    if label:
        # Try to use a platform-specific font, fall back to default if not available
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 16)
                break
            except (OSError, IOError):
                continue
        if font is None:
            font = ImageFont.load_default()

        # Get text bounding box for background
        text_bbox = draw.textbbox((x_min, y_min - 20), label, font=font)

        # Draw background rectangle for text
        draw.rectangle(text_bbox, fill=color)

        # Draw text
        draw.text((x_min, y_min - 20), label, fill="white", font=font)

    return img_copy


def draw_comparison(
    image: Image.Image,
    gt_bbox_norm: Tuple[float, float, float, float],
    pred_bbox_norm: Tuple[float, float, float, float],
    iou: Optional[float] = None
) -> Image.Image:
    """
    Draw both ground truth and predicted bounding boxes on an image.

    Args:
        image: PIL Image to draw on
        gt_bbox_norm: Ground truth normalized bbox (x_min, y_min, x_max, y_max)
        pred_bbox_norm: Predicted normalized bbox (x_min, y_min, x_max, y_max)
        iou: Optional IoU score to display

    Returns:
        New PIL Image with both boxes drawn
    """
    # Draw GT in green
    img_with_gt = draw_bbox_on_image(
        image,
        gt_bbox_norm,
        color="green",
        width=3,
        label="Ground Truth"
    )

    # Draw prediction in red on top
    label = f"Prediction (IoU: {iou:.3f})" if iou is not None else "Prediction"
    img_with_both = draw_bbox_on_image(
        img_with_gt,
        pred_bbox_norm,
        color="red",
        width=3,
        label=label
    )

    return img_with_both


def create_grid(
    images: list[Image.Image],
    grid_width: int = 3,
    padding: int = 10
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of PIL Images
        grid_width: Number of images per row
        padding: Padding between images in pixels

    Returns:
        Single PIL Image containing the grid
    """
    if not images:
        raise ValueError("No images provided")

    # Get dimensions (assume all images same size)
    img_width, img_height = images[0].size

    # Calculate grid dimensions
    n_images = len(images)
    n_rows = (n_images + grid_width - 1) // grid_width

    # Create blank canvas
    canvas_width = grid_width * img_width + (grid_width + 1) * padding
    canvas_height = n_rows * img_height + (n_rows + 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    # Paste images
    for idx, img in enumerate(images):
        row = idx // grid_width
        col = idx % grid_width
        x = col * img_width + (col + 1) * padding
        y = row * img_height + (row + 1) * padding
        canvas.paste(img, (x, y))

    return canvas


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    """
    Convert a binary mask to an RGB image for visualization.

    Args:
        mask: Binary numpy array (H, W) with values 0 or 1

    Returns:
        PIL Image with mask in white on black background
    """
    # Ensure binary
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Convert to RGB
    rgb = np.stack([mask_binary, mask_binary, mask_binary], axis=-1)

    return Image.fromarray(rgb)
