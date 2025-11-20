"""
Device selection utilities for Mac M-series (MPS) or CPU fallback.
"""
import torch


def get_device() -> torch.device:
    """
    Returns MPS device if available (Mac M-series), otherwise CPU.

    Returns:
        torch.device: The device to use for torch operations
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name() -> str:
    """
    Returns a human-readable name for the current device.

    Returns:
        str: Device name like "MPS (Apple Silicon)" or "CPU"
    """
    device = get_device()
    if device.type == "mps":
        return "MPS (Apple Silicon)"
    return "CPU"
