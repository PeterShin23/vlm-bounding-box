"""
Device selection utilities for local (Mac M-series MPS) and cloud (Runpod CUDA) training.
"""
import torch


def normalize_device(device: str | torch.device) -> str:
    """
    Normalize device to string representation.

    Args:
        device: Device as string ("mps", "cuda", "cpu") or torch.device object

    Returns:
        Device type as string

    Example:
        >>> normalize_device(torch.device("cuda"))
        'cuda'
        >>> normalize_device("mps")
        'mps'
    """
    if isinstance(device, torch.device):
        return device.type
    return device


def get_device(prefer_cuda: bool = False) -> torch.device:
    """
    Select the appropriate device based on availability and preference.

    Priority order:
    1. If prefer_cuda=True and CUDA available -> "cuda"
    2. Else if MPS available (Mac M-series) -> "mps"
    3. Else -> "cpu"

    Args:
        prefer_cuda: If True, prioritize CUDA over MPS (for Runpod training)

    Returns:
        torch.device: The selected device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name(prefer_cuda: bool = False) -> str:
    """
    Returns a human-readable name for the selected device.

    Args:
        prefer_cuda: If True, prioritize CUDA over MPS

    Returns:
        str: Device name like "CUDA (GPU)", "MPS (Apple Silicon)", or "CPU"
    """
    device = get_device(prefer_cuda)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        return f"CUDA ({gpu_name})"
    elif device.type == "mps":
        return "MPS (Apple Silicon)"
    else:
        return "CPU"


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS is available."""
    return torch.backends.mps.is_available()


def print_device_info(prefer_cuda: bool = False):
    """
    Print detailed information about the selected device.

    Args:
        prefer_cuda: If True, prioritize CUDA over MPS
    """
    device = get_device(prefer_cuda)
    device_name = get_device_name(prefer_cuda)

    print("\n" + "=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"Selected device: {device}")
    print(f"Device name: {device_name}")
    print(f"CUDA available: {is_cuda_available()}")
    print(f"MPS available: {is_mps_available()}")

    if device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    print("=" * 50 + "\n")
