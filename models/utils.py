import torch


def get_device() -> torch.device:
    """
    Returns the appropriate torch.device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")