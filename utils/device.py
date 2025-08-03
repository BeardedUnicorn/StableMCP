def get_device():
    """
    Detects available compute:
      - CUDA (GPU)
      - ML Compute (Apple Silicon)
      - CPU
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")

    # ML Compute
    try:
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")
