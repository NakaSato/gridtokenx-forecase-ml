import os
import torch

# Fix OpenMP conflict between LightGBM and PyTorch on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
