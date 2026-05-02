import os
import torch

# Fix OpenMP conflict between LightGBM and PyTorch on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def get_device() -> str:
    # MPS has high overhead for small models/datasets; CPU is faster here
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
