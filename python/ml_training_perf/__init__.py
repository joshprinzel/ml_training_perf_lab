from __future__ import annotations

# Load PyTorch and its shared libraries before importing the native extension.
import torch as _torch

from . import _C as _C
from .ops import scaled_relu


__all__ = [
    "scaled_relu",
]