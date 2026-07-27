from __future__ import annotations

import torch

def scaled_relu(
        input_tensor: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Apply max(input, 0) * scale through the custom operator"""

    return torch.ops.ml_training_perf.scaled_relu(
        input_tensor,
        float(scale),
    )