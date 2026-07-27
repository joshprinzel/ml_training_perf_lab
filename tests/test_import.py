import torch

import ml_training_perf

def test_extension_registration_scaled_relu() -> None:
    assert ml_training_perf is not None
    assert hasattr(torch.ops.ml_training_perf, "scaled_relu")
