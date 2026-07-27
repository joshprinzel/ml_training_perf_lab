import math

import pytest
import torch

import ml_training_perf


def test_scaled_relu_rejects_integer_tensor() -> None:
    input_tensor = torch.tensor(
        [1, 2, 3],
        dtype=torch.int64,
    )

    with pytest.raises(
        RuntimeError,
        match="expected dtype torch.float32",
    ):
        ml_training_perf.scaled_relu(
            input_tensor,
            2.0,
        )


def test_scaled_relu_rejects_noncontiguous_tensor() -> None:
    input_tensor = torch.randn(
        3,
        5,
        dtype=torch.float32,
    ).transpose(0, 1)

    assert not input_tensor.is_contiguous()

    with pytest.raises(
        RuntimeError,
        match="expected a contiguous tensor",
    ):
        ml_training_perf.scaled_relu(
            input_tensor,
            2.0,
        )


@pytest.mark.parametrize(
    "scale",
    [
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_scaled_relu_rejects_nonfinite_scale(
    scale: float,
) -> None:
    input_tensor = torch.randn(
        8,
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="scale must be finite",
    ):
        ml_training_perf.scaled_relu(
            input_tensor,
            scale,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_scaled_relu_cuda_rejects_integer_tensor() -> None:
    input_tensor = torch.tensor(
        [1, 2, 3],
        device="cuda",
        dtype=torch.int64,
    )

    with pytest.raises(
        RuntimeError,
        match="expected dtype torch.float32",
    ):
        ml_training_perf.scaled_relu(
            input_tensor,
            2.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_scaled_relu_cuda_rejects_noncontiguous_tensor() -> None:
    input_tensor = torch.randn(
        3,
        5,
        device="cuda",
        dtype=torch.float32,
    ).transpose(0, 1)

    assert not input_tensor.is_contiguous()

    with pytest.raises(
        RuntimeError,
        match="expected a contiguous tensor",
    ):
        ml_training_perf.scaled_relu(
            input_tensor,
            2.0,
        )