import torch
import pytest

import ml_training_perf


def test_scaled_relu_matches_pytorch_reference() -> None:
    torch.manual_seed(0)

    input_tensor = torch.randn(
        4,
        7,
        dtype=torch.float32,
    )
    scale = 1.75

    expected = torch.relu(input_tensor) * scale
    actual = ml_training_perf.scaled_relu(
        input_tensor,
        scale,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_scaled_relu_supports_empty_tensor() -> None:
    input_tensor = torch.empty(
        0,
        8,
        dtype=torch.float32,
    )

    actual = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    assert actual.shape == input_tensor.shape
    assert actual.dtype == input_tensor.dtype
    assert actual.numel() == 0


def test_scaled_relu_does_not_modify_input() -> None:
    input_tensor = torch.tensor(
        [-2.0, 0.0, 3.0],
        dtype=torch.float32,
    )
    original = input_tensor.clone()

    _ = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    torch.testing.assert_close(
        input_tensor,
        original,
        rtol=0.0,
        atol=0.0,
    )


def test_scaled_relu_propagates_nan() -> None:
    input_tensor = torch.tensor(
        [float("nan"), -1.0, 2.0],
        dtype=torch.float32,
    )

    actual = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    assert torch.isnan(actual[0])
    assert actual[1].item() == 0.0
    assert actual[2].item() == 4.0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_scaled_relu_cuda_matches_pytorch_reference() -> None:
    torch.manual_seed(0)

    input_tensor = torch.randn(
        1025,
        device="cuda",
        dtype=torch.float32,
    )
    scale = 1.75

    expected = torch.relu(input_tensor) * scale
    actual = ml_training_perf.scaled_relu(
        input_tensor,
        scale,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )

    assert actual.device == input_tensor.device
    assert actual.dtype == input_tensor.dtype


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_scaled_relu_cuda_supports_empty_tensor() -> None:
    input_tensor = torch.empty(
        0,
        8,
        device="cuda",
        dtype=torch.float32,
    )

    actual = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    assert actual.shape == input_tensor.shape
    assert actual.device == input_tensor.device
    assert actual.numel() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_scaled_relu_cuda_propagates_nan() -> None:
    input_tensor = torch.tensor(
        [float("nan"), -1.0, 2.0],
        device="cuda",
        dtype=torch.float32,
    )

    actual = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    assert torch.isnan(actual[0])
    assert actual[1].item() == 0.0
    assert actual[2].item() == 4.0