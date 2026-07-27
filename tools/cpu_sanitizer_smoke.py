from __future__ import annotations

import math
from collections.abc import Callable

import torch

import ml_training_perf


def expect_runtime_error(
    operation: Callable[[], object],
    expected_message: str,
) -> None:
    try:
        operation()
    except RuntimeError as error:
        actual_message = str(error)

        if expected_message not in actual_message:
            raise AssertionError(
                "RuntimeError did not contain the expected "
                f"message {expected_message!r}.\n"
                f"Actual message: {actual_message!r}"
            ) from error
    else:
        raise AssertionError(
            "Expected RuntimeError containing "
            f"{expected_message!r}"
        )


def check_numerical_cases() -> None:
    sizes = [
        0,
        1,
        255,
        256,
        257,
        1024,
        1025,
        1_000_003,
    ]

    for size in sizes:
        input_tensor = torch.randn(
            size,
            dtype=torch.float32,
        )
        original = input_tensor.clone()

        expected = torch.relu(input_tensor) * 1.75
        actual = ml_training_perf.scaled_relu(
            input_tensor,
            1.75,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
        )

        torch.testing.assert_close(
            input_tensor,
            original,
            rtol=0.0,
            atol=0.0,
        )

        if size > 0:
            assert actual.data_ptr() != input_tensor.data_ptr()


def check_nan_behavior() -> None:
    input_tensor = torch.tensor(
        [
            float("nan"),
            -1.0,
            0.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    actual = ml_training_perf.scaled_relu(
        input_tensor,
        2.0,
    )

    assert torch.isnan(actual[0])
    assert actual[1].item() == 0.0
    assert actual[2].item() == 0.0
    assert actual[3].item() == 4.0


def check_validation_paths() -> None:
    integer_tensor = torch.tensor(
        [1, 2, 3],
        dtype=torch.int64,
    )

    expect_runtime_error(
        lambda: ml_training_perf.scaled_relu(
            integer_tensor,
            2.0,
        ),
        "expected dtype torch.float32",
    )

    noncontiguous_tensor = torch.randn(
        3,
        5,
        dtype=torch.float32,
    ).transpose(0, 1)

    assert not noncontiguous_tensor.is_contiguous()

    expect_runtime_error(
        lambda: ml_training_perf.scaled_relu(
            noncontiguous_tensor,
            2.0,
        ),
        "expected a contiguous tensor",
    )

    for scale in (
        math.inf,
        -math.inf,
        math.nan,
    ):
        expect_runtime_error(
            lambda scale=scale: (
                ml_training_perf.scaled_relu(
                    torch.randn(
                        8,
                        dtype=torch.float32,
                    ),
                    scale,
                )
            ),
            "scale must be finite",
        )


def check_repeated_execution() -> None:
    input_tensor = torch.randn(
        4097,
        dtype=torch.float32,
    )

    expected = torch.relu(input_tensor) * 0.75

    for _ in range(250):
        actual = ml_training_perf.scaled_relu(
            input_tensor,
            0.75,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
        )


def main() -> None:
    check_numerical_cases()
    check_nan_behavior()
    check_validation_paths()
    check_repeated_execution()

    print("CPU ASan/UBSan smoke test passed")


if __name__ == "__main__":
    main()