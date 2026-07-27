from __future__ import annotations

import torch

import ml_training_perf


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

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
            device="cuda",
            dtype=torch.float32,
        )

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

    # Surface asynchronous execution errors before process exit.
    torch.cuda.synchronize()

    print("CUDA sanitizer smoke test passed")


if __name__ == "__main__":
    main()