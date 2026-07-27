from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def environment_flag(
    name: str,
    *,
    default: bool,
) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be a boolean value; "
        f"received {value!r}"
    )


def selected_build_type() -> str:
    build_type = os.environ.get(
        "MLTP_BUILD_TYPE",
        "debug",
    ).strip().lower()

    valid_build_types = {
        "debug",
        "release",
        "sanitize",
    }

    if build_type not in valid_build_types:
        raise RuntimeError(
            "MLTP_BUILD_TYPE must be one of "
            "'debug', 'release', or 'sanitize'; "
            f"received {build_type!r}"
        )

    return build_type


def cxx_compile_flags(
    build_type: str,
) -> list[str]:
    common_flags = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    ]

    if build_type == "debug":
        return common_flags + [
            "-O0",
            "-g3",
            "-fno-omit-frame-pointer",
            "-D_GLIBCXX_ASSERTIONS",
        ]

    if build_type == "release":
        return common_flags + [
            "-O3",
            "-DNDEBUG",
        ]

    return common_flags + [
        "-O1",
        "-g3",
        "-fno-omit-frame-pointer",
        "-fno-optimize-sibling-calls",
        "-D_GLIBCXX_ASSERTIONS",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
    ]


def nvcc_compile_flags(
    build_type: str,
) -> list[str]:
    common_flags = [
        "-std=c++20",
        "-lineinfo",
    ]

    if build_type == "debug":
        return common_flags + [
            "-O0",
            "-g",
            "-Xcompiler=-fno-omit-frame-pointer",
        ]

    if build_type == "release":
        return common_flags + [
            "-O3",
            "-DNDEBUG",
        ]

    raise RuntimeError(
        "The sanitize build is CPU-only. "
        "Set MLTP_ENABLE_CUDA=0."
    )


def linker_flags(
    build_type: str,
) -> list[str]:
    if build_type == "sanitize":
        return [
            "-fsanitize=address,undefined",
        ]

    return []


BUILD_TYPE = selected_build_type()

ENABLE_CUDA = environment_flag(
    "MLTP_ENABLE_CUDA",
    default=True,
)

if BUILD_TYPE == "sanitize" and ENABLE_CUDA:
    raise RuntimeError(
        "The sanitize build is CPU-only. "
        "Run with MLTP_ENABLE_CUDA=0."
    )


sources = [
    str(PROJECT_ROOT / "cpp/src/bindings.cpp"),
    str(PROJECT_ROOT / "cpp/src/scaled_relu_cpu.cpp"),
]

extra_compile_args: dict[str, list[str]] = {
    "cxx": cxx_compile_flags(BUILD_TYPE),
}

extension_type = CppExtension

if ENABLE_CUDA:
    extension_type = CUDAExtension

    sources.append(
        str(PROJECT_ROOT / "cuda/scaled_relu_cuda.cu")
    )

    extra_compile_args["nvcc"] = nvcc_compile_flags(
        BUILD_TYPE
    )


extension = extension_type(
    name="ml_training_perf._C",
    sources=sources,
    include_dirs=[
        str(PROJECT_ROOT / "cpp/include"),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=linker_flags(BUILD_TYPE),
)


setup(
    name="ml-training-perf",
    version="0.0.1",
    description=(
        "Single-node ML training systems and "
        "performance laboratory"
    ),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=["torch"],
    ext_modules=[extension],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True
        ),
    },
    zip_safe=False,
)