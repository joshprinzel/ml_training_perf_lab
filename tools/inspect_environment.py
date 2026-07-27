#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])

    if executable is None:
        return {
            "available": False,
            "command": command,
            "output": None,
        }

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": True,
            "command": command,
            "return_code": None,
            "output": None,
            "error": str(error),
        }

    output = result.stdout.strip()

    if result.stderr.strip():
        output = "\n".join(
            part for part in (output, result.stderr.strip()) if part
        )

    return {
        "available": True,
        "command": command,
        "return_code": result.returncode,
        "output": output,
    }


def collect_pytorch_information() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        return {
            "available": False,
            "error": str(error),
        }

    cuda_available = torch.cuda.is_available()

    information: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "installation_path": str(Path(torch.__file__).resolve()),
        "built_with_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cxx11_abi": getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
    }

    if cuda_available:
        devices = []

        for device_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device_index)

            devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_bytes": properties.total_memory,
                    "multiprocessor_count": properties.multi_processor_count,
                }
            )

        information["devices"] = devices

    return information


def main() -> int:
    report = {
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "environment": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "CXX": os.environ.get("CXX"),
            "CC": os.environ.get("CC"),
        },
        "tools": {
            "cmake": run_command(["cmake", "--version"]),
            "ninja": run_command(["ninja", "--version"]),
            "gcc": run_command(["gcc", "--version"]),
            "g++": run_command(["g++", "--version"]),
            "clang": run_command(["clang", "--version"]),
            "nvcc": run_command(["nvcc", "--version"]),
            "nvidia_smi": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,"
                    "compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "compute_sanitizer": run_command(
                ["compute-sanitizer", "--version"]
            ),
        },
        "pytorch": collect_pytorch_information(),
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())