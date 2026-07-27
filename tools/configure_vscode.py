#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import sys
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VSCODE_DIRECTORY = PROJECT_ROOT / ".vscode"


def normalized_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def unique_paths(paths: list[str | Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for path in paths:
        normalized = normalized_path(path)

        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def main() -> int:
    compiler = shutil.which("g++")

    if compiler is None:
        raise RuntimeError("Could not locate g++ in PATH")

    python_include = sysconfig.get_path("include")

    if python_include is None:
        raise RuntimeError("Could not determine the Python include directory")

    torch_include_paths = include_paths(device_type="cuda")

    include_directories = unique_paths(
        [
            PROJECT_ROOT / "cpp/include",
            PROJECT_ROOT / "cpp/src",
            PROJECT_ROOT / "cuda",
            python_include,
            *torch_include_paths,
        ]
    )

    c_cpp_properties = {
        "configurations": [
            {
                "name": "WSL",
                "compilerPath": compiler,
                "compilerArgs": [
                    "-pthread",
                ],
                "includePath": [
                    "${workspaceFolder}/**",
                    *include_directories,
                ],
                "defines": [
                    "TORCH_EXTENSION_NAME=_C",
                    f"_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                ],
                "cStandard": "c17",
                "cppStandard": "c++20",
                "intelliSenseMode": "linux-gcc-x64",
                "browse": {
                    "path": [
                        "${workspaceFolder}",
                        *include_directories,
                    ],
                    "limitSymbolsToIncludedHeaders": True,
                },
            }
        ],
        "version": 4,
    }

    settings = {
        "C_Cpp.default.compilerPath": compiler,
        "C_Cpp.default.cppStandard": "c++20",
        "C_Cpp.default.intelliSenseMode": "linux-gcc-x64",
        "C_Cpp.intelliSenseEngine": "default",
        "C_Cpp.errorSquiggles": "enabled",
        "python.defaultInterpreterPath": sys.executable,
        "python.analysis.extraPaths": [
            "${workspaceFolder}/python",
        ],
    }

    extensions = {
        "recommendations": [
            "ms-vscode.cpptools",
            "ms-python.python",
            "ms-vscode-remote.remote-wsl",
        ]
    }

    VSCODE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    files = {
        VSCODE_DIRECTORY / "c_cpp_properties.json": c_cpp_properties,
        VSCODE_DIRECTORY / "settings.json": settings,
        VSCODE_DIRECTORY / "extensions.json": extensions,
    }

    for path, contents in files.items():
        path.write_text(
            json.dumps(contents, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"Compiler: {compiler}")
    print(f"Python:   {sys.executable}")
    print(f"PyTorch:  {torch.__version__}")
    print("Include directories:")

    for include_directory in include_directories:
        print(f"  {include_directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())