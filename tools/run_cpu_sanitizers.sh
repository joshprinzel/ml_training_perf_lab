#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "$PROJECT_ROOT"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"
CXX_COMPILER="${CXX:-/usr/bin/g++}"
MAX_BUILD_JOBS="${MAX_JOBS:-2}"

if [[ ! -x "$CXX_COMPILER" ]]; then
    echo "C++ compiler not found: $CXX_COMPILER" >&2
    exit 1
fi

ASAN_RUNTIME="$(
    "$CXX_COMPILER" -print-file-name=libasan.so
)"

UBSAN_RUNTIME="$(
    "$CXX_COMPILER" -print-file-name=libubsan.so
)"

if [[ "$ASAN_RUNTIME" == "libasan.so" ]] ||
   [[ ! -f "$ASAN_RUNTIME" ]]; then
    echo "Could not locate libasan.so" >&2
    exit 1
fi

if [[ "$UBSAN_RUNTIME" == "libubsan.so" ]] ||
   [[ ! -f "$UBSAN_RUNTIME" ]]; then
    echo "Could not locate libubsan.so" >&2
    exit 1
fi

echo "ASan runtime:  $ASAN_RUNTIME"
echo "UBSan runtime: $UBSAN_RUNTIME"

rm -f python/ml_training_perf/_C*.so
rm -rf build

find "$PROJECT_ROOT" \
    -maxdepth 3 \
    -type d \
    -name "*.egg-info" \
    -prune \
    -exec rm -rf {} +

echo
echo "Building CPU-only sanitizer extension..."

CC="${CC:-/usr/bin/gcc}" \
CXX="$CXX_COMPILER" \
MLTP_BUILD_TYPE=sanitize \
MLTP_ENABLE_CUDA=0 \
MAX_JOBS="$MAX_BUILD_JOBS" \
"$PYTHON_EXECUTABLE" -m pip install \
    --editable . \
    --no-build-isolation

EXTENSION_PATH="$(
    find python/ml_training_perf \
        -maxdepth 1 \
        -type f \
        -name "_C*.so" \
        -print \
        -quit
)"

if [[ -z "$EXTENSION_PATH" ]]; then
    echo "Sanitized extension was not produced" >&2
    exit 1
fi

echo
echo "Extension: $EXTENSION_PATH"

if ! ldd "$EXTENSION_PATH" | grep -q "libasan"; then
    echo "Extension is not linked against libasan" >&2
    exit 1
fi

if ! ldd "$EXTENSION_PATH" | grep -q "libubsan"; then
    echo "Extension is not linked against libubsan" >&2
    exit 1
fi

SANITIZER_PRELOAD="$ASAN_RUNTIME:$UBSAN_RUNTIME"

if [[ -n "${LD_PRELOAD:-}" ]]; then
    SANITIZER_PRELOAD="$SANITIZER_PRELOAD:$LD_PRELOAD"
fi

echo
echo "Running CPU sanitizer smoke test..."

CUDA_VISIBLE_DEVICES="" \
PYTHONMALLOC=malloc \
LD_PRELOAD="$SANITIZER_PRELOAD" \
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
"$PYTHON_EXECUTABLE" tools/cpu_sanitizer_smoke.py