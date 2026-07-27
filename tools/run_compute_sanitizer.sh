#!/usr/bin/env bash

set -euo pipefail

compute-sanitizer \
    --tool memcheck \
    --error-exitcode 1 \
    python tools/cuda_sanitizer_smoke.py