#!/bin/sh
set -eu

# No NVIDIA devices were passed through: select CPU before CUDA plugin discovery.
# Explicit user backend settings still take precedence.
if [ ! -c /dev/nvidiactl ] && [ -z "${JAX_PLATFORMS:-}" ]; then
    export JAX_PLATFORMS=cpu
fi
exec "$@"
