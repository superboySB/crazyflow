# syntax=docker/dockerfile:1
# One image supports automatic GPU selection and CPU fallback.
FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCIPY_ARRAY_API=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MUJOCO_GL=osmesa \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Software OpenGL works headlessly on both CPU-only hosts and NVIDIA hosts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libosmesa6 libglib2.0-0 libgomp1 \
        xvfb xauth libglfw3 libxrandr2 libxinerama1 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/crazyflow
# The CUDA wheels are contained in the image. Host Python/Pixi/Conda/CUDA are unused.
COPY requirements-cpu.lock requirements-gpu.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --requirement requirements-gpu.lock

# The toolkit supplies the matching NVIDIA driver library at runtime. Register
# its EGL vendor so headless rendering can discover it without host file binds.
# GLVND skips this vendor on CPU-only hosts, where OSMesa remains the default.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY pyproject.toml README.md LICENSE ./
COPY docs ./docs
COPY config ./config
COPY examples ./examples
COPY models/crazyflow ./models/crazyflow
COPY assets ./assets
COPY crazyrl ./crazyrl
COPY scripts ./scripts
COPY crazyflow ./crazyflow
COPY tests ./tests
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && python -m pip check \
    && JAX_PLATFORMS=cpu python -c "import jax, crazyflow; print(jax.devices())"

# Persist only experiment outputs; the source and runtime are inside the image.
RUN mkdir /outputs && chmod 777 /outputs
COPY docker/entrypoint.sh /usr/local/bin/crazyflow-entrypoint
RUN chmod +x /usr/local/bin/crazyflow-entrypoint
ENTRYPOINT ["/usr/local/bin/crazyflow-entrypoint"]
CMD ["python", "-m", "crazyflow.tof.train", "--help"]
