# Reuse the pinned Isaac Sim 4.5 / IsaacLab / RSL stack, with a dedicated image.
ARG CRAZYRL_BASE_IMAGE=crazyrl_image:sim
FROM ${CRAZYRL_BASE_IMAGE}
# TensorBoard auto-discovers Open3D's plugin; its wheel needs libusb.
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/workspace/crazyflow/sim2sim/reference:/workspace/isaaclab \
    CRAZYRL_ASSET_ROOT=/workspace/crazyflow/sim2sim/reference/assets
WORKDIR /workspace/crazyflow
ENTRYPOINT ["/bin/bash"]
