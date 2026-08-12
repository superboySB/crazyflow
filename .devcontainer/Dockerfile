FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Crazyflow/lsy_drone_racing build tools, headless OpenGL, editors, and
# joystick diagnostics.  Cache removal is deliberately consolidated in the
# final layer so that intermediate setup steps remain debuggable.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    dbus-x11 \
    gedit \
    git \
    htop \
    joystick \
    libegl1 \
    libgl1 \
    libgl1-mesa-dri \
    libgles2 \
    libopenblas-dev \
    libx11-6 \
    libxext6 \
    mesa-utils \
    nano \
    openssh-client \
    tmux \
    vim \
    wget

# Fail the image build immediately if the Docker builder's proxy/network path
# reaches Google with an HTTP error (including the previously observed 503).
RUN curl --fail --location --silent --show-error \
    --retry 3 --retry-all-errors --connect-timeout 15 --max-time 60 \
    --output /dev/null \
    --write-out "docker-build google status=%{http_code} remote=%{remote_ip}\n" \
    https://www.google.com/

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" \
    && mkdir -p /opt/crazyflow /workspace/crazyflow \
    && chown -R "${USERNAME}:${USERNAME}" /opt/crazyflow /workspace

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:/home/user/.pixi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV SCIPY_ARRAY_API=1

USER ${USERNAME}

# Match the Pixi version used by CI and install the repository's exact locked
# GPU test environment.  The Linux lock also includes splax[viewer].
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=v0.76.0 bash

WORKDIR /opt/crazyflow
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pixi install --locked -e gpu-tests

# Keep the environment inside the image.  PYTHONPATH makes a bind-mounted
# checkout at /workspace/crazyflow take precedence over the build-time copy.
ENV PATH="/opt/crazyflow/.pixi/envs/gpu-tests/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nvshmem/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Gymnasium 1.3 still calls the six-argument camera API removed in MuJoCo
# 3.11.  Keep Crazyflow's source untouched by pinning the last compatible
# MuJoCo/MJX pair inside this container environment.
ARG MUJOCO_VERSION=3.10.0
RUN python -m pip install --no-cache-dir --no-deps --force-reinstall \
        "mujoco==${MUJOCO_VERSION}" \
        "mujoco-mjx==${MUJOCO_VERSION}" \
    && python -c 'import mujoco; assert mujoco.__version__ == "3.10.0"'

# Pin the externally maintained racing course so rebuilds do not silently
# change behavior.  Install its simulation, optional gamepad, and RL packages
# into the same CUDA-enabled Python environment as Crazyflow.  The simulator
# runs on JAX/CUDA while the compact PPO policy uses the CPU PyTorch wheel.
ARG LSY_DRONE_RACING_REF=9ecb1cb0a78a232b7cad764344797bfa1c45f462
USER root
RUN mkdir -p /opt/lsy_drone_racing \
    && chown "${USERNAME}:${USERNAME}" /opt/lsy_drone_racing
USER ${USERNAME}
WORKDIR /opt
RUN git clone https://github.com/learnsyslab/lsy_drone_racing.git \
        /opt/lsy_drone_racing \
    && cd /opt/lsy_drone_racing \
    && git checkout --detach "${LSY_DRONE_RACING_REF}" \
    && git apply /opt/crazyflow/docker/lsy_drone_racing-scipy.patch \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
    && python -m pip install wandb==0.28.1 \
    && python -m pip install -e ".[gamepad,rl]"

# The default racing controllers use acados.  The upstream setup script pins
# acados v0.5.1 and t_renderer v0.0.34, builds the native libraries, and
# installs acados_template into the active Python environment.
WORKDIR /opt/lsy_drone_racing
RUN PIXI_PROJECT_ROOT=/opt/lsy_drone_racing bash tools/setup_acados.sh \
    && test -f /opt/lsy_drone_racing/acados/lib/libacados.so \
    && test -x /opt/lsy_drone_racing/acados/bin/t_renderer

ENV ACADOS_SOURCE_DIR=/opt/lsy_drone_racing/acados
ENV ACADOS_INSTALL_DIR=/opt/lsy_drone_racing/acados
ENV PATH="/opt/lsy_drone_racing/acados/bin:/opt/lsy_drone_racing/acados/interfaces/acados_template:${PATH}"
ENV LD_LIBRARY_PATH="/opt/lsy_drone_racing/acados/lib:${LD_LIBRARY_PATH}"

# Recheck outbound HTTPS after all network-dependent build steps.  Keeping this
# close to the end makes a source-changing rebuild perform a fresh check rather
# than relying only on an old base-layer cache entry.
RUN curl --fail --location --silent --show-error \
    --retry 3 --retry-all-errors --connect-timeout 15 --max-time 60 \
    --output /dev/null \
    --write-out "final docker-build google status=%{http_code} remote=%{remote_ip}\n" \
    https://www.google.com/

# Keep all cleanup in the last build step, after every package manager and
# compiler has finished.  Installed Pixi environments and source trees stay.
USER root
RUN rm -rf \
    /var/lib/apt/lists/* \
    /root/.cache \
    /home/user/.cache \
    /opt/crazyflow/.pytest_cache \
    /opt/crazyflow/.ruff_cache \
    /opt/lsy_drone_racing/.pytest_cache \
    && find /opt/crazyflow /opt/lsy_drone_racing \
        -type d -name __pycache__ -prune -exec rm -rf '{}' +

USER ${USERNAME}
ENV PYTHONPATH=/workspace/crazyflow
WORKDIR /workspace/crazyflow

CMD ["bash"]
