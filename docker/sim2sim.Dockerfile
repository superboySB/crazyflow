# syntax=docker/dockerfile:1.4

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=4.5.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS simulation

ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/isaaclab
ARG DOCKER_USER_HOME=/root
ARG ISAACLAB_REPO=https://github.com/Jason-xy/IsaacLab.git
ARG ISAACLAB_REF=083ca3cd6039410eabce69daad869a1426e8635e
ARG RSL_RL_REPO=https://github.com/Jason-xy/rsl_rl.git
ARG RSL_RL_REF=750e84566d91877a8bbabc7971578be24429bca8

ENV ISAACSIM_VERSION=${ISAACSIM_VERSION} \
    ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH} \
    ISAACLAB_PATH=${ISAACLAB_PATH} \
    DOCKER_USER_HOME=${DOCKER_USER_HOME} \
    ISAACSIM_PATH=${ISAACLAB_PATH}/_isaac_sim \
    OMNI_KIT_ALLOW_ROOT=1 \
    ENABLE_CAMERAS=1 \
    PYTHONPATH=/workspace/crazyflow/sim2sim/reference:/workspace/isaaclab \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    tmux \
    vim \
    gedit && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN ${ISAACSIM_ROOT_PATH}/python.sh - <<'PY'
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import urlretrieve

for name, destination in (
    ("ISAACLAB", Path(os.environ["ISAACLAB_PATH"])),
    ("RSL_RL", Path(os.environ["ISAACLAB_PATH"]) / "rsl_rl"),
):
    repository = urlparse(os.environ[f"{name}_REPO"].rstrip("/").removesuffix(".git"))
    if repository.scheme != "https" or repository.netloc != "github.com":
        raise ValueError(f"{name}_REPO must be an HTTPS GitHub repository URL")
    ref = quote(os.environ[f"{name}_REF"], safe="")
    url = f"https://codeload.github.com{repository.path}/tar.gz/{ref}"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.tar.gz"
        urlretrieve(url, archive)
        subprocess.run(
            ["tar", "-xzf", str(archive), "--strip-components=1", "-C", str(destination)],
            check=True,
        )
PY

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

# Torch install in isaaclab.sh --install resolves sympy from PyPI; preinstall to avoid index mismatch.
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --upgrade pip && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install "sympy>=1.13.3"

RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install none

# flatdict's legacy build needs pkg_resources from the bundled setuptools.
# Install the core explicitly because the fork's extension loop can hide errors.
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-build-isolation flatdict==4.0.1 && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e ${ISAACLAB_PATH}/source/isaaclab

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    pynput \
    debugpy \
    rtree \
    "numpy<2" \
    tqdm \
    tensorboard \
    opencv-python \
    open3d \
    rerun-sdk==0.22.1 && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y \
    rsl-rl-lib

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e ${ISAACLAB_PATH}/rsl_rl

# Isaac Sim extensions still rely on NumPy 1.x ABI.
RUN ${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip install --upgrade --no-cache-dir "numpy<2"

RUN ${ISAACSIM_ROOT_PATH}/python.sh - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
import torch

for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "rsl_rl"):
    spec = find_spec(name)
    assert spec is not None, f"Missing required package: {name}"
    print(f"{name}: {spec.origin}")
assert torch.__version__ == "2.7.0+cu128", torch.__version__
assert version("numpy").split(".")[0] == "1", version("numpy")
print(f"torch={torch.__version__}, numpy={version('numpy')}, rsl-rl-lib={version('rsl-rl-lib')}")
PY

# TensorBoard discovers Open3D's plugin, whose wheel requires libusb.
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# The task, dynamics, USD assets and baseline checkpoint are shipped by this repo.
COPY sim2sim/ /workspace/crazyflow/sim2sim/
COPY models/ /workspace/crazyflow/models/

WORKDIR /workspace/crazyflow
ENTRYPOINT ["/bin/bash"]
