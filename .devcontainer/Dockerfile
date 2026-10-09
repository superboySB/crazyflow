FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS dependencies

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
# GPU test environment. Its Python 3.12 constraint also supports PyTorch 2.8,
# and the Linux lock includes splax[viewer].
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=v0.76.0 bash

WORKDIR /opt/crazyflow
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml pixi.lock README.md LICENSE ./
# Large concurrent HTTP/2 transfers stall behind some build proxies. Fetch the
# exact locked wheels with four HTTP/1.1 range transfers, verify their hashes,
# and check installed versions. Temporary wheels stay in this
# layer only; other package-manager caches are cleaned in the final layer.
RUN python3 - <<'PY'
import concurrent.futures
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

lock = Path("pixi.lock").read_text()
block = re.search(r"^  gpu-tests:\n(.*?)(?=^  \S|\Z)", lock, re.M | re.S).group(1)
urls = re.findall(r"^      - pypi: (https://\S+\.whl)$", block, re.M)
hashes = {
    url: re.search(r"^  sha256: ([0-9a-f]{64})$", metadata, re.M).group(1)
    for url, metadata in re.findall(r"^- pypi: ([^\n]+)\n(.*?)(?=^- |\Z)", lock, re.M | re.S)
    if url in urls
}
assert urls and len(hashes) == len(urls), "Every GPU wheel must have a locked SHA256"
pixi = ["pixi", "install", "--locked", "-e", "gpu-tests", "--skip", "crazyflow"]
skips = [arg for url in urls for arg in
         ("--skip", Path(urlsplit(url).path).name.split("-")[0].replace("_", "-").lower())]
print(f"Install locked Conda packages; fetch {len(urls)} PyPI wheels separately", flush=True)
subprocess.run(pixi + skips, check=True)
curl = ["curl", "--http1.1", "--fail", "--location", "--silent", "--show-error",
        "--retry", "3", "--retry-all-errors", "--connect-timeout", "20", "--max-time", "300"]

def size(url):
    headers = subprocess.check_output(curl + ["--head", url], text=True)
    return int(re.findall(r"^content-length: (\d+)", headers, re.M | re.I)[-1])

def fetch(part):
    url, start, end, path = part
    subprocess.run(curl + ["--range", f"{start}-{end}", "--output", str(path), url], check=True)
    assert path.stat().st_size == end - start + 1, f"Incomplete range: {url}"

with tempfile.TemporaryDirectory(prefix="crazyflow-wheels-") as directory:
    root = Path(directory)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(size, urls))
        parts = [(url, start, min(start + 8 * 1024**2, total) - 1, root / f"{index}-{start}.part")
                 for index, (url, total) in enumerate(zip(urls, sizes))
                 for start in range(0, total, 8 * 1024**2)]
        list(pool.map(fetch, parts))
    wheels = []
    for url in urls:
        path = root / Path(urlsplit(url).path).name
        with path.open("wb") as destination:
            for part_url, _, _, segment in parts:
                if part_url == url:
                    with segment.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                    segment.unlink()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
        assert digest.hexdigest() == hashes[url], f"SHA256 mismatch: {path.name}"
        name, version = path.name.split("-")[:2]
        wheels.append(f"{name}=={version}")
        print(f"Verified locked wheel: {path.name}", flush=True)
    python = "/opt/crazyflow/.pixi/envs/gpu-tests/bin/python"
    # Pixi deliberately reinstalls pip-owned packages, even with matching versions.
    # Verify every installed version directly after materializing the locked hashes.
    subprocess.run([python, "-m", "pip", "install", "--no-index", "--no-deps",
                    "--find-links", str(root), *wheels], check=True)
    check = ('from importlib.metadata import version; import sys; '
             'assert all(version(name) == expected for name, expected in '
             '(spec.split("==") for spec in sys.argv[1:]))')
    subprocess.run([python, "-c", check, *wheels], check=True)
PY

# Keep the environment inside the image.  PYTHONPATH makes a bind-mounted
# checkout at /workspace/crazyflow take precedence over the build-time copy.
ENV PATH="/opt/crazyflow/.pixi/envs/gpu-tests/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/crazyflow/.pixi/envs/gpu-tests/lib/python3.12/site-packages/nvidia/nvshmem/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

FROM dependencies AS research

# Source changes are a cheap layer: they do not invalidate the locked CUDA
# dependency environment above.
COPY --chown=${USERNAME}:${USERNAME} . .
RUN python -m pip install --no-deps --no-build-isolation -e .

# The lock already selects a matching MuJoCo/MJX 3.10 pair. Gymnasium 1.4
# supplies the updated camera API, so no post-install downgrade is needed.
RUN python -c 'import sys, gymnasium, mujoco; from importlib.metadata import version; assert sys.version_info[:2] == (3, 12); assert gymnasium.__version__ == "1.4.0"; assert mujoco.__version__ == version("mujoco-mjx") == "3.10.0"'

# Pin the externally maintained racing course so rebuilds do not silently
# change behavior.  Install its simulation, optional gamepad, and RL packages
# into the same CUDA-enabled Python environment as Crazyflow.  PyTorch's
# official CUDA 12.6 wheel lets both the JAX simulator and PPO policy use GPU.
ARG LSY_DRONE_RACING_REF=065a6ecb7a7621f156bdd0bedc70020e3331a3e9
USER root
RUN mkdir -p /opt/lsy_drone_racing \
    && chown "${USERNAME}:${USERNAME}" /opt/lsy_drone_racing
USER ${USERNAME}
WORKDIR /opt
RUN git clone https://github.com/learnsyslab/lsy_drone_racing.git \
        /opt/lsy_drone_racing \
    && cd /opt/lsy_drone_racing \
    && git checkout --detach "${LSY_DRONE_RACING_REF}" \
    && git apply /opt/crazyflow/.devcontainer/lsy_drone_racing-compat.patch \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0 \
    && python -c 'import torch; assert torch.backends.cuda.is_built(); assert torch.version.cuda == "12.6"' \
    && python -m pip install wandb==0.28.1 \
    && python -m pip install -e ".[gamepad,rl]"

# The default racing controllers use acados.  The upstream setup script pins
# acados v0.5.1 and t_renderer v0.0.34, builds the native libraries, and
# installs acados_template into the active Python environment.
WORKDIR /opt/lsy_drone_racing
RUN PIXI_PROJECT_ROOT=/opt/lsy_drone_racing bash tools/setup_acados.sh \
    && test -f /opt/lsy_drone_racing/acados/lib/libacados.so \
    && test -x /opt/lsy_drone_racing/acados/bin/t_renderer \
    && python -m pip check

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
