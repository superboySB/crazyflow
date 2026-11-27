#!/usr/bin/env bash
# Docker/JAX device selection shared by every launcher. Requires Bash and Docker only.
# Auto probes real JAX GPU computation, then falls back to CPU. Forced GPU errors are fatal.
# The host driver and NVIDIA Container Toolkit must already be installed to expose a GPU.
if [[ ${CF_COMMON_LOADED:-0} == 1 ]]; then return 0; fi
CF_COMMON_LOADED=1
CF_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CF_OUTPUTS=$(realpath -m -- "$CF_ROOT/outputs")
CF_IMAGE=${CRAZYFLOW_IMAGE:-crazyflow:dani-crazyrl}
CF_DEVICE_REQUEST=${CF_DEVICE_REQUEST:-auto}

cf_die() { printf '错误：%s\n' "$*" >&2; exit 1; }
cf_require_value() { [[ $# -gt 1 && -n $2 && $2 != --* ]] || cf_die "$1 需要一个值。"; }
cf_docker() {
  command -v docker >/dev/null || cf_die '请先安装并启动 Docker Engine，然后重新运行 ./crazyrl setup。'
  docker info >/dev/null 2>&1 || cf_die 'Docker 不可用；请启动 Docker 并检查当前用户的访问权限。'
  mkdir -p "$CF_OUTPUTS"
}
cf_build() {
  cf_docker
  printf '构建 %s（Python、JAX/CUDA 和 MuJoCo 都在镜像中）…\n' "$CF_IMAGE"
  docker build -f "$CF_ROOT/docker/crazyflow.Dockerfile" -t "$CF_IMAGE" "$CF_ROOT"
}
cf_ensure_image() {
  cf_docker
  if ! docker image inspect "$CF_IMAGE" >/dev/null 2>&1; then cf_build; fi
}
cf_select_device() {
  case "$CF_DEVICE_REQUEST" in auto|cpu|gpu) ;; *) cf_die '--device 只能是 auto、cpu 或 gpu。';; esac
  local probe_log
  probe_log=$(mktemp)
  if [[ $CF_DEVICE_REQUEST != cpu ]]; then
    printf '检测容器中的 GPU 与 JAX 计算能力…\n'
    if docker run --rm --init --gpus all --shm-size=2g \
        --user "$(id -u):$(id -g)" -e JAX_PLATFORMS=cuda "$CF_IMAGE" \
        python -c 'import jax; import jax.numpy as jnp; assert jax.devices("gpu"); x=jnp.ones((8,8)); (x@x).block_until_ready(); print(jax.devices("gpu"))' \
        >"$probe_log" 2>&1; then
      CF_DEVICE=gpu
      printf '使用 GPU：%s\n' "$(tail -n 1 "$probe_log")"
      rm -f "$probe_log"
      return
    fi
    if [[ $CF_DEVICE_REQUEST == gpu ]]; then
      tail -n 8 "$probe_log" >&2
      rm -f "$probe_log"
      cf_die '指定的 GPU 不可用；请检查宿主驱动与 NVIDIA Container Toolkit，或使用 --device cpu。'
    fi
    printf 'GPU 不可用，自动使用 CPU（训练并行数默认降为 32）。\n'
    # Retain the actual probe error for troubleshooting without a long startup traceback.
    cp "$probe_log" "$CF_OUTPUTS/.gpu-probe.log"
  fi
  if ! docker run --rm --init --user "$(id -u):$(id -g)" -e JAX_PLATFORMS=cpu \
      "$CF_IMAGE" python -c 'import jax; import jax.numpy as jnp; (jnp.ones(8)+1).block_until_ready(); print(jax.devices())' \
      >"$probe_log" 2>&1; then
    cat "$probe_log" >&2
    rm -f "$probe_log"
    cf_die '容器 CPU/JAX 检查失败。请重新运行 ./crazyrl setup。'
  fi
  CF_DEVICE=cpu
  printf '使用 CPU。\n'
  rm -f "$probe_log"
}
cf_runtime() {
  cf_ensure_image
  cf_select_device
}
cf_run_args() {
  CF_RUN_ARGS=(--rm --init --user "$(id -u):$(id -g)" --shm-size=2g
    -v "$CF_OUTPUTS:/outputs" --workdir /opt/crazyflow
    -e "JAX_COMPILATION_CACHE_DIR=/outputs/.cache/jax-$(id -u)"
    -e JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
    -v "$CF_ROOT:/opt/crazyflow:ro")
}
cf_container() {
  cf_run_args
  if [[ $CF_DEVICE == gpu ]]; then
    CF_RUN_ARGS+=(--gpus all -e JAX_PLATFORMS=cuda -e MUJOCO_GL=egl)
  else
    CF_RUN_ARGS+=(-e JAX_PLATFORMS=cpu -e MUJOCO_GL=osmesa)
  fi
  if [[ ${CF_GUARD_GPU:-0} == 1 && $CF_DEVICE == gpu ]]; then
    local container_name="crazyflow-tof-${BASHPID}-${RANDOM}"
    python3 "$CF_ROOT/scripts/guard_gpu.py" --name "$container_name" \
      --output "$CF_MEMORY_LOG" -- docker run --name "$container_name" \
      "${CF_RUN_ARGS[@]}" "$CF_IMAGE" "$@"
  else
    docker run "${CF_RUN_ARGS[@]}" "$CF_IMAGE" "$@"
  fi
}
cf_container_cpu() {
  cf_run_args
  docker run "${CF_RUN_ARGS[@]}" -e JAX_PLATFORMS=cpu -e MUJOCO_GL=osmesa "$CF_IMAGE" "$@"
}
cf_container_gui() {
  [[ -n ${DISPLAY:-} ]] || cf_die '没有 DISPLAY；在 Linux/X11 桌面运行，或使用 ./crazyrl play --headless。'
  cf_run_args
  CF_RUN_ARGS+=(--hostname "$(hostname)" -e "DISPLAY=$DISPLAY" -e MUJOCO_GL=glfw
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
  if [[ $CF_DEVICE == gpu ]]; then
    CF_RUN_ARGS+=(--gpus all -e JAX_PLATFORMS=cuda
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display)
  else
    CF_RUN_ARGS+=(-e JAX_PLATFORMS=cpu)
  fi
  local xauthority_file=${XAUTHORITY:-$HOME/.Xauthority}
  if [[ -f $xauthority_file ]]; then
    CF_RUN_ARGS+=(-e XAUTHORITY=/tmp/host.xauthority -v "$xauthority_file:/tmp/host.xauthority:ro")
  fi
  docker run "${CF_RUN_ARGS[@]}" "$CF_IMAGE" "$@"
}
# Paths are repository-relative (independent of the caller's cwd). Resolve symlinks
# before enforcing the single outputs bind mount; never invent other host mounts.
cf_to_host() {
  local path=${1:?path required}
  case "$path" in
    /outputs) path=$CF_OUTPUTS ;;
    /outputs/*) path="$CF_OUTPUTS/${path#/outputs/}" ;;
    /*) ;;
    *) path="$CF_ROOT/$path" ;;
  esac
  path=$(realpath -m -- "$path")
  case "$path" in "$CF_OUTPUTS"|"$CF_OUTPUTS/"*) printf '%s\n' "$path";;
    *) cf_die "文件必须放在项目 outputs/ 内：$1";; esac
}
cf_to_container() {
  local path
  path=$(cf_to_host "$1") || return
  if [[ $path == "$CF_OUTPUTS" ]]; then printf '/outputs\n';
  else printf '/outputs/%s\n' "${path#"$CF_OUTPUTS/"}"; fi
}
cf_new_directory() {
  mkdir -p "$CF_OUTPUTS"
  mktemp -d "$CF_OUTPUTS/${1:-run}-$(date +%Y%m%d-%H%M%S)-XXXXXX"
}
cf_checkpoint_to_container() {
  local path=$1
  case "$path" in
    /outputs|/outputs/*) cf_to_container "$path"; return ;;
    /opt/crazyflow/models/*) path="$CF_ROOT/${path#/opt/crazyflow/}" ;;
    /*) ;;
    *) path="$CF_ROOT/$path" ;;
  esac
  path=$(realpath -m -- "$path")
  case "$path" in
    "$CF_ROOT/models/"*)
      [[ -s $path ]] || cf_die "模型不存在：$path"
      printf '/opt/crazyflow/%s\n' "${path#"$CF_ROOT/"}" ;;
    *) cf_to_container "$path" ;;
  esac
}
cf_bundled_policy() {
  cf_checkpoint_to_container "$CF_ROOT/models/crazyflow/policy.npz"
}
cf_default_checkpoint() {
  if [[ -s $CF_OUTPUTS/.latest-policy ]]; then
    cf_to_container "$(cf_latest_policy)/policy.npz"
  else
    cf_bundled_policy
  fi
}
cf_record_latest() {
  local key=${1#.} path pointer_tmp
  path=$(cf_to_host "$2") || return
  [[ -d $path && $path != "$CF_OUTPUTS" ]] || cf_die '不能记录无效的结果目录。'
  case "$key" in latest-policy|latest-evaluation) ;; *) cf_die '无效的最近结果指针。';; esac
  pointer_tmp=$(mktemp "$CF_OUTPUTS/.$key.XXXXXX")
  printf '%s\n' "${path#"$CF_OUTPUTS/"}" >"$pointer_tmp"
  mv -f -- "$pointer_tmp" "$CF_OUTPUTS/.$key"
}
cf_latest_directory() {
  local key=$1 value path
  [[ -s $CF_OUTPUTS/.$key ]] || cf_die "没有最近成功结果（$key）；请先训练/评测，或显式指定文件。"
  IFS= read -r value <"$CF_OUTPUTS/.$key"
  [[ -n $value && $value != /* ]] || cf_die '结果指针无效，请显式指定文件。'
  path=$(cf_to_host "/outputs/$value") || return
  [[ -d $path ]] || cf_die "结果目录不存在：$path"
  printf '%s\n' "$path"
}
cf_latest_policy() {
  local path
  path=$(cf_latest_directory latest-policy) || return
  [[ -s $path/policy.npz ]] || cf_die "缺少策略文件：$path/policy.npz"
  printf '%s\n' "$path"
}
cf_latest_evaluation() {
  local path
  path=$(cf_latest_directory latest-evaluation) || return
  [[ -s $path/summary.json ]] || cf_die "缺少评测文件：$path/summary.json"
  printf '%s\n' "$path"
}
