#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    -h|--help)
      printf '用法：./crazyrl setup [--device auto|cpu|gpu]\n构建统一的 CPU/GPU 镜像，并在容器内检查实际 JAX 计算。默认优先 GPU。\n宿主需 Docker；GPU 还需已安装的驱动与 NVIDIA Container Toolkit。\n'
      exit 0 ;;
    *) cf_die "未知参数：$1" ;;
  esac
done
cf_build
cf_select_device
printf '\n配置完成。下一步：./crazyrl train\n'
