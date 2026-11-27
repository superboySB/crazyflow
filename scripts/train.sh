#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
args=()
output=
smoke=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --output) cf_require_value "$@"; output=$(cf_to_host "$2"); shift 2 ;;
    --resume|--warm-start) cf_require_value "$@"; args+=("$1" "$(cf_to_container "$2")"); shift 2 ;;
    --smoke) smoke=true; args+=(--smoke); shift ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ " ${args[*]} " == *" --help "* || " ${args[*]} " == *" -h "* ]]; then
  cf_ensure_image
  cf_container_cpu python -m crazyflow.tof.train --help
  printf '\n%s\n' '启动器另支持 --device auto|gpu|cpu；--output 可省略并自动创建新目录。'
  exit
fi
cf_runtime
output=${output:-$(cf_new_directory tof-ppo)}
mkdir -p "$output"
CF_GUARD_GPU=1
CF_MEMORY_LOG="$output/gpu-memory.json"
cf_container python -m crazyflow.tof.train --output "$(cf_to_container "$output")" "${args[@]}"
if ! "$smoke" && [[ " ${args[*]} " != *" --benchmark "* ]]; then
  cf_record_latest latest-policy "$output"
fi
printf '训练结果：%s\n' "$output"
