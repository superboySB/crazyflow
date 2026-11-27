#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
args=()
headless=false
checkpoint=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --checkpoint) cf_require_value "$@"; checkpoint=$(cf_checkpoint_to_container "$2"); shift 2 ;;
    --headless) headless=true; args+=(--headless); shift ;;
    --output|--record|--trajectory|--evaluation|--screenshot|--gui-report)
      cf_require_value "$@"; flag=$1; path=$2
      if [[ "$flag" == --evaluation ]]; then flag=--trajectory; path="${path%/}/trajectories.npz"; fi
      args+=("$flag" "$(cf_to_container "$path")"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ " ${args[*]} " == *" --help "* || " ${args[*]} " == *" -h "* ]]; then
  cf_ensure_image
  cf_container_cpu python -m crazyflow.tof.play --help
  printf '\n%s\n' '使用 --checkpoint models/crazyflow/policy.npz 或 outputs/<run>/policy.npz 选择模型；--device auto|gpu|cpu 选择设备。 --evaluation DIR 回放该目录的 trajectories.npz。'
  exit
fi
if [[ -n "$checkpoint" ]]; then
  args+=(--checkpoint "$checkpoint")
elif [[ " ${args[*]} " != *" --trajectory "* ]]; then
  args+=(--checkpoint "$(cf_default_checkpoint)")
fi
cf_runtime
args+=(--device "$CF_DEVICE")
if "$headless"; then
  cf_container python -m crazyflow.tof.play "${args[@]}"
else
  cf_container_gui python -m crazyflow.tof.play "${args[@]}"
fi
