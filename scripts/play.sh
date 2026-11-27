#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
args=()
headless=false
checkpoint=
example=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --example) example=true; shift ;;
    --checkpoint) cf_require_value "$@"; checkpoint=$(cf_to_container "$2"); shift 2 ;;
    --headless) headless=true; args+=(--headless); shift ;;
    --output|--record|--trajectory|--evaluation)
      cf_require_value "$@"; flag=$1; path=$2
      if [[ "$flag" == --evaluation ]]; then flag=--trajectory; path="${path%/}/trajectories.npz"; fi
      args+=("$flag" "$(cf_to_container "$path")"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ " ${args[*]} " == *" --help "* || " ${args[*]} " == *" -h "* ]]; then
  cf_ensure_image
  cf_container_cpu python -m crazyflow.tof.play --help
  printf '\n%s\n' '启动器另支持 --device auto|gpu|cpu、--example；省略 --checkpoint 时读取最近正式模型。 --evaluation DIR 回放该目录的 trajectories.npz。'
  exit
fi
if "$example"; then
  [[ -z "$checkpoint" ]] || cf_die '--example 与 --checkpoint 请选择其一。'
  checkpoint=$(cf_to_container "$(cf_stage_example_policy)")
fi
if [[ -n "$checkpoint" ]]; then
  args+=(--checkpoint "$checkpoint")
elif [[ " ${args[*]} " != *" --trajectory "* ]]; then
  args+=(--checkpoint "$(cf_to_container "$(cf_latest_policy)/policy.npz")")
fi
cf_runtime
args+=(--device "$CF_DEVICE")
if "$headless"; then
  cf_container python -m crazyflow.tof.play "${args[@]}"
else
  cf_container_gui python -m crazyflow.tof.play "${args[@]}"
fi
