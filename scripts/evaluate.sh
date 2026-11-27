#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
args=()
checkpoint=
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --checkpoint) cf_require_value "$@"; checkpoint=$(cf_checkpoint_to_container "$2"); shift 2 ;;
    --output) cf_require_value "$@"; output=$(cf_to_host "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ " ${args[*]} " == *" --help "* || " ${args[*]} " == *" -h "* ]]; then
  cf_ensure_image
  cf_container_cpu python -m crazyflow.tof.evaluate --help
  printf '\n%s\n' '使用 --checkpoint models/crazyflow/policy.npz 或 outputs/<run>/policy.npz 选择模型；--device auto|gpu|cpu 选择设备。'
  exit
fi
checkpoint=${checkpoint:-$(cf_default_checkpoint)}
output=${output:-$(cf_new_directory tof-eval)}
cf_runtime
cf_container python -m crazyflow.tof.evaluate --checkpoint "$checkpoint" \
  --output "$(cf_to_container "$output")" --device "$CF_DEVICE" "${args[@]}"
cf_record_latest latest-evaluation "$output"
printf '评测结果：%s\n' "$output"
