#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
args=()
checkpoint=
example=false
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --example) example=true; shift ;;
    --checkpoint) cf_require_value "$@"; checkpoint=$(cf_checkpoint_to_container "$2"); shift 2 ;;
    --output) cf_require_value "$@"; output=$(cf_to_host "$2"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ " ${args[*]} " == *" --help "* || " ${args[*]} " == *" -h "* ]]; then
  cf_ensure_image
  cf_container_cpu python -m crazyflow.tof.evaluate --help
  printf '\n%s\n' '启动器另支持 --device auto|gpu|cpu、--example；省略 --checkpoint 时读取最近正式模型，没有正式运行则读取内置最佳模型。'
  exit
fi
if "$example"; then
  [[ -z "$checkpoint" ]] || cf_die '--example 与 --checkpoint 请选择其一。'
  checkpoint=$(cf_example_policy)
fi
checkpoint=${checkpoint:-$(cf_default_checkpoint)}
output=${output:-$(cf_new_directory tof-eval)}
cf_runtime
cf_container python -m crazyflow.tof.evaluate --checkpoint "$checkpoint" \
  --output "$(cf_to_container "$output")" --device "$CF_DEVICE" "${args[@]}"
cf_record_latest latest-evaluation "$output"
printf '评测结果：%s\n' "$output"
