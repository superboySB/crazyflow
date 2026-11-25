#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
train_args=()
output_dir=
smoke=false
settings_source=$CF_ROOT/config/training.yaml
camera_source=$CF_ROOT/config/camera.json
model_source=$CF_ROOT/config/vehicle.toml
controller_source=$CF_ROOT/config/controller.toml
source_path() {
  case "$1" in
    /outputs|/outputs/*) cf_to_host "$1";;
    /*) printf '%s\n' "$1";;
    *) printf '%s/%s\n' "$CF_ROOT" "$1";;
  esac
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) cf_require_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --output) cf_require_value "$@"; output_dir=$(cf_to_host "$2"); shift 2 ;;
    --settings|--camera|--config|--model-params|--controller-params)
      cf_require_value "$@"
      input_path=$(source_path "$2")
      [[ -f $input_path ]] || cf_die "配置文件不存在：$input_path"
      case "$1" in
        --settings) settings_source=$input_path;;
        --camera|--config) camera_source=$input_path;;
        --model-params) model_source=$input_path;;
        --controller-params) controller_source=$input_path;;
      esac
      shift 2 ;;
    --iterations|--batch-size|--steps|--seed|--lr)
      cf_require_value "$@"; train_args+=("$1" "$2"); shift 2 ;;
    --smoke) smoke=true; train_args+=("$1"); shift ;;
    --nominal) train_args+=("$1"); shift ;;
    -h|--help)
      cat <<'HELP'
用法：./dani train [选项]
默认读取 config/ 下的四份配置，并保存本次副本，不需要终端变量。
  --settings PATH           训练/任务/噪声/目标 YAML（config/training.yaml）
  --camera PATH             相机内外参 JSON（config/camera.json）
  --model-params PATH       完整机体物理 TOML（config/vehicle.toml）
  --controller-params PATH  姿态/分配控制器 TOML（config/controller.toml）
  --device auto|cpu|gpu     自动优先 GPU，失败回退 CPU；默认 auto
  --smoke                   4 次小更新，仅检查流程，不替换正式结果
  --output outputs/my-run   新结果目录；默认自动创建唯一目录
命令行 --iterations、--batch-size、--steps、--seed、--lr 可覆盖 YAML。
--nominal 关闭噪声，仅调试。未手动覆盖机体/控制器时使用官方 cf21B_500 默认值。
实测辨识结果示例：./dani train --model-params outputs/measurements/physical.toml
HELP
      exit 0 ;;
    *) cf_die "未知参数：$1。请运行 ./dani train --help。" ;;
  esac
done
for path in "$settings_source" "$camera_source" "$model_source" "$controller_source"; do
  [[ -f $path ]] || cf_die "缺少配置：$path"
done
if [[ -n $output_dir && ( -e $output_dir || -L $output_dir ) ]]; then
  cf_die "结果目录已存在，请更换 --output：$output_dir"
fi
cf_runtime
if [[ -z $output_dir ]]; then
  output_dir=$(cf_new_directory train)
else
  mkdir -p -- "$(dirname -- "$output_dir")"
  mkdir -- "$output_dir" || cf_die "结果目录已存在或无法创建：$output_dir"
fi
mkdir "$output_dir/config"
cp -- "$settings_source" "$output_dir/config/training.yaml"
cp -- "$camera_source" "$output_dir/config/camera.json"
cp -- "$model_source" "$output_dir/config/vehicle.toml"
cp -- "$controller_source" "$output_dir/config/controller.toml"
container_config=$(cf_to_container "$output_dir/config")
printf '训练结果：%s\n本次配置：%s/config/\n' "$output_dir" "$output_dir"
cf_container python -m crazyflow.balloon.train --device "$CF_DEVICE" \
  --settings "$container_config/training.yaml" --config "$container_config/camera.json" \
  --model-params "$container_config/vehicle.toml" \
  --controller-params "$container_config/controller.toml" \
  --output "$(cf_to_container "$output_dir")" "${train_args[@]}"
[[ -s $output_dir/policy.npz ]] || cf_die '训练未生成策略；保留已有的最近成功结果。'
if [[ $smoke == true ]]; then
  printf '流程检查完成；未替换最近一次正式训练。\n'
else
  cf_record_latest latest-policy "$output_dir"
  printf '训练完成。下一步：./dani evaluate\n'
fi
