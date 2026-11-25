#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

usage() {
  cat <<'EOF'
用法：./dani evaluate [选项]

默认读取最近训练的 policy.npz，在 native MuJoCo 中评测三组各 256 次，生成视频。
每次使用新目录，成功后更新最近评测指针；不会覆盖已有评测。

  --example           复制随项目提供的策略到新目录并评测，不修改最近训练
  --checkpoint PATH   outputs 内的策略路径，与 --example 二选一
  --output DIR        新评测目录；该目录必须尚不存在
  --episodes N        每组评测次数（默认 256）
  --seed N            评测随机种子（默认 20261125）
  --device auto|cpu|gpu  自动选 GPU 或 CPU（默认 auto）
  --no-video          不生成视频
  --plant mujoco|crazyflow  飞行后端（默认 mujoco）
  --gui               评测完成后打开 Linux/X11 GUI
  -h, --help          显示帮助，不启动 Docker

参数必须使用完整名称；路径统一限制在项目 outputs/ 内。
EOF
}

checkpoint=
checkpoint_given=false
example=false
output=
episodes=256
seed=20261125
video=true
gui=false
plant=mujoco

need_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "参数 $1 缺少值。"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --example) example=true; shift ;;
    --checkpoint) need_value "$@"; checkpoint=$2; checkpoint_given=true; shift 2 ;;
    --checkpoint=*) checkpoint=${1#*=}; checkpoint_given=true; shift ;;
    --output) need_value "$@"; output=$2; shift 2 ;;
    --output=*) output=${1#*=}; shift ;;
    --episodes) need_value "$@"; episodes=$2; shift 2 ;;
    --episodes=*) episodes=${1#*=}; shift ;;
    --seed) need_value "$@"; seed=$2; shift 2 ;;
    --seed=*) seed=${1#*=}; shift ;;
    --device) need_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --device=*) CF_DEVICE_REQUEST=${1#*=}; shift ;;
    --plant) need_value "$@"; plant=$2; shift 2 ;;
    --plant=*) plant=${1#*=}; shift ;;
    --no-video) video=false; shift ;;
    --video) video=true; shift ;;
    --gui) gui=true; shift ;;
    *) cf_die "不支持参数 $1；请使用完整参数名，参见 ./dani evaluate --help。" ;;
  esac
done

[[ "$example" != true || "$checkpoint_given" != true ]] || \
  cf_die '--example 与 --checkpoint 不能同时使用。'
[[ "$episodes" =~ ^[1-9][0-9]*$ ]] || cf_die '--episodes 必须是正整数。'
[[ "$seed" =~ ^-?[0-9]+$ ]] || cf_die '--seed 必须是整数。'
[[ "$plant" == mujoco || "$plant" == crazyflow ]] || \
  cf_die '--plant 只能是 mujoco 或 crazyflow。'
if [[ "$example" == true ]]; then
  checkpoint=$(cf_stage_example_policy)
fi
if [[ -z "$checkpoint" ]]; then
  policy_dir=$(cf_latest_policy)
  checkpoint="$policy_dir/policy.npz"
fi
container_checkpoint=$(cf_to_container "$checkpoint")
host_checkpoint="$CF_OUTPUTS${container_checkpoint#/outputs}"
[[ -f "$host_checkpoint" && -r "$host_checkpoint" ]] || \
  cf_die "找不到或无法读取策略文件：$host_checkpoint。请先运行 ./dani train。"
policy_dir=$(dirname -- "$host_checkpoint")
if [[ -z "$output" ]]; then
  output="$policy_dir/evaluation-$(date +%Y%m%d-%H%M%S)-$$"
fi
container_output=$(cf_to_container "$output")
host_output="$CF_OUTPUTS${container_output#/outputs}"
[[ ! -e "$host_output" && ! -L "$host_output" ]] || \
  cf_die "评测目录已存在，拒绝覆盖：$host_output。请换一个 --output。"

cf_runtime
mkdir -p -- "$(dirname -- "$host_output")"
mkdir -- "$host_output" || cf_die "无法创建新的评测目录：$host_output。"
args=(--checkpoint "$container_checkpoint" --plant "$plant" --episodes "$episodes"
  --seed "$seed" --device "$CF_DEVICE" --output "$container_output")
if [[ "$video" == true ]]; then
  args+=(--video)
fi
if [[ "$gui" == true ]]; then
  args+=(--gui)
fi
printf '开始评测，输出目录：%s\n' "$host_output"
runner=cf_container
if [[ "$gui" == true ]]; then
  runner=cf_container_gui
fi
if "$runner" python -m crazyflow.balloon.evaluate "${args[@]}"; then
  for file in metrics.json nominal.npz noisy.npz perturbed.npz; do
    [[ -s "$host_output/$file" ]] || \
      cf_die "评测未生成完整结果，未更新最近评测指针：缺少 $host_output/$file。"
  done
  cf_record_latest .latest-evaluation "$host_output"
  printf '评测完成。运行 ./dani play 查看最近结果。\n'
else
  status=$?
  printf '评测失败（退出码 %s），保留本次目录供检查：%s\n' "$status" "$host_output" >&2
  exit "$status"
fi
