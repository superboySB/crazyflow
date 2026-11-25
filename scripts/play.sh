#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

usage() {
  cat <<'EOF'
用法：./dani play [选项]

默认打开最近一次评测的原生 GUI，显示世界视图、相机画面和策略遥测。

  --example           复制随项目提供的策略到新目录并生成飞行
  --evaluation DIR    outputs 内的评测目录，或 /outputs/... 目录
  --checkpoint PATH   从指定策略生成新轨迹；与 --evaluation/--example 互斥
  --plant mujoco|crazyflow  --checkpoint 的飞行后端（默认 mujoco）
  --seed N            --checkpoint 的随机种子（默认 20261125）
  --device auto|cpu|gpu  自动选 GPU 或 CPU（默认 auto）
  --headless          用 Xvfb 运行；无 --record 时默认 10 帧自测并截图
  --record PATH       保存 outputs 内的 GIF；默认完整轨迹加末尾 1 秒
  --fps N             GIF 帧率，5..60（默认 20）
  --suite NAME        nominal/noisy/perturbed（默认 nominal）
  --episode N         选择第 N 条轨迹，从 0 开始（默认 0）
  --frames N          自动关闭帧数；headless 自测至少需要 10 帧
  --screenshot PATH   outputs 内的截图路径（headless 默认生成唯一路径）
  --self-test         运行至少 10 帧交互自测；不能与 --record 同用
  -h, --help          显示帮助，不启动 Docker

GUI 使用 Linux/X11 的 DISPLAY 与现有 Xauthority，不修改 xhost 权限。
按 Space 暂停、R 重置、E 换样本、N 换种子、1/2/3 换场景、T 切换遥测/标定。
参数必须使用完整名称；已保存评测始终使用其记录的飞行后端。
EOF
}

evaluation=
evaluation_given=false
checkpoint=
checkpoint_given=false
example=false
screenshot=
record=
fps=20
suite=nominal
episode=0
frames=
headless=false
self_test=false
plant=mujoco
seed=20261125

need_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "参数 $1 缺少值。"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --example) example=true; shift ;;
    --evaluation) need_value "$@"; evaluation=$2; evaluation_given=true; shift 2 ;;
    --evaluation=*) evaluation=${1#*=}; evaluation_given=true; shift ;;
    --checkpoint) need_value "$@"; checkpoint=$2; checkpoint_given=true; shift 2 ;;
    --checkpoint=*) checkpoint=${1#*=}; checkpoint_given=true; shift ;;
    --device) need_value "$@"; CF_DEVICE_REQUEST=$2; shift 2 ;;
    --device=*) CF_DEVICE_REQUEST=${1#*=}; shift ;;
    --plant) need_value "$@"; plant=$2; shift 2 ;;
    --plant=*) plant=${1#*=}; shift ;;
    --seed) need_value "$@"; seed=$2; shift 2 ;;
    --seed=*) seed=${1#*=}; shift ;;
    --frames) need_value "$@"; frames=$2; shift 2 ;;
    --frames=*) frames=${1#*=}; shift ;;
    --screenshot) need_value "$@"; screenshot=$2; shift 2 ;;
    --screenshot=*) screenshot=${1#*=}; shift ;;
    --record) need_value "$@"; record=$2; shift 2 ;;
    --record=*) record=${1#*=}; [[ -n "$record" ]] || cf_die '--record 需要 GIF 路径。'; shift ;;
    --fps) need_value "$@"; fps=$2; shift 2 ;;
    --fps=*) fps=${1#*=}; shift ;;
    --suite) need_value "$@"; suite=$2; shift 2 ;;
    --suite=*) suite=${1#*=}; shift ;;
    --episode) need_value "$@"; episode=$2; shift 2 ;;
    --episode=*) episode=${1#*=}; shift ;;
    --headless) headless=true; shift ;;
    --self-test) self_test=true; shift ;;
    *) cf_die "不支持参数 $1；请使用完整参数名，参见 ./dani play --help。" ;;
  esac
done

[[ "$checkpoint_given" != true || "$evaluation_given" != true ]] || \
  cf_die '--checkpoint 与 --evaluation 不能同时使用。'
if [[ "$example" == true && ( "$checkpoint_given" == true || "$evaluation_given" == true ) ]]; then
  cf_die '--example 与 --checkpoint/--evaluation 不能同时使用。'
fi
[[ "$seed" =~ ^-?[0-9]+$ ]] || cf_die '--seed 必须是整数。'
[[ "$plant" == mujoco || "$plant" == crazyflow ]] || \
  cf_die '--plant 只能是 mujoco 或 crazyflow。'
[[ "$fps" =~ ^[0-9]{1,2}$ ]] && (( 10#$fps >= 5 && 10#$fps <= 60 )) || \
  cf_die '--fps 必须是 5..60 的整数。'
[[ "$episode" =~ ^[0-9]+$ ]] || cf_die '--episode 必须是非负整数。'
case "$suite" in nominal|noisy|perturbed) ;; *) cf_die '--suite 只能是 nominal、noisy 或 perturbed。';; esac
container_record=
if [[ -n "$record" ]]; then
  [[ ${record,,} == *.gif ]] || cf_die '--record 必须使用 .gif 后缀。'
  [[ "$self_test" != true ]] || cf_die '--record 不能与 --self-test 同时使用。'
  container_record=$(cf_to_container "$record")
fi
if [[ -n "$frames" ]]; then
  [[ "$frames" =~ ^[0-9]+$ ]] || cf_die '--frames 必须是非负整数。'
fi
if [[ "$headless" == true && -z "$record" ]]; then
  frames=${frames:-10}
  self_test=true
fi
if [[ "$self_test" == true ]]; then
  [[ "$frames" =~ ^[1-9][0-9]*$ && ${#frames} -ge 2 ]] || \
    cf_die '交互自测的 --frames 必须是至少 10 的整数。'
fi

args=(--fps "$fps" --suite "$suite" --episode "$episode")
if [[ -n "$container_record" ]]; then
  args+=(--record "$container_record")
fi
if [[ "$example" == true ]]; then
  checkpoint=$(cf_stage_example_policy)
fi
if [[ -n "$checkpoint" ]]; then
  container_checkpoint=$(cf_to_container "$checkpoint")
  host_checkpoint="$CF_OUTPUTS${container_checkpoint#/outputs}"
  [[ -f "$host_checkpoint" && -r "$host_checkpoint" ]] || \
    cf_die "找不到或无法读取策略文件：$host_checkpoint。请先运行 ./dani train。"
  args+=(--checkpoint "$container_checkpoint" --plant "$plant" --seed "$seed")
  screenshot_parent=$(dirname -- "$host_checkpoint")
else
  evaluation=${evaluation:-$(cf_latest_evaluation)}
  container_evaluation=$(cf_to_container "$evaluation")
  host_evaluation="$CF_OUTPUTS${container_evaluation#/outputs}"
  for file in metrics.json nominal.npz noisy.npz perturbed.npz; do
    [[ -f "$host_evaluation/$file" && -r "$host_evaluation/$file" ]] || \
      cf_die "评测目录不完整，缺少或无法读取：$host_evaluation/$file。请先运行 ./dani evaluate。"
  done
  args+=(--evaluation "$container_evaluation")
  screenshot_parent=$host_evaluation
fi
if [[ "$headless" == true && -z "$record" && -z "$screenshot" ]]; then
  screenshot="$screenshot_parent/gui-headless-$(date +%Y%m%d-%H%M%S)-$$.png"
fi
if [[ -n "$screenshot" ]]; then
  [[ -n "$frames" && ! "$frames" =~ ^0+$ ]] || cf_die '--screenshot 需要正整数 --frames。'
  args+=(--screenshot "$(cf_to_container "$screenshot")")
fi
if [[ -n "$frames" ]]; then
  args+=(--frames "$frames")
fi
if [[ "$self_test" == true ]]; then
  args+=(--self-test)
fi
cf_runtime
args+=(--device "$CF_DEVICE")
if [[ "$headless" == true ]]; then
  if [[ -n "$record" ]]; then
    printf '录制无桌面 GUI：%s\n' "$record"
  else
    printf '运行无桌面 GUI 自测，截图：%s\n' "$screenshot"
  fi
  cf_container env MUJOCO_GL=glfw xvfb-run -a -s '-screen 0 1600x1100x24' \
    python -m crazyflow.balloon.play "${args[@]}"
else
  cf_container_gui python -m crazyflow.balloon.play "${args[@]}"
fi
