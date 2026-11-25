#!/usr/bin/env bash
# File-based physical calibration; never connects to a flight controller.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/common.sh
source "$script_dir/common.sh"

usage() {
  cat <<'HELP'
用法：./dani identify <init|flight|rotor|static|check|inertia|drag-init|drag|drag-check> [选项]

  init      生成测量CSV/物理清单和控制器模板；保留所有已有文件
  flight    用独立flight_train.csv / flight_valid.csv拟合physical.toml
  rotor     同时拟合flight与rotor_train.csv / rotor_valid.csv到physical.toml
  static    用static_train.csv / static_valid.csv导出static-check.toml
  check     在唯一新目录生成SYNTHETIC数据并检查完整物理导出
  drag-init 生成独立的closed_loop_train/valid.csv闭环日志模板
  drag      用闭环日志拟合so_rpy_rotor_drag；必须显式--mass KG
  drag-check 用有已知非零阻力/推力滞后的合成日志检查官方拟合
  inertia   转发双线摆计算参数，例如 --suspended-mass 0.0434
            --spacing 0.08 --length 0.4 --period 1.0

选项：
  --directory DIR       测量目录，默认outputs/measurements；必须在outputs内
  --device cpu|gpu|auto  运行环境选择；默认cpu，辨识计算始终使用CPU
  --negligible-drag     flight/rotor必需：明确确认所选片段可忽略阻力/外力
  --mass KG             drag必需：实测装机质量（kg）
  --data FILE           drag训练日志；默认目录内closed_loop_train.csv
  --validation FILE     drag独立验证日志；默认目录内closed_loop_valid.csv
  -h, --help            显示帮助

本入口只生成模板和拟合已有CSV，不会自动采集实机数据。
只有实测并核实质量/惯量等字段后，才能设置confirmed_measurements=true。
具体采集、单位和验证要求见docs/identification.md。
阻力闭环示例见docs/identification.md；其输出不是first_principles表，不能直接训练。
HELP
}

[[ $# -gt 0 ]] || { usage; exit 0; }
mode=$1
shift
case "$mode" in
  -h|--help|help) usage; exit 0 ;;
  init|flight|rotor|static|check|inertia|drag-init|drag|drag-check) ;;
  *) cf_die "未知辨识命令：$mode；请运行 ./dani identify --help。" ;;
esac

directory="$CF_OUTPUTS/measurements"
CF_DEVICE_REQUEST=cpu
negligible_drag=false
inertia_args=()
measured_mass=
data_file=
validation_file=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --directory)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "--directory 后需要目录。"
      directory=$2
      shift 2
      ;;
    --directory=*) directory=${1#*=}; shift ;;
    --device)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "--device 后需要 cpu、gpu 或 auto。"
      CF_DEVICE_REQUEST=$2
      shift 2
      ;;
    --device=*) CF_DEVICE_REQUEST=${1#*=}; shift ;;
    --negligible-drag) negligible_drag=true; shift ;;
    --mass)
      [[ "$mode" == drag ]] || cf_die "--mass 仅用于drag；惯量计算使用--suspended-mass。"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "--mass 后需要实测质量kg。"
      measured_mass=$2
      shift 2
      ;;
    --mass=*)
      [[ "$mode" == drag ]] || cf_die "--mass 仅用于drag。"
      measured_mass=${1#*=}
      shift
      ;;
    --data|--validation)
      [[ "$mode" == drag ]] || cf_die "$1 仅用于drag。"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || cf_die "$1 后需要CSV或NPZ文件路径。"
      if [[ "$1" == --data ]]; then data_file=$2; else validation_file=$2; fi
      shift 2
      ;;
    --data=*|--validation=*)
      [[ "$mode" == drag ]] || cf_die "--data/--validation 仅用于drag。"
      if [[ "$1" == --data=* ]]; then data_file=${1#*=}; else validation_file=${1#*=}; fi
      shift
      ;;
    -h|--help)
      usage
      if [[ "$mode" == inertia ]]; then
        cat <<'HELP'

双线摆参数（全部SI单位）：
  --suspended-mass KG  全部悬挂质量，包含夹具
  --spacing M         两根悬线的全间距
  --length M          单根悬线长度
  --period S          多周期小角度扭摆的平均周期
  --fixture-inertia I 可选：扣除同轴夹具惯量，kg*m^2
  --offset M          可选：无人机质心偏离悬挂轴的距离
  --drone-mass KG     使用--offset时必需：仅无人机质量
HELP
      fi
      exit 0
      ;;
    *)
      if [[ "$mode" == inertia ]]; then
        inertia_args+=("$1")
        shift
      else
        cf_die "无法识别的选项：$1；请运行 ./dani identify --help。"
      fi
      ;;
  esac
done
case "$CF_DEVICE_REQUEST" in
  cpu|gpu|auto) ;;
  *) cf_die "--device 只接受 cpu、gpu 或 auto。" ;;
esac

if [[ "$mode" == flight || "$mode" == rotor ]]; then
  "$negligible_drag" || cf_die "此飞行拟合需要显式 --negligible-drag；先按手册选取低空速、无其它外力的片段。"
fi

if [[ "$mode" == inertia ]]; then
  cf_runtime
  cf_container_cpu python -m crazyflow.identify_physics inertia "${inertia_args[@]}"
  exit
fi

[[ -n "$directory" ]] || cf_die "测量目录不能为空。"
container_dir=$(cf_to_container "$directory") || exit $?
case "$container_dir" in
  /outputs|/outputs/*) ;;
  *) cf_die "测量目录必须位于项目 outputs 目录内。" ;;
esac
host_dir="$CF_OUTPUTS${container_dir#/outputs}"

require_log() {
  local file=$1
  [[ -s "$host_dir/$file" && -f "$host_dir/$file" ]] ||
    cf_die "缺少非空输入文件：$host_dir/$file。请先独立采集/整理日志；init仅生成列名模板。"
}

if [[ "$mode" == flight || "$mode" == rotor || "$mode" == static ]]; then
  require_log measurements.toml
  if [[ "$mode" == static ]]; then
    require_log static_train.csv
    require_log static_valid.csv
  else
    require_log flight_train.csv
    require_log flight_valid.csv
    if [[ "$mode" == rotor ]]; then
      require_log rotor_train.csv
      require_log rotor_valid.csv
    fi
  fi
fi

if [[ "$mode" == drag ]]; then
  [[ -n "$measured_mass" ]] || cf_die "drag需要 --mass KG；请填写实测最终装机质量，不自动使用默认质量。"
  data_file=${data_file:-"$host_dir/closed_loop_train.csv"}
  validation_file=${validation_file:-"$host_dir/closed_loop_valid.csv"}
  drag_data=$(cf_to_container "$data_file") || exit $?
  drag_validation=$(cf_to_container "$validation_file") || exit $?
  [[ -f "$CF_OUTPUTS${drag_data#/outputs}" && -s "$CF_OUTPUTS${drag_data#/outputs}" ]] || cf_die "缺少训练日志：$data_file；先运行drag-init并整理真实数据。"
  [[ -f "$CF_OUTPUTS${drag_validation#/outputs}" && -s "$CF_OUTPUTS${drag_validation#/outputs}" ]] || cf_die "缺少独立验证日志：$validation_file。"
fi

cf_runtime
mkdir -p -- "$host_dir"

fit_converged() {
  local line success_count=0
  while IFS= read -r line; do
    [[ "$line" != *"Training success=False"* ]] || return 1
    if [[ "$line" == *"Training success=True"* ]]; then
      success_count=$((success_count + 1))
    fi
  done < "$1"
  [[ "$success_count" -eq 2 ]]
}

case "$mode" in
  init)
    # Generate elsewhere because the low-level template CLI replaces its outputs.
    stage=$(mktemp -d "$CF_OUTPUTS/.identify-template.XXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    stage_container=$(cf_to_container "$stage")
    cf_container_cpu python -m crazyflow.identify_physics template --output "$stage_container"
    cf_container_cpu python -m crazyflow.identify_physics controller-template \
      --output "$stage_container/controller.toml"
    templates=(measurements.toml:measurements.toml controller.toml:controller.toml
      flight.csv:flight_train.csv flight.csv:flight_valid.csv
      rotor.csv:rotor_train.csv rotor.csv:rotor_valid.csv
      static.csv:static_train.csv static.csv:static_valid.csv)
    for entry in "${templates[@]}"; do
      source_name=${entry%%:*}
      name=${entry#*:}
      if [[ -e "$host_dir/$name" || -L "$host_dir/$name" ]]; then
        printf '保留已有文件：%s\n' "$host_dir/$name"
      else
        cp -n -- "$stage/$source_name" "$host_dir/$name"
        printf '已生成：%s\n' "$host_dir/$name"
      fi
    done
    printf '模板尚未确认，不是实测结果。下一步见：%s/docs/identification.md\n' "$CF_ROOT"
    ;;
  drag-init)
    stage=$(mktemp -d "$CF_OUTPUTS/.identify-template.XXXXXX")
    trap 'rm -rf -- "$stage"' EXIT
    stage_container=$(cf_to_container "$stage")
    cf_container_cpu python -m crazyflow.identify template --output "$stage_container/closed_loop.csv"
    for name in closed_loop_train.csv closed_loop_valid.csv; do
      if [[ -e "$host_dir/$name" || -L "$host_dir/$name" ]]; then
        printf '保留已有文件：%s\n' "$host_dir/$name"
      else
        cp -n -- "$stage/closed_loop.csv" "$host_dir/$name"
        printf '已生成空列名模板：%s\n' "$host_dir/$name"
      fi
    done
    printf '请分别采集训练和验证飞行；格式与单位见docs/identification.md。\n'
    ;;
  drag)
    if ! cf_container_cpu python -m crazyflow.identify fit \
        --dynamics so_rpy_rotor_drag --mass "$measured_mass" \
        --data "$drag_data" --validation "$drag_validation" \
        --output "$container_dir/closed-loop-so-rpy-rotor-drag.toml" \
        2>&1 | tee "$host_dir/fit.log"; then
      cf_die "闭环阻力拟合失败，请检查fit.log中的同步、单位、独立日志和激励诊断。"
    fi
    fit_converged "$host_dir/fit.log" || cf_die "官方优化器未收敛，请检查fit.log；已写出的候选参数不能当成有效辨识结果。"
    printf '闭环候选参数：%s/closed-loop-so-rpy-rotor-drag.toml\n请人工检查阻力符号/验证误差；此文件不是first_principles表，不能直接训练。\n' "$host_dir"
    ;;
  drag-check)
    check_dir=$(mktemp -d "$host_dir/drag-check-$(date +%Y%m%d-%H%M%S).XXXXXX")
    check_container=$(cf_to_container "$check_dir")
    cf_container_cpu python -m crazyflow.identify synthetic \
      --dynamics so_rpy_rotor_drag --seconds 15 --seed 0 \
      --output "$check_container/closed_loop_train.csv"
    cf_container_cpu python -m crazyflow.identify synthetic \
      --dynamics so_rpy_rotor_drag --seconds 15 --seed 1 \
      --output "$check_container/closed_loop_valid.csv"
    if ! cf_container_cpu python -m crazyflow.identify fit \
        --dynamics so_rpy_rotor_drag --mass 0.0434 \
        --data "$check_container/closed_loop_train.csv" \
        --validation "$check_container/closed_loop_valid.csv" \
        --output "$check_container/closed-loop-so-rpy-rotor-drag.toml" \
        2>&1 | tee "$check_dir/fit.log"; then
      cf_die "合成阻力检查失败；日志位于$check_dir/fit.log。"
    fi
    fit_converged "$check_dir/fit.log" || cf_die "合成阻力优化未收敛，不能报告检查通过；请查看$check_dir/fit.log。"
    printf 'SYNTHETIC非零阻力/推力滞后检查完成：%s\n已知参数在两份CSV同名JSON中。输出为闭环模型，不能直接用于first_principles训练。\n' "$check_dir"
    ;;
  flight|rotor|static)
    fit_args=(export --measurements "$container_dir/measurements.toml")
    if [[ "$mode" == static ]]; then
      fit_args+=(--static "$container_dir/static_train.csv"
        --static-validation "$container_dir/static_valid.csv"
        --output "$container_dir/static-check.toml")
    else
      fit_args+=(--flight "$container_dir/flight_train.csv"
        --flight-validation "$container_dir/flight_valid.csv" --negligible-drag)
      if [[ "$mode" == rotor ]]; then
        fit_args+=(--rotor "$container_dir/rotor_train.csv"
          --rotor-validation "$container_dir/rotor_valid.csv")
      fi
      fit_args+=(--output "$container_dir/physical.toml")
    fi
    if ! cf_container_cpu python -m crazyflow.identify_physics "${fit_args[@]}"; then
      cf_die "辨识未完成，请查看上方诊断。核实实测质量/惯量后才能设 confirmed_measurements=true；还需有效机械RPM、正确单位、足够激励及独立验证日志。合成数据只能用 check。"
    fi
    printf '导出完成：%s；请检查同名JSON中的拟合误差和未测量项。\n' "$host_dir"
    ;;
  check)
    check_dir=$(mktemp -d "$host_dir/check-$(date +%Y%m%d-%H%M%S).XXXXXX")
    check_container=$(cf_to_container "$check_dir")
    cf_container_cpu python -m crazyflow.identify_physics synthetic --output "$check_container"
    cf_container_cpu python -m crazyflow.identify_physics export \
      --measurements "$check_container/measurements.toml" \
      --flight "$check_container/flight_train.csv" \
      --flight-validation "$check_container/flight_valid.csv" --negligible-drag \
      --rotor "$check_container/rotor_train.csv" \
      --rotor-validation "$check_container/rotor_valid.csv" \
      --allow-synthetic --output "$check_container/fitted.toml"
    printf 'SYNTHETIC软件检查完成：%s\n不是实机参数，不要作为实测训练配置。\n' "$check_dir"
    ;;
esac
