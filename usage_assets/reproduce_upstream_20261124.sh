#!/usr/bin/env bash
# Run inside Docker from the Crazyflow checkout. Historical outputs stay untouched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f /opt/crazyflow/activate-research.sh ]]; then
  source /opt/crazyflow/activate-research.sh
fi
export MPLBACKEND=Agg MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
unset PYTEST_DISABLE_PLUGIN_AUTOLOAD
out=usage_assets/upstream_20261124
mkdir -p "$out"
run() {
  local name=$1
  shift
  { printf '$'; printf ' %q' "$@"; printf '\n'; "$@"; } 2>&1 | tee "$out/$name.txt"
}
features() {
  run features_gpu python usage_assets/upstream_features_20261124.py --device gpu --overwrite
  run control_cpu env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python usage_assets/upstream_features_20261124.py --device cpu --section control --overwrite
  run splats_cpu env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python usage_assets/upstream_features_20261124.py --device cpu --section splats --overwrite
}
frozen() {
  run frozen_gym python usage_assets/revalidate_frozen_20261124.py --suite gym
  run frozen_official python usage_assets/revalidate_frozen_20261124.py --suite official
}
racing() {
  run racing env JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python usage_assets/revalidate_racing_20261124.py
}
regression() {
  run core pytest -q tests
  run render xvfb-run -a pytest -q -m render tests
  run markdown pytest -q --markdown-docs --markdown-docs-syntax=superfences --doctest-glob='*.md' crazyflow/ docs/ --ignore=docs/gen_ref_pages.py
  run continuation_python python usage_assets/train_gymnasium_checks_20261015.py
  run continuation_fused python usage_assets/train_gymnasium_checks_20261015.py --env-backend fused
  run fused_equivalence python usage_assets/fast_gymnasium_20261015.py --verify-device gpu --verify-steps 64 --skip-benchmark --output "$out/fused_equivalence.json"
}
media() {
  mkdir -p "$out/docs_hover" "$out/docs_integrators"
  run media_hover python docs/capture_example.py examples/control/hover.py video "$out/docs_hover" --cam_config 'distance=0.7 elevation=-10 lookat=0.07,0.07,0.17'
  run media_integrators python docs/capture_example.py examples/control/integrators.py plot,output "$out/docs_integrators"
}
artifacts() {
  run historical_artifacts python usage_assets/verify_fulltrain_20261015.py --require-complete --output "$out/historical_artifact_verification.json"
}
fuzzing() {
  if [[ ! -x /opt/crazyflow/.pixi/envs/fuzzing/bin/python ]]; then
    run firmware_environment pixi install --manifest-path /opt/crazyflow/pyproject.toml --locked -e fuzzing
  fi
  run firmware env JAX_PLATFORMS=cpu /opt/crazyflow/.pixi/envs/fuzzing/bin/python -m pytest -q -o addopts= tests/fuzzing
}
case ${1:-all} in
  features|frozen|racing|regression|media|artifacts|fuzzing) "$1" ;;
  all) features; frozen; racing; regression; media; artifacts; fuzzing ;;
  *) echo 'Usage: reproduce_upstream_20261124.sh [features|frozen|racing|regression|media|artifacts|fuzzing|all]' >&2; exit 2 ;;
esac
