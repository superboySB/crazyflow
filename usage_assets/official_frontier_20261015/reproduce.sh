#!/usr/bin/env bash
set -euo pipefail
cd /workspace/crazyflow
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
python usage_assets/official_frontier_20261015/run.py
python usage_assets/official_frontier_20261015/run.py --algorithm ppo --periods 3.5 --seeds 43 --budget-multiplier 2
python usage_assets/official_frontier_20261015/finish.py
python usage_assets/official_frontier_20261015/replay.py
python usage_assets/official_frontier_20261015/replay.py \
    --checkpoint trained_models/official_frontier_ppo_T3.5s_budget2_seed43.ckpt \
    --output usage_assets/official_frontier_20261015/ppo_T3.5s_position_holdout.mp4
