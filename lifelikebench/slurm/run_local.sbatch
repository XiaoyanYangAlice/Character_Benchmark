#!/bin/bash
#SBATCH --job-name=lifelike-local
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2                  # 2 H100s: both judges run at once; 1 also works (sequential)
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# Fully local benchmark run. Every model is served by vLLM on this node, phase by phase.
# Usage (from the repo dir):  sbatch slurm/run_local.sbatch [configs/local_surf.yaml]
# Resumable: if the job hits its time limit, just submit it again.
# On A100-40GB nodes (`sinfo` to check the partition name) use tp: 2 for the 27–32B judges.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/surf.env
activate_env
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
mkdir -p logs "$RESULTS_DIR"
CONFIG=${1:-configs/local_surf.yaml}

python -m lifelike_bench.cli plan  "$CONFIG"
python -m lifelike_bench.cli local "$CONFIG"
