#!/bin/bash
#SBATCH --job-name=lifelike-train
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Train the 8B student on one H100.
# Usage (from the repo dir):
#   sbatch slurm/train.sbatch sft configs/train_sft.yaml [--smoke]
#   sbatch slurm/train.sbatch opd configs/train_opd.yaml [--smoke]
# For --smoke runs you can shorten the queue wait with:  sbatch --time=00:30:00 slurm/train.sbatch ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/surf.env
activate_env
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
KIND=${1:?sft or opd}
CONFIG=${2:?config file}
shift 2
python -m lifelike_bench.train.check_env
python -m lifelike_bench.train."$KIND" "$CONFIG" "$@"
