#!/bin/bash
#SBATCH --job-name=lifelike-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2                  # harness_x32b runs the 8B and 32B models together
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
# Quick check that vLLM, the weights and the judges work: 1 character, 4 probes, 1 seed,
# Qwen3-8B base / harness / harness with a 32B extractor, one judge. Look at results/smoke/report.md and the transcripts.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source slurm/surf.env
activate_env
cd "$REPO_DIR"; export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"; mkdir -p logs
python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/local_surf.yaml"))
c.update(run_name="smoke", seeds=[0], only_characters=["lin_wan"],
         only_probes=["affect_conflict", "memory_conflict_claim", "memory_compatible_claim", "memory_canon"])
c["systems"] = [s for s in c["systems"] if s["id"] in ("base_qwen3_8b", "harness_qwen3_8b", "harness_x32b_qwen3_8b")]
c["judges"] = c["judges"][:1]
yaml.safe_dump(c, open("configs/_smoke.yaml", "w"), allow_unicode=True)
PY
python -m lifelike_bench.cli local configs/_smoke.yaml
