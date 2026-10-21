#!/bin/bash
#SBATCH --job-name=char_audit_strat
#SBATCH --partition=rome
#SBATCH --time=00:15:00
#SBATCH --output=logs/audit_strat_%j.out

# 分层审核抽样。**不需要 GPU**：v2 的 run_eval.py 把每一轮的 judge 判定写进了
# per_turn.csv，这个脚本直接从那里抽样，不用重判。v1 必须重判是因为只存了
# 每条对话的汇总值。
#
# 其实在登录节点直接跑就行（几秒钟），这个作业脚本只是为了留一份可复现的命令。

# 站点相关设置。各集群的模块名不同（module avail 查），所以这里不写死：
# 把 CLUSTER_MODULES 设进 ~/.bashrc，例如
#   export CLUSTER_MODULES="2024 Anaconda3/2024.02"
#   export MODELS_DIR=/your/shared/storage/models
# 详见 docs/CLUSTER_GUIDE.md 第 0 节。
module purge
for _m in ${CLUSTER_MODULES:-}; do module load "$_m"; done
source activate "${CONDA_ENV:-memory-eval}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

python src/eval/ooc_audit_stratified.py \
    --per-turn results/per_turn.csv results_seed1/per_turn.csv results_seed2/per_turn.csv \
    --raw-dirs results/raw_outputs results_seed1/raw_outputs results_seed2/raw_outputs \
    --dialogues data/processed/dialogues.jsonl \
    --model qwen2.5-7b \
    --schemes raw_context structured_memory \
    --n-flagged 20 --n-unflagged 20 \
    --out-dir results/ooc_audit_v2

echo "Done. 盲评清单在 results/ooc_audit_v2/sheet.md"
echo "标完 labels_template.csv 之后："
echo "  python src/eval/ooc_audit_score.py --audit-dir results/ooc_audit_v2 --labels <你填好的.csv>"
