#!/bin/bash
#SBATCH --job-name=char_ooc_audit
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ooc_audit_%j.out

# 抽样重判 OOC：只需要 judge 模型（Qwen2.5-14B），1 张 GPU。
# 从 qwen 的生成结果里随机抽 600 个角色轮次重判，导出 20 条被标出戏 + 20 条未标的盲评清单。
# 600 次短调用、8 路并发，预计 5–10 分钟；启动 vLLM 约 2–3 分钟。
#
# 提交：sbatch slurm/ooc_audit.slurm
# 产出：results/ooc_audit/{sheet.md, labels_template.csv, key.csv, meta.json}

# 站点相关设置。各集群的模块名不同（module avail 查），所以这里不写死：
# 把 CLUSTER_MODULES 设进 ~/.bashrc，例如
#   export CLUSTER_MODULES="2024 Anaconda3/2024.02"
#   export MODELS_DIR=/your/shared/storage/models
# 详见 docs/CLUSTER_GUIDE.md 第 0 节。
module purge
for _m in ${CLUSTER_MODULES:-}; do module load "$_m"; done
source activate "${CONDA_ENV:-memory-eval}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results

JUDGE_PATH=$MODELS_DIR/Qwen2.5-14B-Instruct

CUDA_VISIBLE_DEVICES=0 vllm serve "$JUDGE_PATH" \
    --served-model-name judge-model --port 8002 --dtype bfloat16 --max-model-len 8192 &
JUDGE_PID=$!

UP=0
for i in $(seq 1 30); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:8002/v1/models" | grep -q 200 && { UP=1; break; }
    sleep 10
done
if [ "$UP" -ne 1 ]; then
    echo "judge 服务 5 分钟内没起来，检查上面的日志。"
    kill $JUDGE_PID 2>/dev/null
    exit 1
fi

python src/eval/ooc_audit_sample.py \
    --config configs/experiment.yaml \
    --judge-port 8002 \
    --model qwen2.5-7b \
    --raw-dirs results/raw_outputs results_seed1/raw_outputs results_seed2/raw_outputs \
    --n-sample 600 --n-flagged 20 --n-unflagged 20 \
    --out-dir results/ooc_audit || {
    echo "!!!!! ooc_audit_sample.py 报错退出，不是正常完成 !!!!!"
    kill $JUDGE_PID 2>/dev/null
    exit 1
}

kill $JUDGE_PID 2>/dev/null
echo "Done. 清单在 results/ooc_audit/sheet.md"
