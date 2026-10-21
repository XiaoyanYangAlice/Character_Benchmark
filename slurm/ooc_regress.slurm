#!/bin/bash
#SBATCH --job-name=char_ooc_regress
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ooc_regress_%j.out

# OOC judge prompt 回归：用 results/ooc_audit 里已标好的 40 条，V1 / V2 各判 5 次对比。
# 2 个 prompt × 40 条 × 5 次 = 400 次短调用，几分钟；启动 vLLM 约 2–3 分钟。
# 前提：results/ooc_audit/ 里有 sheet.md、key.csv、labels.csv。
#
# 提交（默认 judge = Qwen2.5-14B，1 张卡）：sbatch slurm/ooc_regress.slurm
# 换 judge（环境变量传参，TAG 用来区分输出文件，TP = 张量并行卡数，--gpus 要跟 TP 一致）：
#   JUDGE_PATH=$MODELS_DIR/Llama-3.1-8B-Instruct TAG=llama8b sbatch slurm/ooc_regress.slurm
#   JUDGE_PATH=$MODELS_DIR/Mistral-Small-24B-Instruct-2501 TP=2 TAG=mistral24b sbatch --gpus=2 slurm/ooc_regress.slurm

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

for f in sheet.md key.csv labels.csv; do
    [ -f "results/ooc_audit/$f" ] || { echo "缺 results/ooc_audit/$f"; exit 1; }
done

JUDGE_PATH=${JUDGE_PATH:-$MODELS_DIR/Qwen2.5-14B-Instruct}
TP=${TP:-1}
TAG=${TAG:-qwen14b}
echo "judge=$JUDGE_PATH tp=$TP tag=$TAG"
[ -d "$JUDGE_PATH" ] || { echo "找不到模型目录 $JUDGE_PATH"; exit 1; }

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1))) vllm serve "$JUDGE_PATH" \
    --served-model-name judge-model --port 8002 --dtype bfloat16 --max-model-len 8192 \
    --tensor-parallel-size "$TP" &
JUDGE_PID=$!

UP=0
for i in $(seq 1 60); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:8002/v1/models" | grep -q 200 && { UP=1; break; }
    sleep 10
done
if [ "$UP" -ne 1 ]; then
    echo "judge 服务 10 分钟内没起来，检查上面的日志。"
    kill $JUDGE_PID 2>/dev/null
    exit 1
fi

python src/eval/ooc_judge_regress.py \
    --config configs/experiment.yaml \
    --judge-port 8002 \
    --audit-dir results/ooc_audit \
    --raw-dirs results/raw_outputs results_seed1/raw_outputs results_seed2/raw_outputs \
    --repeats 5 --tag "$TAG" || {
    echo "!!!!! ooc_judge_regress.py 报错退出，不是正常完成 !!!!!"
    kill $JUDGE_PID 2>/dev/null
    exit 1
}

kill $JUDGE_PID 2>/dev/null
echo "Done. 结果见上面的表和 results/ooc_audit/regress_per_item_${TAG}.csv"
