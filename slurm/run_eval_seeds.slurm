#!/bin/bash
#SBATCH --job-name=char_run_eval_seeds
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=3
#SBATCH --time=24:00:00
#SBATCH --output=logs/run_eval_seeds_%j.out

# 重复实验：用不同的采样种子把全量矩阵再跑 2 遍（seed 1、seed 2），每遍写到独立
# 目录 results_seed1/ results_seed2/，不会覆盖第一次（未固定种子）的 results/。
# 两遍复用同一组 vLLM 服务，只启动一次。
#
# ⚠ v2 的时间上限从 3.5 小时提到 16 小时：对话更长、方案从 3 个变 5 个、
# judge_repeats 从 1 变 3，还多了事件抽取。**先跑 run_eval_calibrate.slurm
# 拿实测单条耗时**，再按 (单条耗时 x 总次数 x 2 个 seed) 决定最终 --time，
# 不要直接按这里的默认值提交。
# 以下是 v1 的估算记录，留作对照：
# 单遍实测约 73 分钟生成 + 2 分钟一次性启动，两遍约 148 分钟，
# 加约 40% 余量。
#
# 提交：sbatch slurm/run_eval_seeds.slurm
# 跑完：python src/eval/compare_schemes.py --csv results/summary.csv results_seed1/summary.csv results_seed2/summary.csv

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
mkdir -p logs

QWEN_PATH=$MODELS_DIR/Qwen2.5-7B-Instruct
LLAMA_PATH=$MODELS_DIR/Llama-3.1-8B-Instruct
JUDGE_PATH=$MODELS_DIR/Qwen2.5-14B-Instruct

CUDA_VISIBLE_DEVICES=0 vllm serve "$QWEN_PATH" \
    --served-model-name agent-model-qwen --port 8000 --dtype bfloat16 --max-model-len 8192 &
QWEN_PID=$!
CUDA_VISIBLE_DEVICES=1 vllm serve "$LLAMA_PATH" \
    --served-model-name agent-model-llama --port 8001 --dtype bfloat16 --max-model-len 8192 &
LLAMA_PID=$!
CUDA_VISIBLE_DEVICES=2 vllm serve "$JUDGE_PATH" \
    --served-model-name judge-model --port 8002 --dtype bfloat16 --max-model-len 8192 &
JUDGE_PID=$!

wait_for_port() {
    local port=$1
    for i in $(seq 1 30); do
        curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" | grep -q 200 && return 0
        sleep 10
    done
    return 1
}

ALL_UP=1
wait_for_port 8000 || { echo "端口 8000 没起来"; ALL_UP=0; }
wait_for_port 8001 || { echo "端口 8001 没起来"; ALL_UP=0; }
wait_for_port 8002 || { echo "端口 8002 没起来"; ALL_UP=0; }
if [ "$ALL_UP" -ne 1 ]; then
    echo "至少一个 vLLM 服务没起来，不再继续。"
    kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null
    exit 1
fi

for SEED in 1 2; do
    echo "===== seed ${SEED} 开始 ====="
    python src/eval/run_eval.py --workers ${WORKERS:-12} ${EXTRA_ARGS} \
        --config configs/experiment.yaml \
        --model-port qwen2.5-7b=8000 \
        --model-port llama3.1-8b=8001 \
        --model-port judge=8002 \
        --seed ${SEED} --out-dir results_seed${SEED} || {
        echo "!!!!! seed ${SEED} 的 run_eval.py 报错退出，不再继续，不是正常完成 !!!!!"
        kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null
        exit 1
    }
    echo "===== seed ${SEED} 结束 ====="
done

kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null
echo "Done. 结果在 results_seed1/ 和 results_seed2/"
