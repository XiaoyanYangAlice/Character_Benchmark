#!/bin/bash
#SBATCH --job-name=char_run_eval
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=3
#SBATCH --time=10:00:00
#SBATCH --output=logs/run_eval_%j.out

# 跑 run_eval.py 主实验矩阵：3 个 vLLM 服务（两个角色模型 + 一个 judge），各占
# 一张 GPU、一个端口，起完之后跑 src/eval/run_eval.py。
#
# ⚠ v2 的规模比 v1 大很多，--time 从 2.5 小时提到 8 小时。粗算（相对 v1）：
#   对话轮次  6 -> 约 20（--min-user-turns 12，--max-user-turns 30）  约 3x
#   记忆方案  3 -> 5（加了 no_memory / shuffled_facts）              约 1.7x
#   judge 调用 judge_repeats 1 -> 3                                  约 3x
#   structured 还要额外跑事件抽取（每个来回 1 次 LLM 调用）
#   对话条数可能从 61 降到更少（长对话筛选），这会抵消一部分
# **先用 --limit 3 跑一次校准作业，拿到实测单条耗时再决定 --time**，不要直接
# 提交全量——v1 就是靠校准作业才把时间估准的。校准：
#   sbatch slurm/run_eval_calibrate.slurm
#
# 用法：确认 configs/experiment.yaml 里的模型/judge 没有变化后直接提交：
#   sbatch slurm/run_eval.slurm

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
# 原因同 ontology_smoke_test.slurm：计算节点没有 nvcc，关掉 FlashInfer 采样器
# 避免它尝试 JIT 编译。
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results

# 跟 configs/experiment.yaml 里的 name/path 一一对应，改配置时这里也要同步改。
QWEN_PATH=$MODELS_DIR/Qwen2.5-7B-Instruct
LLAMA_PATH=$MODELS_DIR/Llama-3.1-8B-Instruct
JUDGE_PATH=$MODELS_DIR/Qwen2.5-14B-Instruct

# CUDA_VISIBLE_DEVICES 把每个 vLLM 进程钉在自己的 GPU 上，避免三个进程抢同一张卡。
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
wait_for_port 8000 || { echo "端口 8000（qwen2.5-7b）5 分钟内没起来"; ALL_UP=0; }
wait_for_port 8001 || { echo "端口 8001（llama3.1-8b）5 分钟内没起来"; ALL_UP=0; }
wait_for_port 8002 || { echo "端口 8002（judge, qwen2.5-14b）5 分钟内没起来"; ALL_UP=0; }

if [ "$ALL_UP" -ne 1 ]; then
    echo "至少一个 vLLM 服务没起来，检查上面的日志找报错，不再继续跑 run_eval.py。"
    kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null
    exit 1
fi

# 显式检查退出码：v1 踩过"脚本报错但仍然打印 Done"的坑（seed 作业因为参数
# 不存在秒退，日志末尾照样是 Done，差点把空结果当成跑完了）。
python src/eval/run_eval.py --workers ${WORKERS:-12} ${EXTRA_ARGS} \
    --config configs/experiment.yaml \
    --model-port qwen2.5-7b=8000 \
    --model-port llama3.1-8b=8001 \
    --model-port judge=8002 || {
    echo "!!!!! run_eval.py 报错退出，这不是正常完成 !!!!!"
    kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null
    exit 1
}

kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null

echo "Done. 汇总 results/summary.csv，逐轮判定 results/per_turn.csv，"
echo "探针 results/probe_results.csv，完整生成记录 results/raw_outputs/"
