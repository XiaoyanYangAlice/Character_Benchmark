#!/bin/bash
#SBATCH --job-name=char_run_eval_calibrate
#SBATCH --partition=gpu_a100
#SBATCH --gpus=3
#SBATCH --time=01:00:00
#SBATCH --output=logs/run_eval_%j.out

# run_eval.slurm 的校准版：只跑前 3 条对话（--limit 3，写到 results_calib/，不覆盖正式结果），--time 也相应改小成
# 30 分钟，目的是在提交全量作业（366 次生成，--time 没有实测校准过）之前，先
# 花一点点 GPU 时间看看实际单条耗时，再回去把 slurm/run_eval.slurm 里的
# --time 调准，避免全量跑到一半被系统按超时杀掉、或者申请了用不上的 GPU 小时。
#
# 跑完看 logs/run_eval_<jobid>.out 里每条打印的 (X.Xs) 耗时，乘以 366/5 估算
# 全量需要多久，回填 slurm/run_eval.slurm 的 --time，再 sbatch 那个文件。

module purge; module load 2025; module load Anaconda3/2025.06-1
source activate memory-eval
export HF_HUB_OFFLINE=1
# 原因同 ontology_smoke_test.slurm：计算节点没有 nvcc，关掉 FlashInfer 采样器
# 避免它尝试 JIT 编译。
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results

# 跟 configs/experiment.yaml 里的 name/path 一一对应，改配置时这里也要同步改。
QWEN_PATH=/projects/prjs1995/models/Qwen2.5-7B-Instruct
LLAMA_PATH=/projects/prjs1995/models/Llama-3.1-8B-Instruct
JUDGE_PATH=/projects/prjs1995/models/Qwen2.5-14B-Instruct

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

python src/eval/run_eval.py --workers ${WORKERS:-12} ${EXTRA_ARGS} \
    --config configs/experiment.yaml \
    --model-port qwen2.5-7b=8000 \
    --model-port llama3.1-8b=8001 \
    --model-port judge=8002 \
    --limit 3 --out-dir results_calib

kill $QWEN_PID $LLAMA_PID $JUDGE_PID 2>/dev/null

echo "Done. 结果在 results/summary.csv，完整生成记录在 results/raw_outputs/"
