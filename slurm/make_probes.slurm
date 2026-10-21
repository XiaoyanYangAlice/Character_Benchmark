#!/bin/bash
#SBATCH --job-name=char_make_probes
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/make_probes_%j.out

# 给 dialogues.jsonl 生成「记忆探针」，只需要 judge 那一个模型，1 张卡。
#
# 必须在 build_dialogues.py 之后、run_eval.py 之前跑。没有探针的话，整个实验
# 里**没有任何指标在直接测记忆**——ooc_rate 和 fact_consistency 都只测"像不像
# 这个角色"，一个完全没有记忆的模型只要人设稳就能拿高分（v1 的 Llama 就是
# fact_consistency=1.000）。
#
# 用法：sbatch slurm/make_probes.slurm

module purge; module load 2025; module load Anaconda3/2025.06-1
source activate memory-eval
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs

JUDGE_PATH=${JUDGE_PATH:-/projects/prjs1995/models/Qwen2.5-14B-Instruct}

CUDA_VISIBLE_DEVICES=0 vllm serve "$JUDGE_PATH" \
    --served-model-name judge-model --port 8002 --dtype bfloat16 --max-model-len 8192 &
JUDGE_PID=$!

for i in $(seq 1 60); do
    curl -s -o /dev/null -w "%{http_code}" "http://localhost:8002/v1/models" | grep -q 200 && break
    sleep 10
done
curl -s -o /dev/null -w "%{http_code}" "http://localhost:8002/v1/models" | grep -q 200 || {
    echo "judge 服务 10 分钟内没起来，退出"; kill $JUDGE_PID 2>/dev/null; exit 1; }

# 先备份：make_probes.py 是原地改写 dialogues.jsonl（只加 memory_probes 字段），
# 万一生成的探针质量不行，要能退回上一版重来。
cp data/processed/dialogues.jsonl data/processed/dialogues.jsonl.bak

python src/generate/make_probes.py \
    --config configs/experiment.yaml \
    --dialogues data/processed/dialogues.jsonl \
    --output data/processed/dialogues.jsonl \
    --judge-port 8002 --n-probes 3 --n-candidates 8 --from-first 10 || {
    echo "!!!!! make_probes.py 报错退出 !!!!!"
    cp data/processed/dialogues.jsonl.bak data/processed/dialogues.jsonl
    kill $JUDGE_PID 2>/dev/null; exit 1; }

kill $JUDGE_PID 2>/dev/null
echo "Done. 探针已写进 data/processed/dialogues.jsonl（原文件备份在 .bak）"
echo "下一步：抽查几条探针是不是真的能从用户发言里答出来，再跑 run_eval。"
