#!/bin/bash
#SBATCH --job-name=char_extract_smoke
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_%j.out

# 冒烟测试：只跑 --limit 2 个故事，人工检查结果没问题后再用 ontology_run_extraction.slurm 放量。
# 在仓库根目录提交：sbatch slurm/ontology_smoke_test.slurm
# 模型名换成你 huggingface-cli download 下载到 $MODELS_DIR/ 的那个目录名
# （存储路径见 docs/CLUSTER_GUIDE.md 第 0 节，用 MODELS_DIR 指定）。

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
# 计算节点上没有 CUDA 开发工具链（找不到 nvcc），FlashInfer 的 top-k/top-p 采样
# kernel 第一次要 JIT 编译会报 "Could not find nvcc"——关掉 FlashInfer 采样器，
# 退回 vLLM 自带的默认实现，不需要临时编译。
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs data/processed

MODEL_PATH=$MODELS_DIR/Qwen2.5-14B-Instruct   # 按你实际下载的模型改

# --max-model-len 8192：我们的输入（抽取 prompt + 单个角色故事文本）远用不到模型
# 默认的 32768 上下文长度，之前冒烟测试实测 14B 模型 bf16 权重占满显存后，
# KV cache 按 32768 长度申请会报 "available KV cache memory" 不够，直接把上限
# 降到 8192（对我们的输入绰绰有余）即可解决，不用改 GPU 资源申请。
vllm serve "$MODEL_PATH" \
    --served-model-name agent-model --port 8000 --dtype bfloat16 --max-model-len 8192 &
VLLM_PID=$!

VLLM_UP=0
for i in $(seq 1 30); do
    curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models | grep -q 200 && { VLLM_UP=1; break; }
    sleep 10
done

if [ "$VLLM_UP" -ne 1 ]; then
    echo "vLLM 服务 5 分钟内没起来，检查上面的日志找报错，不再继续跑抽取。"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# 当前数据源是 PIPPA（纯英文角色描述），用 --lang en（对应 v4 英文版 prompt）。
# 如果之后加入中文语料（SillyTavern 中文角色卡 / AO3 中文故事），改回 --lang zh
# 并参考 ontology_run_extraction.slurm 里注释掉的双语对照逻辑。
python src/ontology_extraction/extract_character_features.py \
    --stories data/processed/stories.jsonl \
    --prompts-dir src/ontology_extraction/prompts \
    --lang en \
    --api-base http://localhost:8000/v1 \
    --model agent-model \
    --output data/processed/character_features_en_smoke.jsonl \
    --limit 2

kill $VLLM_PID
