#!/bin/bash
#SBATCH --job-name=char_extract_full
#SBATCH --partition=gpu_a100   # 按你的站点改，或 sbatch --partition=<名字> 覆盖
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_%j.out

# 全量跑：冒烟测试通过后再提交这个。--time 按冒烟测试记录的单次调用耗时 ×
# 故事数 × 4 个类别估算，留出余量；样本量大时考虑拆成多个数组作业(--array)并行。
# 在仓库根目录提交：sbatch slurm/ontology_run_extraction.slurm
#
# 当前数据源（PIPPA）是纯英文角色描述，所以只跑 en(v4) 一个语言版本。
# 之前设计的 zh(v2)/en(v4) 双语一致性对照（validate_extractions.py --compare-with）
# 需要同一批角色卡的中英文平行文本才有意义——PIPPA 没有中文版，暂时不适用。
# 如果之后补充中文语料（SillyTavern 中文角色卡 / AO3 中文故事，见 docs/DATASET.md），
# 可以再加一段 --lang zh 的抽取 + --compare-with 对照，用来做真正的双语一致性检查。

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
# 原因同 ontology_smoke_test.slurm 里的注释：计算节点没有 nvcc，关掉 FlashInfer
# 采样器避免它尝试 JIT 编译。
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs data/processed

MODEL_PATH=$MODELS_DIR/Qwen2.5-14B-Instruct   # 按你实际下载的模型改

# --max-model-len 8192：原因同 ontology_smoke_test.slurm 里的注释——我们的输入
# 用不到默认的 32768 上下文，降低上限能省出足够的 KV cache 显存，避免启动失败。
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

python src/ontology_extraction/extract_character_features.py \
    --stories data/processed/stories.jsonl \
    --prompts-dir src/ontology_extraction/prompts \
    --lang en \
    --api-base http://localhost:8000/v1 \
    --model agent-model \
    --output data/processed/character_features_en.jsonl

kill $VLLM_PID

# 目前只有单一语言（en），没有平行的中文数据可比较，用 basic_report 做质检
# （类别分布、每故事角色数、modality 分布），不跑 --compare-with。
python src/ontology_extraction/validate_extractions.py \
    --extractions data/processed/character_features_en.jsonl \
    > logs/validation_report.txt

echo "Done. See logs/validation_report.txt"
