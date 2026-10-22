# 在 SLURM 集群上复现：逐步指南

这份指南覆盖从登录一台 GPU 集群到跑出 `results/summary.csv` 的完整链路。它假设
集群用 SLURM 调度、有 A100 级别的 GPU、计算节点不能联网——这是学术 HPC 最常见
的形态。原始实验是在一台荷兰的国家级集群上跑的，但下面刻意不绑定任何站点：
凡是各站点不同的东西都抽成了变量，第 0 节说明怎么查。

## 0. 先确定这几个站点变量

每个集群的分区名、模块名、存储路径、配额命令都不一样，**没有一套命令是到处都能
用的**。先把下面这张表填出来，后面所有命令都引用它。

| 变量 | 含义 | 怎么查 |
|---|---|---|
| `LOGIN_HOST` | 登录节点主机名 | 你所在机构或集群的文档 |
| `PARTITION` | GPU 分区名 | `sinfo -s`，找带 GPU 的那几行 |
| `ACCOUNT` | 计费账号（部分站点必填） | `sacctmgr show assoc where user=$USER format=account,partition`；有的站点提供自己的包装命令 |
| `MODELS_DIR` | 放模型权重的目录 | 见下面关于存储的说明 |
| 模块名 | 加载 Python/Anaconda 的 `module load` 参数 | `module avail python`、`module avail anaconda` |

存储要特别注意：**不要把模型权重放 `$HOME`**，那里的配额通常只有几十 GB，而一个
14B 模型的 bf16 权重约 28GB。多数站点提供一块共享的高速盘（常见名字是
`/scratch`、`/scratch-shared/$USER`、`/projects/<项目号>`、`/work/$USER`）。
注意 `df -h` 显示的是整个文件系统的空闲量，**不等于你的配额上限**，配额要用站点
自己的命令查（`quota`、`myquota`、`lfs quota` 之类）。原始实验遇到过
`/scratch-shared/$USER` 这个目录对该账号根本没创建的情况，所以别假设它存在，先
`ls` 一下。

本仓库的 SLURM 脚本统一通过 `MODELS_DIR` 环境变量找模型，默认 `$HOME/models`：

```bash
export MODELS_DIR=/your/shared/storage/models    # 放进 ~/.bashrc 省得每次敲
```

## 1. 传代码、建环境

```bash
ssh <LOGIN_HOST>
git clone https://github.com/XiaoyanYangAlice/Character_Benchmark.git
cd Character_Benchmark
```

建环境。模块名按第 0 节查到的改：

```bash
module purge
module load <你的 anaconda 模块>        # 例如 Anaconda3/2024.02
conda create -n memory-eval python=3.11 -y
source activate memory-eval
pip install -r requirements.txt         # vllm + openai + sentence-transformers
```

只做一次。之后每次作业在 SLURM 脚本里 `source activate memory-eval` 就够。

先跑测试确认环境没问题，不需要 GPU：

```bash
python run_tests.py                     # 预期 62 tests, 0 failed
```

## 2. 下载模型权重（必须在登录节点）

**计算节点不能联网**，所以权重必须提前在登录节点下好。下载前先在浏览器确认
HuggingFace 的 repo id——很多系列同名不同规模（Qwen2.5 有 7B/14B/32B/72B），
手滑下错很常见。

```bash
pip install -U huggingface_hub
mkdir -p "$MODELS_DIR"

hf download Qwen/Qwen2.5-7B-Instruct   --local-dir "$MODELS_DIR/Qwen2.5-7B-Instruct"
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir "$MODELS_DIR/Llama-3.1-8B-Instruct"
hf download Qwen/Qwen2.5-14B-Instruct  --local-dir "$MODELS_DIR/Qwen2.5-14B-Instruct"
```

三个模型合计约 60GB。Llama-3.1 需要先在 HuggingFace 上接受许可并
`hf auth login`。

句向量模型（`vector_memory` 用的 all-MiniLM-L6-v2）只有 90MB，
`sentence-transformers` 第一次用会自动下载——但同样**必须在登录节点触发一次**，
否则计算节点上会卡在联网：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

SLURM 脚本里都有 `export HF_HUB_OFFLINE=1`，**不要删**，否则作业会卡住尝试联网
校验。

## 3. 准备数据

PIPPA 不随仓库分发，需要自己下载并重建，步骤见
[`DATASET.md`](./DATASET.md)。简要地说：

```bash
python src/ontology_extraction/fetch_pippa.py --limit 90
python src/generate/build_dialogues.py --min-user-turns 12 --max-user-turns 30
```

`build_dialogues.py` 跑完会打印实际的用户轮次分布，**每次都要看一眼这行**。原始
实验的 v1 就是因为没看，在"对话有 20 轮"的错误假设下写完了整份报告，而实际只有
6 轮、滑动窗口一次都没截断过——那次实验测的根本不是记忆。

角色事实的本体抽取需要 GPU，用 `slurm/ontology_run_extraction.slurm`。

## 4. 配置

编辑 `configs/experiment.yaml`，把三个 `path:` 改成你的实际路径。注意这几个字段
**只是说明性的**，代码按 `name` 和 `--model-port` 找服务，真正决定加载哪个权重的
是 SLURM 脚本里的 `MODELS_DIR`——两边要一致，否则你会以为换了模型其实没换。

```bash
sbatch --partition=<PARTITION> ...      # 或直接改 slurm/*.slurm 里的 #SBATCH 行
```

如果你的站点要求指定计费账号，在每个 `slurm/*.slurm` 顶部加一行：

```bash
#SBATCH --account=<ACCOUNT>
```

## 5. 冒烟测试与耗时校准

先用 `--dry-run` 在登录节点验证配置和数据，不连接任何服务，几秒钟跑完：

```bash
python src/eval/run_eval.py --config configs/experiment.yaml --dry-run
```

然后跑一次小规模校准作业，**不要跳过这步直接提交全量**：

```bash
sbatch slurm/run_eval_calibrate.slurm
```

它只跑 3 条对话、`--time=01:00:00`。跑完看日志里每条打印的 `(X.Xs)`，按
组合数比例外推全量需要的 `--time`。原始实验的全量（564 次对话生成）在 3×A100 上
实测 1 小时 45 分，但这取决于你的 GPU、并发数和 judge 的重复次数，不要直接照搬。

校准作业被时限杀掉不代表代码有问题——第一次就是这样，加大 `--workers` 的并发后
全量才降到不到两小时。

## 6. 全量跑

```bash
sbatch --export=ALL,EXTRA_ARGS="--out-dir results_v2" slurm/run_eval.slurm
```

记下返回的 job id。查状态：

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed -X
```

跑完必须检查三件事，缺一不可：

```bash
sacct -j <jobid> --format=State -X          # 要是 COMPLETED
wc -l results_v2/summary.csv                # 6 方案 × 2 模型 × 47 对话 + 表头 = 565
tail -n 15 logs/run_eval_<jobid>.out        # 末尾要有 Done，不能有 !!!!! 报错
```

日志末尾那一堆 `EngineDeadError` 是脚本跑完关闭 vLLM 服务时的正常噪声，不是失败。
但真正的失败长得也像，所以要确认有没有脚本自己打的那句报错——`run_eval.slurm`
显式检查了退出码，因为之前踩过"脚本秒退但日志末尾照样是 Done"的坑。

## 7. 统计与诊断

统计不需要 GPU，在登录节点跑就行：

```bash
python src/eval/compare_schemes.py --csv results_v2/summary.csv --out docs/COMPARE_V2.txt
python src/eval/report_tables.py --runs v2=results_v2/summary.csv --out docs/REPORT_TABLES_V2.txt
python src/eval/probe_diagnosis.py \
    --raw-dirs results_v2/raw_outputs --dialogues data/processed/dialogues.jsonl \
    --probe-csv results_v2/probe_results.csv --out docs/PROBE_DIAGNOSIS_V2.txt
```

拿结果回本地（在你自己机器上执行）：

```bash
scp <你的用户名>@<LOGIN_HOST>:<仓库路径>/results_v2/summary.csv .
```

`raw_outputs/` 含 PIPPA 对话原文，注意不要误传到公开仓库，参见 README 的数据政策。

## 8. 常见坑

- **计算节点不能联网。** 模型和句向量模型都要在登录节点先下好，`HF_HUB_OFFLINE=1`
  不要删。
- **`nvcc` 不在计算节点上。** 脚本里 `export VLLM_USE_FLASHINFER_SAMPLER=0` 是为了
  避免 FlashInfer 采样器尝试 JIT 编译而失败，不要删。
- **三个 vLLM 服务会抢同一张卡。** `slurm/run_eval.slurm` 用
  `CUDA_VISIBLE_DEVICES` 把每个进程钉在自己的 GPU 上，这是必要的。
- **僵尸 vLLM 进程。** 作业正常结束会 kill 掉服务；被 `scancel` 强杀时有小概率残留
  占着 GPU，下次提交前 `squeue -u $USER` 确认一下。
- **分区名或计费账号不对。** `sbatch` 报分区/账号无效时，按第 0 节查到的实际值改
  `slurm/*.slurm`。
- **模型下错规模。** 下载前在浏览器确认 repo id。
- **HuggingFace 数据集字段改名。** `fetch_pippa.py` 里固定了 split 名，因为
  `load_dataset("PygmalionAI/pippa", split="train")` 直接跑会报错（该数据集有三个
  非标准 split）。字段名有改动的先例，`--inspect` 这一步别跳过。
- **`relationship_accuracy` 整列为空**是预期行为，那个弱标签打分脚本没实现。

## 9. 这份指南替你确定不了的事

- 你的账号能用哪些 GPU 分区、剩余多少配额；
- 你的共享存储实际配额够不够放约 60GB 权重；
- 全量跑需要的准确 `--time`——必须自己校准（第 5 节）。

代码或流程有变化时同步更新这份文档，让它描述的始终是实际能跑起来的流程。
