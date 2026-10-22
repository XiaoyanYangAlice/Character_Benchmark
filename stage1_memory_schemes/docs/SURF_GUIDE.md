# SURF (Snellius) 逐步操作指南——从登录开始

这份指南对应「按方案(1)走：先对角色卡文本跑角色本体抽取，建 ground truth 事实库」
这条路径，覆盖从登录 Snellius 到拿到 `character_features_en.jsonl` 的完整链路。
每一步都给出实际命令，照着敲就行；标 **[需要你确认]** 的地方本指南给不出答案，
执行前自己查一下。

> **数据语言更新**：实测导出的 PIPPA 角色描述是纯英文（角色卡本身就是英文写
> 的），所以下面第 6/7/9 步都改成 `--lang en`（对应 v4 英文版 prompt），产出
> `character_features_en.jsonl`。原计划的中英双语一致性对照（`--compare-with`）
> 需要同一批角色卡有平行的中文文本才有意义，PIPPA 没有，所以暂时跳过；如果以后
> 补充中文语料（SillyTavern 中文角色卡等），可以恢复这一步。

---

## 0. 开始前确认这些事

- [ ] 你有 SURF 账号（一般通过所在机构/项目的 SURF 计算时间申请获得）。
- [ ] 本地机器上已经生成 SSH key 并加进了 SURF 账号（SURF 门户 → SSH keys）。
- [ ] **[需要你确认]** 你的计费账号名（`accinfo` 会显示，登录后第一步就能查到）。
- [ ] **[需要你确认]** 你的账号能用哪个 GPU 分区（`gpu_a100` 还是 `gpu_h100`，
      或者更小的 MIG 切片分区），用 `sinfo` 查。

---

## 1. 登录 Snellius

```bash
ssh <你的用户名>@snellius.surf.nl
```

首次登录会提示确认 host key，正常输入 `yes` 即可。登录后默认停在 `$HOME`，这个
目录配额很小，**代码放这里，但模型权重和数据集不要放这里**（放
`/scratch-shared/$USER/`，见下）。

## 2. 查账号和配额

```bash
accinfo                 # 看你名下的计费账号、剩余 GPU 小时
sinfo                   # 看有哪些分区可用（gpu_a100 / gpu_h100 / 是否有更小的 MIG 分区）
myquota                 # 看 $HOME 和 /scratch-shared 的存储配额（不同系统命令名可能略有差异，
                         # 具体以 SURF 文档/`module load` 后 `--help` 为准）
```

把 `accinfo` 看到的计费账号名记下来，后面 SLURM 脚本里如果要求指定
`#SBATCH --account=<计费账号>`，加上这一行（本仓库的 slurm 脚本目前没写死这行，
如果你的账号强制要求指定，需要自己加）。

## 3. 把代码传上去

在**本地机器**（不是 Snellius 上）执行，把仓库传到 Snellius 的 `$HOME` 下：

```bash
# 方式一：如果仓库已经推到 GitHub
ssh <你的用户名>@snellius.surf.nl
git clone https://github.com/<你的用户名>/character-memory-eval.git
cd character-memory-eval

# 方式二：还没推 GitHub，先用 scp/rsync 直接传
rsync -avz --exclude 'data/raw*' --exclude '__pycache__' \
    ./character-memory-eval/ <你的用户名>@snellius.surf.nl:~/character-memory-eval/
```

后面所有 `sbatch`、`python` 命令都假设你的工作目录是 `~/character-memory-eval`。

## 4. 建 conda 环境（登录节点上执行）

```bash
module purge
module load 2025
module load Anaconda3/2025.06-1
conda create -n memory-eval python=3.11 -y
source activate memory-eval
pip install -r requirements.txt        # vllm + openai
```

这一步只做一次，之后每次跑作业在 SLURM 脚本里 `source activate memory-eval` 就够。

## 5. 下载模型权重（必须在登录节点，计算节点不能联网）

先在浏览器打开对应 HuggingFace 页面确认 repo id 和模型规模（很多系列同名不同
规模，容易下错）：

```bash
pip install -U huggingface_hub  # 装新版 hf 命令行工具
mkdir -p /projects/prjs1995/models

hf download Qwen/Qwen2.5-14B-Instruct \
    --local-dir /projects/prjs1995/models/Qwen2.5-14B-Instruct
```

> 权重放 `/projects/prjs1995`，不是 `/scratch-shared/$USER`：实测
> `/scratch-shared/$USER` 对这个账号没有分配个人目录（`df` 报 "No such file or
> directory"），而 `/projects/prjs1995` 已确认可用且空间充足。如果你的账号
> `/scratch-shared` 确实可用，用哪个都行，但要和 `slurm/*.slurm` 里的
> `MODEL_PATH` 保持一致。

> 方案(1)只需要一个抽取用的模型（建议 Qwen2.5-14B-Instruct 起步）。后面跑主项目
> 的生成实验时，再下 Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct 和判断器模型，
> 步骤完全一样，只是换 repo id，都放进 `/projects/prjs1995/models/`。

**[需要你确认]** `/projects/prjs1995` 的个人/项目配额够不够放下这些模型（一个
14B 模型 bf16 权重大约 28GB）——`df -h` 只显示整个文件系统的空闲量，不等于你
的配额上限，具体配额用 `myquota` 或类似命令确认。

## 6. 准备数据：把角色卡文本转成编号句子格式

数据来源是 PIPPA（HuggingFace 数据集，主数据源）。没有 SillyTavern 角色卡的话
（`cards_to_stories.py` 保留在仓库里，之后想补中文语料时可以用），直接走 PIPPA
这一条路径就够。

```bash
pip install datasets
hf auth login   # 需要先在 https://huggingface.co/datasets/PygmalionAI/pippa
                # 登录网页版接受使用条款，再用这里生成的 token 登录命令行

cd src/ontology_extraction
python fetch_pippa.py --inspect                       # 第一次先看真实字段名
python fetch_pippa.py --output-dir ../../data/raw_stories --min-turns 15 --limit 90
cd ../..
```

`--inspect` 这一步不能跳过：HuggingFace 上数据集字段名有改动的先例，脚本里
`bot_description`/`conversation` 是按常见命名写的默认值，真实字段名不一致时用
`--description-field`/`--conversation-field` 指定。

> **实测踩坑**：`load_dataset("PygmalionAI/pippa", split="train")` 直接跑会报
> `datasets.table.CastError`——PIPPA 的 HuggingFace 仓库里其实有三个 jsonl 文件
> （`pippa.jsonl`/`pippa_deduped.jsonl`/`pippa_metharme.jsonl`），字段不完全一致，
> 默认会尝试把三个拼成一个 train split 从而报错。`fetch_pippa.py` 里已经固定成
> `load_dataset("PygmalionAI/pippa", data_files="pippa.jsonl", split="train")`
> 只读主文件，不需要再改。
>
> **实测结果**：导出的 90 个角色描述是纯英文的（PIPPA 的角色卡本身就是英文），
> 所以下面第 7/9 步统一用 `--lang en`。

如果之后想补充中文语料（比如手动收集 SillyTavern 中文角色卡），转换方式是：

```bash
cd src/ontology_extraction
python cards_to_stories.py --input-dir ../../data/raw_cards \
    --output-dir ../../data/raw_stories
cd ../..
```

`.txt` 最终都会落在 `data/raw_stories/`，下一步统一处理，不用区分来源
（`prepare_stories.py` 不关心文件从哪来）。

```bash
source activate memory-eval
python src/ontology_extraction/prepare_stories.py \
    --input-dir data/raw_stories \
    --output data/processed/stories.jsonl \
    --min-sentences 5
```

跑完检查一下 `data/processed/stories.jsonl` 的行数和内容，确认句子切分合理
（中文长句子有没有被切碎、对话有没有被误合并）。

## 7. 冒烟测试：先跑 2 个故事

```bash
sbatch slurm/ontology_smoke_test.slurm
squeue -u $USER          # 看作业排队/运行状态
```

作业跑完后（`squeue` 里看不到了，或者状态变成 CD/CG），检查日志和输出：

```bash
tail -50 logs/smoke_*.out                       # 看有没有报错
cat data/processed/character_features_en_smoke.jsonl | head -20
```

**人工检查这三件事**（对应主项目文档里"先冒烟测试再放量"的纪律）：
1. `span` 是不是完整、合理的证据片段，不是被切得七零八落；
2. 同一个故事里的 `character` 字段有没有把"她""姜医生""姜遥"正确归并成
   同一个规范角色名；
3. `modality` 是不是只有 `present`——如果全是 `present`，说明模型没识别出
   `past`/`potential`，需要人工看几条原文核实是 prompt 的问题还是模型能力问题。

如果发现问题，改 `temperature`（`extract_character_features.py` 里
`call_model` 函数的默认参数）或者换更大的模型，重新跑冒烟测试，**不要在没验证
过的情况下直接放量**。

## 8. 记录单次调用耗时，估算全量时间

冒烟测试日志里能看到 2 个故事 × 4 类别 = 8 次调用的总耗时，除以 8 得到单次调用
平均耗时，乘以你实际故事数 × 4，估算全量需要多久，用来设 `run_extraction.slurm`
里的 `--time`（目前脚本里写的是 6 小时，样本量大或模型更大时可能不够）。

## 9. 全量跑 + 质检报告

冒烟测试通过后：

```bash
sbatch slurm/ontology_run_extraction.slurm
squeue -u $USER
```

跑完后：

```bash
cat logs/validation_report.txt
```

报告会列出角色数量异常的故事（可能是角色名归并出问题）、modality 分布等基础质检
信息（`basic_report`）。当前只有英文一个语言版本，没有平行的中文数据，所以不跑
`--compare-with` 双语一致性对照；如果以后补了中文语料，可以在
`ontology_run_extraction.slurm` 里恢复那一段（脚本注释里留了说明）。

## 10. 把结果拿回本地 / 接入主项目

```bash
# 本地机器上执行
scp <用户名>@snellius.surf.nl:~/character-memory-eval/data/processed/character_features_en.jsonl \
    ./data/processed/
```

`character_features_en.jsonl` 就是 `src/memory/structured_memory.py` 里
`load_character_facts()` 要读的文件，也是 `docs/DATASET.md` 里 `fact_labels`
字段的数据来源。

## 11.（下一阶段）关联对话数据，产出 dialogues.jsonl

角色本体抽取只是"建 ground truth 事实库"这一条线，到这里就完成了。下一阶段是
用这批事实 + PIPPA 原始对话跑三种记忆方案的对比实验，第一步是产出
`data/processed/dialogues.jsonl`（结构见 `docs/DATASET.md`）：

```bash
# 还在 SURF 登录节点，src/ontology_extraction 同一层
cd src/generate
python build_dialogues.py --facts ../../data/processed/character_features_en.jsonl --output ../../data/processed/dialogues.jsonl
cd ../..
```

不需要重新下载数据——复用的是第 6 步 `fetch_pippa.py` 已经建好的本地 HuggingFace
缓存。跑完检查一下：

```bash
wc -l data/processed/dialogues.jsonl
head -1 data/processed/dialogues.jsonl | python -m json.tool | head -20
```

按 `build_dialogues.py` 文件头写的，**抽样核对几条 `character_card.name` 是不是
真的对应这个故事该演的角色**（主角选择用的是"事实条数最多"的启发式，不是精确
方法，见 `docs/DATASET.md`"已知局限"一节）。

## 12.（下一阶段）跑 `run_eval.py` 主实验矩阵

`run_eval.py` 现在已经实现完了（读 `configs/experiment.yaml`，按
model × memory_scheme × dialogue 三层循环跑生成 + 四个指标，落盘
`results/summary.csv` + `results/raw_outputs/`）。真跑之前先在登录节点用
`--dry-run` 验证一遍配置和数据本身没问题（不连接任何 vLLM 服务，几秒钟跑完）：

```bash
python src/eval/run_eval.py --config configs/experiment.yaml --dry-run
```

真跑需要 **同时起 3 个 vLLM 服务**（`configs/experiment.yaml` 里两个模型
Qwen2.5-7B/Llama-3.1-8B 各一个，judge 模型 Qwen2.5-72B 一个），各占一个端口，
跑法参考 `slurm/ontology_run_extraction.slurm` 起单个服务的写法（health-check
循环 + `VLLM_USE_FLASHINFER_SAMPLER=0` 那几行照抄）。

judge 已经从 Qwen2.5-72B 换成复用抽取阶段下载过的 Qwen2.5-14B-Instruct（够用：
judge 只需要做几个简单的 JSON 判断，不需要 72B 的能力），`configs/experiment.yaml`
的 `judge_model` 已经同步改好。三个服务（两个角色模型 + judge）各占一张卡，
合计申请 3 张 A100，比原计划的 4-6 张风险小很多。

`slurm/run_eval.slurm` 已经写好，起 3 个 vLLM 服务（`CUDA_VISIBLE_DEVICES`
把每个进程钉在自己的 GPU 上，避免抢卡），健康检查通过后跑
`src/eval/run_eval.py`，跑完自动 kill 掉三个 vLLM 进程。

⚠ **提交全量作业前先校准一次耗时**——`slurm/run_eval.slurm` 里
`--time=04:00:00` 是按 61 条对话 × 2 模型 × 3 记忆方案 × 每条约 18 次 LLM
调用粗估的，没有实测校准过。建议先复制一份改成小规模验证：

```bash
cp slurm/run_eval.slurm slurm/run_eval_calibrate.slurm
```

把最后 `python src/eval/run_eval.py ...` 那几行加上 `--limit 5`（只跑 5 条
对话），`--time` 改成 `00:30:00` 就够，提交：

```bash
sbatch slurm/run_eval_calibrate.slurm
```

跑完看 `logs/run_eval_<jobid>.out` 里每条打印的 `(X.Xs)` 耗时，乘以
`366 / 5`（全量组合数 / 校准跑的组合数）估算全量需要的 `--time`，改回
`slurm/run_eval.slurm` 里对应的值，再提交全量：

```bash
sbatch slurm/run_eval.slurm
```

跑完检查：

```bash
wc -l results/summary.csv           # 应该是 366 行数据 + 1 行表头
head -5 results/summary.csv
ls results/raw_outputs/ | wc -l     # 应该也是 366 个 json 文件
```

`relationship_accuracy` 这一列现在全部会是空的——`relationship_trajectory`
弱标签打分脚本还没做（见 `docs/DATASET.md`），这是预期行为，不是
`run_eval.py` 跑错了。

---

## 常见坑（复用之前项目踩过的坑）

- **计算节点不能联网**：模型必须在登录节点用 `hf download` 提前下好，
  SLURM 脚本里已经加了 `export HF_HUB_OFFLINE=1`，不要删掉，否则作业会卡住尝试
  联网校验。
- **忘记 `kill $VLLM_PID`**：本仓库的 SLURM 脚本已经在最后加了这一行，正常情况
  下服务会随作业结束自动关闭；如果作业是被 `scancel` 强制杀掉的，有小概率
  vLLM 子进程没清理干净，占着 GPU——下次提交前 `squeue -u $USER` 确认没有僵尸
  作业。
- **`/scratch-shared/$USER` 不存在**：不是所有账号都自动分配了 `/scratch-shared`
  个人目录（这个项目实测就没有），别死等，直接确认能用的存储路径（比如
  `/projects/<你的项目号>`）就切过去，模型权重 + 数据加起来可能不小，下载前
  务必先查配额。
- **模型下错规模**：下载前一定先在浏览器确认 HuggingFace repo id，很多系列
  （比如 Qwen2.5）同名不同规模（7B/14B/32B/72B），容易手滑下错。
- **`--partition` 和 `--account` 不对**：如果 `sbatch` 报错提示分区或账号无效，
  按第 2 步 `sinfo`/`accinfo` 的实际结果改 `slurm/*.slurm` 里对应的行。

## 这份指南没法替你确定的事

- 你的 Snellius 账号具体能用哪些 GPU 分区、剩余配额——需要你自己用
  `accinfo`/`sinfo` 查；
- `/scratch-shared` 实际配额是否够放模型——同上；
- 全量跑需要的准确 `--time`——需要你在冒烟测试后自己估算（见第 8 步）。

代码或流程有变化时，把这份指南对应更新，保持"文档描述的就是实际能跑起来的
流程"。
