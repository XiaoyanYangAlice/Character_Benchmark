# 实验结果（v2）

全量一次运行：**47 条 PIPPA 对话 × 2 个角色模型 × 6 个记忆方案 = 564 次对话生成**
（即 `summary.csv` 的 564 行；生成的角色轮次共 13,272 个）。
原始输出在 `results_v2/`（未入库，见文末"数据与合规"），聚合指标在
`results_v2/summary.csv`（已入库，可直接复跑下面所有统计）。

v1 的结果（61 条短对话、3 个方案）已被本文档取代，保留在 git 历史和
`docs/PROJECT_REPORT.md` 第 4 节。v1 的主要缺陷与 v2 的对应改动见
[`docs/EXPERIMENT_V2.md`](./EXPERIMENT_V2.md)。

## 1. 思路

长期角色扮演对话里，用户早期说过的信息会被模型忘掉，而把历史硬塞回上下文又会
让角色"出戏"。这两件事通常被分开研究：长期记忆基准（LoCoMo、LongMemEval）只考
事后召回，角色扮演基准只看人格一致性。本实验把两者放在同一批对话、同一套用户
输入下测，看不同记忆方案在"记得住"和"不出戏"之间各自落在什么位置。

关键设计是**两个对照组**：

- `no_memory` 是下界。不加这一条，就无法说明任何方案"有用"。
- `shuffled_facts` 是内容对照。它注入的事实来自**别的角色**，格式、长度、位置与
  结构化方案完全一致，只有内容是错的。如果某个方案相对它没有优势，说明收益来自
  "又贴了一遍结构化文本"这个格式效应，而不是记忆内容正确。

## 2. 方法

**对话**：PIPPA 真实角色扮演对话，筛选后 47 条，用户发言中位数 26 轮。用户侧是
脚本化重放（固定输入），只有角色侧由被测模型生成，因此不同方案之间可配对比较。

**模型**：角色侧 Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct；judge 为
Qwen2.5-14B-Instruct。vLLM + SLURM，3×A100 各起一个服务。

**六个记忆方案**：

| 方案 | 注入内容 | 平均上下文（字符，llama / qwen） |
|---|---|---|
| `no_memory` | 仅 system prompt 里的角色卡 | 0 / 0 |
| `raw_context` | 最近 10 轮对话原文（会截断） | 2804 / 1582 |
| `vector_memory` | 句向量检索 top-5 历史轮次（all-MiniLM-L6-v2） | 729 / 599 |
| `structured_memory` | 角色卡事实 + 增量抽取事件，取**最近** 8 条 | 678 / 606 |
| `structured_hybrid` | 同上，取**最近 4 条 + 按相关度 4 条** | 682 / 611 |
| `shuffled_facts` | 对照：形式同结构化，事实来自别的角色 | 229 / 229 |

三个检索类方案的上下文预算基本相同（约 600–730 字符），`raw_context` 是它们的
2–4 倍。

**指标**：

- `probe_accuracy`（主指标，唯一直接测记忆）：71 道探针覆盖 28 条对话，全部取自
  用户第 1–10 轮的发言，在对话后段提问，由 judge 判定答对与否。
- `ooc_rate`：judge 标记的出戏率；`_late` 后缀为对话后半段。
- `fact_consistency`、`leakage_rate`：角色事实一致性与助手腔泄漏。

**统计**：按对话配对 bootstrap，20000 次重采样。Holm 校正在 (模型 × 指标) 族内
进行，共 12 族 × 12 个比较 = 144 个。族的划分在看结果之前就定了，主指标是
`probe_accuracy`。

## 3. 结果

### 3.1 探针准确率（相对 `no_memory` 的提升）

n = 28 条带探针的对话。

| 方案 | Llama-3.1-8B | Qwen2.5-7B |
|---|---|---|
| `vector_memory` | **+0.345** [+0.214, +0.476] ✓ | **+0.286** [+0.155, +0.417] ✓ |
| `structured_hybrid` | **+0.333** [+0.208, +0.464] ✓ | **+0.262** [+0.155, +0.381] ✓ |
| `structured_memory` | +0.161 [+0.054, +0.280] ✓ | +0.095 [−0.000, +0.214] — |
| `raw_context` | +0.071 [+0.012, +0.143] — | +0.060 [−0.030, +0.155] — |
| `shuffled_facts` | +0.036 [−0.036, +0.131] — | −0.048 [−0.107, +0.000] — |

✓ = Holm 校正后 p < 0.05；— = 不显著。

其他关键配对：

| 比较 | Llama | Qwen |
|---|---|---|
| `structured_hybrid` vs `vector_memory` | −0.012 — | −0.024 — |
| `structured_hybrid` vs `structured_memory` | +0.173 ✓ | +0.167 — |
| `structured_hybrid` vs `shuffled_facts` | +0.298 ✓ | +0.310 ✓ |
| `structured_memory` vs `shuffled_facts` | +0.125 — | +0.143 ✓ |
| `vector_memory` vs `raw_context` | +0.274 ✓ | +0.226 ✓ |

要点：

1. `structured_hybrid` 与 `vector_memory` 在召回上**无可检测差异**（两个模型均
   不显著），两者都显著优于 `no_memory`。
2. `shuffled_facts` 相对 `no_memory` 在两个模型上都不显著——错的事实不提供召回
   收益。因此 `structured_hybrid` 相对它的优势来自内容正确，不是格式。
3. `raw_context` 相对 `no_memory` 不显著，且显著差于 `vector_memory`。把最近
   10 轮原文塞进去并不能让模型答出第 1–10 轮说过的事。

### 3.2 出戏标记率（相对 `no_memory` 的变化，越低越好）

n = 47 条对话。正值表示标记率升高（更差）。

| 方案 | Llama | Qwen |
|---|---|---|
| `raw_context` | +0.048 ✗ | +0.177 ✗ |
| `shuffled_facts` | +0.067 ✗ | +0.074 ✗ |
| `vector_memory` | +0.006 — | +0.067 ✗ |
| `structured_hybrid` | +0.008 — | +0.045 ✗ |
| `structured_memory` | −0.000 — | +0.008 — |

✗ = Holm 校正后显著变差。`structured_hybrid` vs `vector_memory` 在两个模型上均
不显著（+0.001 / −0.022）。

要点：

1. 召回与出戏的权衡**是模型相关的**，不是记忆类型固有的。Llama 上两个检索类方
   案都没有可检测的出戏代价；Qwen 上两者都显著变差，且彼此无法区分。
2. `shuffled_facts` 在两个模型上都显著推高出戏率，召回却无提升——注入错误事实
   是纯粹的损害。
3. `raw_context` 出戏代价最大（Qwen +0.177），同时上下文开销最高、召回无提升。

`ooc_rate_late`（后半段）方向一致：仅 `raw_context` 与 `shuffled_facts` 在两个模
型上显著变差。

### 3.3 其余两个指标无结论

`fact_consistency` 与 `leakage_rate` 的全部 48 个比较在 Holm 校正后均不显著。
前者在 0.98–1.00 区间（天花板），后者接近 0（地板），本实验规模下无区分力。
不要用这两个指标下结论。

### 3.4 失败定位：答案在不在注入的上下文里

对每道探针检查标准答案是否出现在当时注入的记忆文本中，再与是否答对交叉
（`src/eval/probe_diagnosis.py`，71 道探针）：

| 模型 | 方案 | 答案在上下文 | 在·对 | 在·错 | 不在·对 | 不在·错 |
|---|---|---|---|---|---|---|
| Llama | `no_memory` | 0.0% | 0 | 0 | 3 | 68 |
| Llama | `shuffled_facts` | 0.0% | 0 | 0 | 4 | 67 |
| Llama | `raw_context` | 22.5% | 5 | 11 | 4 | 51 |
| Llama | `structured_memory` | 19.7% | 6 | 8 | 7 | 50 |
| Llama | `structured_hybrid` | 39.4% | 16 | 12 | 8 | 35 |
| Llama | `vector_memory` | 62.0% | 22 | 22 | 3 | 24 |
| Qwen | `no_memory` | 0.0% | 0 | 0 | 5 | 66 |
| Qwen | `shuffled_facts` | 0.0% | 0 | 0 | 1 | 70 |
| Qwen | `raw_context` | 16.9% | 4 | 8 | 5 | 54 |
| Qwen | `structured_memory` | 23.9% | 6 | 11 | 5 | 49 |
| Qwen | `structured_hybrid` | 45.1% | 21 | 11 | 2 | 37 |
| Qwen | `vector_memory` | 62.0% | 23 | 21 | 1 | 26 |

两个对照组的命中率都是 0.0%，这是匹配器的自检。

这张表把失败归到具体阶段：命中率低 = 写入或检索失败；"在·错"多 = 记忆给对了但
模型没用上。`structured_hybrid` 以 39–45% 的命中率达到了 `vector_memory` 在 62%
命中率下的同等准确率。

### 3.5 抽取与检索的诊断（3 方案专项运行 `results_fix`）

修复前后的抽取器与检索策略对比，用 `src/eval/inspect_extraction.py` 测量：

| 指标 | `structured_memory` | `structured_hybrid` |
|---|---|---|
| 抽取事件中"关于用户"的占比 | 57.6% / 60.7% | 58.5% / 56.3% |
| 答案抽到过但被预算挤出 | 40.8% / 36.6% | 18.3% / 16.9% |
| 平均每对话抽取事件数 | 27.0 / 21.7 | 26.1 / 22.2 |

（Llama / Qwen。两个方案共用同一个抽取器，差异只在检索策略，所以第二行单独量化
了检索预算这一个因素。）

修复前的抽取器只有约 27% 的事件是关于用户的，其余记录的是角色动作，而探针问的
全部是用户说过的内容。

## 4. 局限（必读）

- **`ooc_rate` 是 judge 的标记率，不是真实出戏率。** 审核 20 条被标记 + 20 条未标记
  显示精确率 8/20 = 40%（95% CI 22%–61%，见
  [`docs/JUDGE_AUDIT.md`](./JUDGE_AUDIT.md)）。绝对值被高估，只能做方案间相对比较。
- **那次审核的参考标注由 Claude 产生，不是本人标注。** 所以"40% 精确率"是与一位
  非人类标注者在刻意挑边界样本上的一致率。本人对同一份 40 条的独立标注、以及与之
  的 Cohen's kappa 仍未完成（见 `JUDGE_AUDIT.md` 的"待补"）。这是整份文档里最弱的
  一环。
- **审核测的 judge 配置与上面的结果不是同一个。** 审核重判的是早先带种子那几次运行
  里 qwen2.5-7b 的输出，用 OOC prompt V1、`judge_repeats=1`；而第 3 节的数字是用
  prompt V2、3 票多数产生的（见 `configs/experiment.yaml`）。产生第 3 节数字的那个
  配置，精确率尚未估计。标记率也不同：审核样本里是 7.0%，第 3 节 qwen 的 `ooc_rate`
  在 0.113–0.291 之间。
- **方案间的 OOC 比较依赖"judge 的误报与记忆方案无关"，而这一点没有验证过。** 审核
  没有按方案分层，它报出的误报计数本身也不平（raw 6、vector 4、structured 2，样本
  太小无法检验）。
- **judge 另有少量解析失败**：全部 564 次运行合计 24 次 `n_judge_fail_fact`、2 次
  `n_judge_fail_ooc`。相对 13,272 个轮次很小，但记在这里而不是留在 CSV 里。
- **只有一次运行，没有种子重复。** 同一套 v2 流程的带种子重复（`results_seed1`、
  `results_seed2`，未入库）显示绝对水平在运行间会漂动——向量检索的探针准确率在
  0.33–0.55 之间——方案间的差异则较稳定。本文档的 CI 只反映对话间变异，不含运行间
  变异。
- **"检索修复的贡献"只在一个模型上显著。** `structured_hybrid` vs
  `structured_memory` 在 Llama 上 Holm 校正后显著（+0.173），Qwen 上同向但不显著
  （+0.167，原始 p 0.039，校正后 0.248）。效应量接近，差异在检验力。
- **抽取 prompt 的贡献未经同源配对验证。** 本次运行中两个结构化方案都使用新版
  抽取 prompt，所以 prompt 改动的贡献只能跨运行比较，需打折扣。
- **探针质量未经系统清洗。** 人工抽样发现约四分之一的探针存在归属翻转（用户问
  "你的头发是红的吧"，标准答案记成用户的头发）或答案过短无信息量。这类噪声同等
  作用于所有方案，但会压低所有方案的绝对准确率。
- **命中率是下界。** 匹配用朴素子串与实词匹配，会漏判语义等价的改写。
- **`fact_consistency` 与 `leakage_rate` 在本规模下无区分力**，见 3.3。

## 5. 复现

```bash
# 统计（不需要 GPU，用已入库的 summary.csv）
python src/eval/compare_schemes.py --csv results_v2/summary.csv --out docs/COMPARE_V2.txt

# 完整重跑（需要 3×A100）
sbatch --export=ALL,EXTRA_ARGS="--out-dir results_v2" slurm/run_eval.slurm
```

诊断脚本需要 `results_v2/raw_outputs/`（未入库，需自行重跑生成）：

```bash
python src/eval/probe_diagnosis.py \
    --raw-dirs results_v2/raw_outputs --dialogues data/processed/dialogues.jsonl \
    --probe-csv results_v2/probe_results.csv --out docs/PROBE_DIAGNOSIS_V2.txt

python src/eval/inspect_extraction.py \
    --raw-dirs results_v2/raw_outputs --dialogues data/processed/dialogues.jsonl \
    --scheme structured_hybrid --samples 12 --out docs/EXTRACTION_AUDIT.txt
```

## 6. 数据与合规

`results_*/raw_outputs/`、`probe_results.csv`、`per_turn.csv` 以及
`inspect_extraction.py` 的人工核对样本都包含 PIPPA 对话原文（真实用户数据，含成人
内容），**一律不入库**。仓库中只保留聚合指标 `results_v2/summary.csv`，其中不含任
何对话文本。PIPPA 数据的获取方式见 [`docs/DATASET.md`](./DATASET.md)。
