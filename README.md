# Character Memory Eval

**角色扮演 AI 的记忆方案对比：原始上下文 vs 向量记忆 vs 结构化事实记忆，以及对 LLM-judge 的审核**

[English summary](./README_EN.md) · [项目报告](./docs/PROJECT_REPORT.md) · [结果与局限](./docs/RESULTS.md) · [Judge 审核](./docs/JUDGE_AUDIT.md)

长期角色扮演对话里，角色会忘设定、前后矛盾、突然"出戏"成助手腔。这个仓库在 61 段真实角色扮演对话（PIPPA）上，
用同一套脚本化的用户输入，对比三种记忆方案对"角色事实一致性"和"出戏率"的影响，
用两个开源模型（Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct）生成、Qwen2.5-14B-Instruct 做 LLM-judge，
在 SURF Snellius（vLLM + SLURM）上跑完整实验矩阵（2 模型 × 3 方案 × 61 对话 × 3 次重复）。

这个项目的重点不只是结果，而是**评测本身是否可信**：过程中抓到并修复了 4 类会静默污染结果的问题，
并对 judge 做了人工审核，发现它的精确率只有约 40%——这一点在所有结论里都已经如实标注。

## 主要结果（详见 [`docs/RESULTS.md`](./docs/RESULTS.md)）

三次重复（第一次 + seed 1 + seed 2），按对话（n=61）做配对 bootstrap，95% CI：

| 模型 | 比较 | ooc 标记率差 | 95% CI |
|---|---|---|---|
| Qwen2.5-7B | structured vs raw | −5.7 个百分点 | [−9.9, −2.3] |
| Qwen2.5-7B | vector vs raw | −2.6 个百分点 | [−4.6, −0.8] |
| Qwen2.5-7B | structured vs vector | −3.1 个百分点 | [−7.4, +0.1]（含 0，不能说 structured 更好） |
| Llama-3.1-8B | 三个 ooc 比较 | −0.3 ~ −1.1 个百分点 | 全部含 0，三种方案无可检测差异（事实一致性上 structured 比 raw 略低 0.5 个百分点，CI [−0.9, −0.1]，量级很小） |

**这些数字怎么读（务必看）：**

- 这里的"出戏率"是 judge 的**标记率**。人工审核 40 条显示 judge 精确率约 40%（95% CI 22%–61%），
  所以绝对值被高估，相对差异是否真实未经验证。
- 事实一致性的提升很小（三次平均约 +1.2 个百分点，指标接近天花板），不能说"记忆提升了事实一致性"。
- 对话短（≤20 用户轮次），`raw_context` 窗口几乎等于完整历史；structured_memory 目前只是"角色事实清单"，
  与 system prompt 里的角色卡高度重叠，实验没有区分"记住了新信息"和"再次提醒人设"。
- `evidence_traceability`（恒为 1.000）是代理指标，不是有效结果；`relationship_accuracy` 未实现。

## 仓库结构

```
configs/experiment.yaml        # 模型、记忆方案参数、judge、重复次数
docs/                          # RESULTS（结果+局限）、JUDGE_AUDIT、DATASET、SURF_GUIDE、PROJECT_REPORT
slurm/                         # 抽取、全量实验、多 seed、judge 审核/回归的作业脚本
src/ontology_extraction/       # 角色事实抽取（对齐 GOLEM 本体的四类特征），8 个 prompt
src/memory/                    # BaseMemory + 三种方案
src/generate/                  # build_dialogues.py（数据关联）、dialogue_runner.py（驱动一整段对话）
src/eval/                      # 指标、judge prompt、主入口、配对比较、judge 审核/回归脚本
tests/                         # 37 个单元测试（假 client，不依赖 GPU）
run_tests.py                   # 没装 pytest 时的测试入口
```

## 快速开始

```bash
pip install -r requirements.txt
python run_tests.py            # 或 python -m pytest tests/ ；37 个测试，不需要 GPU
```

完整实验需要 GPU（见 [`docs/SURF_GUIDE.md`](./docs/SURF_GUIDE.md)，含 SURF Snellius 上的完整步骤和踩过的坑）：

```bash
sbatch slurm/run_eval.slurm                # 全量一次（3 张 A100，约 75 分钟）
sbatch slurm/run_eval_seeds.slurm          # seed 1、2 两次重复
python src/eval/summarize_results.py       # 各方案均值
python src/eval/compare_schemes.py --csv results/summary.csv results_seed1/summary.csv results_seed2/summary.csv
```

Judge 审核（需要 judge 的 vLLM 服务，1 张卡）：

```bash
sbatch slurm/ooc_audit.slurm               # 抽样重判，导出盲评清单
python src/eval/ooc_audit_score.py         # 标注完成后算精确率 / 漏判率
sbatch --export=ALL,JUDGE_PATH=<模型路径>,TAG=<名字> slurm/ooc_regress.slurm   # 换 judge / prompt 回归
```

## 数据与许可

- 代码：MIT。
- 对话数据来自 [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA)（Apache-2.0，数据集卡标注含敏感/成人内容）。
  **本仓库不分发任何对话原文和模型生成的原始输出**；按 [`docs/DATASET.md`](./docs/DATASET.md) 自行下载并运行
  `src/ontology_extraction/fetch_pippa.py` 与 `src/generate/build_dialogues.py` 复现。
- 角色事实抽取的字段设计对齐 GOLEM 本体：Pianzola, F., Cheng, L., Pannach, F., Yang, X., & Scotti, L. (2025). The GOLEM Ontology for Narrative and Fiction. *Humanities*, 14(10), 193. https://doi.org/10.3390/h14100193。
- PIPPA：Gosling et al. 2023, *PIPPA: A Partially Synthetic Conversational Dataset*, arXiv:2308.05884。

## 已知局限

完整列表见 [`docs/RESULTS.md`](./docs/RESULTS.md) 的"不能说什么 / 限制"。要点：judge 未经充分校验；
12 个比较未做多重比较校正；样本 61 条对话、主角由启发式选出；ground truth 与 structured 记忆同源；
structured_memory 的事件/关系抽取（`on_new_turn()`）尚未实现。
