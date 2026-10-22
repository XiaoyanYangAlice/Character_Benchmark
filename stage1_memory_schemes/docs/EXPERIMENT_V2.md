# v2 实验设计：改了什么、为什么改、怎么跑

这份文档是 v2 的总说明。v1 的结果和局限仍然保留在 [`RESULTS.md`](./RESULTS.md)
和 [`JUDGE_AUDIT.md`](./JUDGE_AUDIT.md) 里，不删除——v2 的大部分改动都是针对
v1 暴露出来的具体问题，两份放在一起才看得出改进的理由。

## 0. 一句话

v1 测的其实不是记忆（对话只有 6 轮、窗口从不截断、没有任何指标直接考记忆），
v2 把"记忆"重新变成一个真问题：对话加长到会截断、加两个对照组区分"记住了"
和"又贴了一遍人设"、加一个直接考记忆的探针指标、把 judge 的逐轮判定存下来
以便审核。

## 1. v1 的五个问题，以及 v2 的对应改动

### 问题 1：对话太短，记忆没有压力（最严重）

v1 的 `build_dialogues.py` 默认只取每段对话的**前 6 个用户发言**，而
`raw_context` 的窗口是 20 轮——窗口一次都没有截断过。也就是说 raw_context
拿到的就是完整历史，三种方案比的根本不是"谁记得住"，而是"完整历史 / 检索
片段 / 事实清单，这三种提示词写法哪种让模型更像角色"。这是整个 v1 实验最
根本的问题，而且我在写完报告之后才发现（文档里一直按配置上限写成"≤20 轮"）。

**v2 改动**
- `build_dialogues.py`：`--max-user-turns` 6 → 30，新增 `--min-user-turns 12`，
  短对话直接剔除，并在结束时打印用户轮次分布。
- `configs/experiment.yaml`：`raw_context.max_turns` 20 → 10（5 个来回）。
  窗口必须小于对话长度，截断才会发生。
- `run_eval.py` 开头会打印实际的用户轮次分布和窗口大小。**每次重跑都要看
  这一行**——v1 就是因为没看这个数字，在错误假设下写了整份报告。

代价：PIPPA 里够长的对话变少，样本量会从 61 下降。样本少但能测出东西，比
样本多但测的是别的东西好。如果筛出来不足 40 条，需要先扩大上游抽取规模
（见第 4 节）。

### 问题 2：无法区分"记住了新信息"和"又提醒了一遍人设"

system prompt 里本来就有完整角色卡，而 `structured_memory` 注入的事实就是
从同一张角色卡抽出来的。所以"structured 的出戏标记率比 raw 低 5.7 个百分点"
有一个平凡解释：它等于把人设又说了一遍，而 raw 塞的是大段对话历史，把角色卡
的相对权重稀释了。v1 没有对照能排除这个解释。

**v2 改动**：加两个对照方案（`src/memory/no_memory.py`、
`src/memory/shuffled_facts.py`）

| 方案 | 注入内容 | 用来回答 |
|---|---|---|
| `no_memory` | 什么都不注入（只有 system prompt 里的角色卡） | 下界：不给任何历史会怎样 |
| `shuffled_facts` | **别的角色**的事实清单，格式/长度与 structured 相同 | 起作用的是内容，还是"又出现了一段人设风格的文本" |

读法：
- `structured` 明显优于 `shuffled_facts` → 事实内容本身起作用，是真的记忆效应。
- `structured ≈ shuffled_facts`，两者都优于 `no_memory` → 效应来自提示词里人设
  的占比，**v1 的结论必须改写成提示词工程效应**。
- `shuffled_facts` 反而更差 → 错误记忆会污染生成（这本身也是有价值的发现：
  记错比不记更糟）。

`shuffled_facts` 的捐赠者由 `pick_donor()` 确定性选取（按 dialogue_id 哈希，
优先挑事实条数接近的），保证每次重跑、每个 seed 配到的都是同一个捐赠者。

### 问题 3：没有任何指标在直接测记忆

`ooc_rate` 和 `fact_consistency` 都只测"像不像这个角色"。一个完全没有记忆的
模型，只要角色卡够详细，这两个指标都能拿高分——v1 里 Llama 的 fact_consistency
就是 1.000。`evidence_traceability` 恒为 1.000，是个代理指标。
`relationship_accuracy` 没实现。换句话说，一个关于记忆的项目，没有一个指标
在测记忆。

**v2 改动**
- **记忆探针**（`src/generate/make_probes.py`）：用户在对话早期说过的具体信息
  （名字、数字、地点、承诺），到对话末尾再问一次，看角色答不答得出来。
  题目只从**用户发言**里出——`scripted_user_turns` 在所有方案间完全相同，
  所以对所有方案是同一套题，可比；从角色回复里出题就不可比了。
  生成的每道题都要通过 `validate_probe()` 校验（答案必须真的出现在它声称的
  那条用户发言里），防止模型编出用户没说过的"标准答案"。
  新指标 `probe_accuracy`，**这是唯一直接测记忆的指标，报告里优先看它**。
- **`evidence_traceability` 废弃，换成 `evidence_use_rate`**：检查回复里是否
  真的出现了检索到的内容词（去掉虚词）。仍是词面重合、不等于语义上用到了，
  但至少不是恒为 1。`raw_context`/`no_memory` 没有检索语义，这一列留空
  （`supports_evidence = False`），不是 0。
- **前段/后段分层**：`ooc_rate_early` / `ooc_rate_late` 等。记忆方案的差别
  应该主要出现在后段，前段几乎没有历史可记。如果差异在前后段一样大，那它
  多半不是记忆带来的。

### 问题 4：`structured_memory` 名不副实

v1 的 `on_new_turn()` 是空实现，事件和关系两段永远为空。所谓"结构化事件记忆"
实际上只是"开局导入一次角色卡事实"，对话里新出现的信息一律不进记忆。

**v2 改动**：实现 `on_new_turn()`。每完成一个"用户发言 + 角色回复"的来回，
调用一次抽取器（judge 模型，temperature 0）抽出"本轮新确立的具体信息"，
写进 `self.events`，后续检索时带出来。按来回抽取而不是每轮抽一次，是因为
指代经常跨越两边（用户说"我妹妹叫 Anna"、角色回"Anna 这名字真好听"），
顺带把 LLM 调用次数减半。

抽取失败会计数（`n_extraction_failures` 列）而不是静默跳过——"事件记忆全程
为空却没人发现"正是 v1 那类静默失败的形态。想做消融就把
`eval.event_extraction` 设成 false 跑一遍，对比"有没有增量抽取"的差别。

注意：这让 structured 方案比别的方案多消耗一个模型。这是**方案本身的成本**
（结构化记忆需要一个抽取器），不是评测上的不公平，但报告里要写清楚。

### 问题 5：judge 审核和主实验之间隔着一层

v1 只存了每条对话的汇总 `ooc_rate`，没存逐轮判定，所以审核时必须用同一套
配置重判 600 轮。judge 温度 0.3，重判和主实验那次不是同一批判定。而且审核
没有按方案分层（被标的 20 条里 raw 6、vector 4、structured 2），无法回答最
关键的问题：**judge 的误报率是不是跟记忆方案有关？**如果 judge 对 raw 的回复
更容易误报，那"structured 标记率更低"就可能完全是 judge 的偏差。

**v2 改动**
- `run_eval.py` 把每个角色轮次的判定写进 `results/per_turn.csv`：逐次投票、
  理由、规则检测结果、上下文长度都在里面。审核直接从这里抽样
  （`src/eval/ooc_audit_stratified.py`），审的就是主实验真正用过的判定，
  而且**不需要 GPU**。
- 审核按方案分层抽样，`ooc_audit_score.py` 分别给出各方案的精确率和置信区间。
- `judge_repeats` 1 → 3。审核发现系统性误报压不下去，但多数投票至少能压掉
  采样噪声那一部分。
- 主实验改用 V2 prompt（`ooc_prompt_version: v2`）。V2 在 v1 的 40 条回归集上
  精确率没有提升，但那是样本内测试；v2 用全新的分层审核重新估。
- **新增规则型检测器**（`detect_assistant_leakage`）：v1 审核发现真正的出戏
  几乎全是安全说教腔、通用拒绝腔和模板残留。既然集中在这么窄的一类现象上，
  就可以用确定性正则直接数出来，作为 LLM judge 的独立交叉验证——规则没有
  judge 那种"把符合设定的行为当出戏"的误报，代价是只覆盖这一类、必然漏判。
  两个指标一起报告：**规则检测率给真实出戏率的下界，judge 标记率给上界**。

### 顺带修掉的

- `compare_schemes.py` 内置 Holm 多重比较校正（v1 是事后补的）和 bootstrap p。
- 每轮注入的上下文长度记进 `context_chars`，`compare_schemes.py` 会先打印
  各方案的平均上下文长度——先检查预算是否可比，再看结果。预算差太多时，
  任何方案间差异都可能只是"上下文长度"的效应。
- `run_eval.slurm` 加了退出码检查（v1 踩过"脚本报错但仍打印 Done"的坑）。

## 2. 跑的顺序

```bash
# 0) 本地先跑测试（不需要 GPU，55 个测试）
python run_tests.py

# 1) 重新抽取更多、更长的故事（v1 只有 62 个故事，筛 ≥12 轮后会不够用）
#    注意 --min-turns 指的是 PIPPA 原对话的消息总数，不是用户轮次数
python src/ontology_extraction/fetch_pippa.py --output-dir data/raw_stories --min-turns 60 --limit 400
sbatch slurm/ontology_run_extraction.slurm        # 角色事实抽取，按新规模可能要几小时

# 2) 重建对话数据（长对话）
python src/generate/build_dialogues.py \
    --facts data/processed/character_features_en.jsonl \
    --output data/processed/dialogues.jsonl \
    --max-user-turns 30 --min-user-turns 12
#    ⚠ 看一眼打印出来的"用户轮次分布"，中位数应该在 15 以上；
#    如果筛出来不足 40 条对话，回到第 1 步把 --limit 调大再来一遍。

# 3) 生成记忆探针（1 张卡，约 20 分钟）
sbatch slurm/make_probes.slurm
#    ⚠ 跑完抽查几条 memory_probes，确认题目真的能从用户发言里答出来

# 4) 校准：先跑 3 条对话，拿实测耗时再决定全量的 --time
sbatch slurm/run_eval_calibrate.slurm
#    用实测单条耗时 x 总次数（对话数 x 2 模型 x 5 方案）估算，改 run_eval.slurm 的 --time

# 5) 全量 + 重复
sbatch slurm/run_eval.slurm                       # 第一次（不固定 seed）
sbatch slurm/run_eval_seeds.slurm                 # seed 1、2

# 6) 统计
python src/eval/compare_schemes.py \
    --csv results/summary.csv results_seed1/summary.csv results_seed2/summary.csv \
    --out results/compare_v2.txt
python src/eval/supplementary_stats.py \
    --dialogues data/processed/dialogues.jsonl \
    --csv results/summary.csv results_seed1/summary.csv results_seed2/summary.csv \
    --logs "logs/*.out" --out results/supplementary_stats.txt

# 7) 分层审核（不需要 GPU，登录节点直接跑）
python src/eval/ooc_audit_stratified.py \
    --per-turn results/per_turn.csv results_seed1/per_turn.csv \
    --raw-dirs results/raw_outputs results_seed1/raw_outputs \
    --dialogues data/processed/dialogues.jsonl \
    --model qwen2.5-7b --schemes raw_context structured_memory \
    --out-dir results/ooc_audit_v2
# 人工标注 results/ooc_audit_v2/labels_template.csv 之后：
python src/eval/ooc_audit_score.py --audit-dir results/ooc_audit_v2 \
    --labels results/ooc_audit_v2/labels_user.csv
```

## 3. 结果出来之后，先看什么

按这个顺序读，不要先看 `ooc_rate`：

1. **`run_eval.py` 开头打印的用户轮次分布**。中位数 < 12 就说明筛选没生效，
   后面的结果和 v1 一样没有意义。
2. **`compare_schemes.py` 开头的各方案平均上下文长度**。某个方案是别人的
   2 倍以上时，方案间差异可能只是上下文长度的效应，要在报告里说明。
3. **`probe_accuracy`**：唯一直接测记忆的指标。
4. **`structured_memory vs shuffled_facts`**：不显著就说明效应来自"又贴了一遍
   人设"，这时 `structured vs raw` 的差异**不能**解释成记忆更好。
5. **`ooc_rate_late` vs `ooc_rate_early`**：差异只出现在后段，才像是记忆效应。
6. **`leakage_rate`（规则）vs `ooc_rate`（judge）**：两者的差距就是 judge 误报
   的量级。报告里把规则值作为下界、judge 值作为上界一起给。
7. 最后才是 `ooc_rate`，并且必须写成"judge 的标记率"。

## 4. 已知仍未解决的

- **ground truth 与 structured 记忆同源**：事实清单和 fact_consistency 的标准
  答案都来自同一份角色卡抽取。`shuffled_facts` 对照能部分缓解（它用错误事实，
  理论上应该推高矛盾率），但没有独立标注的 ground truth。
- **指代消解噪声**：抽取仍是四次独立 LLM 调用，代词被当成独立伪角色的问题
  还在（见 `DATASET.md`）。
- **主角选择仍是启发式**（事实条数最多的角色）。
- **judge 与 Qwen agent 同系**，可能偏向同系回复，仍未单独检验。换一个非
  Qwen 系的强 judge 需要先在集群上下载模型。
- **探针覆盖率**取决于 PIPPA 用户发言的信息密度，可能只覆盖部分对话。覆盖率
  低不是 bug，报告里按"有探针的对话"这个子集报探针指标，并写清子集大小，
  **不要为了凑数放宽 `validate_probe` 的校验**。
- **PIPPA 数据污染**：无法排除模型训练时见过 PIPPA，未检验。对所有方案同等
  影响，但会影响绝对数值。
