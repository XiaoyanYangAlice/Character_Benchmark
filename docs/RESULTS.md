# 实验结果（全量，61 条 PIPPA 对话 × 2 模型 × 3 记忆方案 = 366 次生成）

作业 27547775，`results/summary.csv`。judge：Qwen2.5-14B-Instruct。
结果是在修复三个会静默污染结果的问题之后得到的（见文末），之前几次全量跑的结果已作废。

## 第一次全量的均值（`src/eval/summarize_results.py`，仅第一次，未固定种子）

| 模型 | 记忆方案 | fact_consistency ↑ | ooc_rate ↓ |
|---|---|---|---|
| llama3.1-8b | raw_context | 1.000 | 0.027 |
| llama3.1-8b | vector_memory | 0.997 | 0.022 |
| llama3.1-8b | structured_memory | 0.995 | 0.025 |
| qwen2.5-7b | raw_context | 0.962 | 0.107 |
| qwen2.5-7b | vector_memory | 0.992 | 0.098 |
| qwen2.5-7b | structured_memory | 0.986 | 0.044 |

## 三次重复的配对比较（`src/eval/compare_schemes.py`）

三次重复 = 第一次全量（未固定种子）+ seed 1 + seed 2。同一 (模型, 方案, 对话) 先对三次取平均，再按对话（n=61）做配对 bootstrap，95% CI。差值 = A − B；ooc_rate 为负表示 A 出戏更少。

| 模型 | 指标 | A vs B | 均值差 | 95% CI | 三次单独的均值差 |
|---|---|---|---|---|---|
| qwen2.5-7b | ooc_rate | structured vs raw | −0.057 | [−0.099, −0.023] | −0.063 / −0.067 / −0.041 |
| qwen2.5-7b | ooc_rate | vector vs raw | −0.026 | [−0.046, −0.008] | −0.008 / −0.046 / −0.022 |
| qwen2.5-7b | ooc_rate | structured vs vector | −0.031 | [−0.074, +0.001] | −0.055 / −0.020 / −0.019 |
| qwen2.5-7b | fact_consistency | structured vs raw | +0.012 | [+0.000, +0.027] | +0.025 / +0.008 / +0.003 |
| qwen2.5-7b | fact_consistency | vector vs raw | +0.013 | [+0.002, +0.028] | +0.030 / +0.008 / +0.000 |
| qwen2.5-7b | fact_consistency | structured vs vector | −0.001 | [−0.005, +0.004] | −0.005 / +0.000 / +0.003 |
| llama3.1-8b | ooc_rate | structured vs raw | −0.011 | [−0.026, +0.004] | −0.003 / +0.000 / −0.030 |
| llama3.1-8b | ooc_rate | vector vs raw | −0.008 | [−0.021, +0.004] | −0.005 / +0.000 / −0.019 |
| llama3.1-8b | ooc_rate | structured vs vector | −0.003 | [−0.015, +0.010] | +0.003 / +0.000 / −0.011 |
| llama3.1-8b | fact_consistency | structured vs raw | −0.005 | [−0.009, −0.001] | −0.005 / +0.000 / −0.009 |
| llama3.1-8b | fact_consistency | vector vs raw | −0.003 | [−0.007, +0.000] | −0.003 / −0.005 / +0.000 |
| llama3.1-8b | fact_consistency | structured vs vector | −0.002 | [−0.007, +0.003] | −0.003 / +0.005 / −0.009 |

## 能说什么

- 在 Qwen2.5-7B 上，注入角色事实清单（structured_memory）使 judge 的出戏标记率平均降低约 5.7 个百分点（judge 精确率仅约 40%，见限制 6，不等于真实出戏率下降）（三次重复平均，配对 bootstrap 95% CI [−9.9, −2.3]，三次单独看方向一致）。
- 在 Qwen2.5-7B 上，向量记忆也有较小的出戏率降低（−2.6 个百分点，CI [−4.6, −0.8]）。
- 在 Llama-3.1-8B 上三种方案没有可检测的差异；该模型出戏率和事实一致性已接近天花板。

## 不能说什么 / 限制

1. **structured_memory 目前只是"角色事实清单"**。`on_new_turn()` 尚未实现，事件和关系两段始终为空，不能称为完整的事件/关系记忆。
2. **evidence_traceability（structured 与 vector 均为 1.000）不是有效结果**。它只检查本轮检索结果是否非空（代理指标），非空即为 1，不衡量回复是否真的用了证据。
3. **relationship_accuracy 为空**：`relationship_trajectory` 弱标注未实现。
4. **探索性分析**：12 个比较没有做多重比较校正，没有计算校正后的区间。
5. **采样次数有限**：每个条件每条对话生成 3 次（温度 0.7，第一次未固定种子，另两次 seed 1、2），judge 温度 0.3 且未固定种子，judge 噪声包含在差异里。
5b. **以下结论不成立**：「structured 优于 vector」（CI [−7.4, +0.1] 含 0，且差异在三次里逐次缩小）；「记忆提升事实一致性」（第一次的 +3.0 个百分点在三次平均后只剩约 +1.2，三次单独看分别为 +3.0 / +0.8 / 0.0，由个位数对话贡献，且该指标接近天花板）。
6. **OOC judge 不准，且只做过一次小样本审核**（`src/eval/ooc_audit_sample.py` / `ooc_audit_score.py`，清单与标注在 `results/ooc_audit/`）。做法：从 qwen2.5-7b 三次生成里随机抽 600 个角色回复，用主实验同一 judge 配置重判（标记率 7.0%，42/600），再取被标出戏的 20 条和没被标的 20 条，打乱、隐去 judge 判定后逐条标注。
   - 结果：**精确率 40%**（8/20，95% Wilson CI 22%–61%）；漏判 1/20（5%，CI 1%–24%）；据此估计召回率约 38%（量级参考，分母只有 1 条漏判，非常不稳）。
   - **标注者是 Claude，不是人工**，所以这里的数字是"Claude 与 judge 的一致率"，不是人工准确率；Claude 的标注本身也不是真值。边界样本（item 17 通用拒绝腔、item 23 输出里混入 `assistant` 模板残留、item 35 淫荡设定被改成"纯洁游戏"）若全部改判为不出戏，精确率降到 30%，漏判 0%。需要人工复核同一份清单才能把它升级成"人工校验"。
   - 错误模式：12 个误报里多数是 judge 把"合设定的行为"当成出戏——角色提到 programming/mermaid cakes、角色语气合理软化、列点回答角色自己提议的内容、旁白型角色卡里的 "User:" 标签；真正被抓到的出戏几乎全是"安全拒绝/说教腔"（Mia 设定为施虐、无视 safeword，却回复"Safety always comes first"，共 5 条，另有 M16A1 的通用拒绝与一条 assistant 模板残留）。
   - 含义：**ooc_rate 的绝对值不能当"出戏率"解读**（约六成的标记并非出戏）；上面"structured 降低出戏率 5.7 个百分点"只能说成"降低了 judge 的标记率"。相对差异是否反映真实出戏变化，取决于 judge 的误报是否与记忆方案无关，这次审核没有按方案分层，无法验证（误报按方案：raw 6、vector 4、structured 2，样本太小，不能下结论）。
   - **只改 prompt 修不好**（`ooc_judge_regress.py`，同一份 40 条标注，每条判 5 次）：V2 prompt（保守判定、先理由后结论、明确列出常见误报类型、新增"安全拒绝/说教腔"和"生成残留"两条）精确率 41%（V1 40%），误报 13 条（V1 12 条），召回 89%→100%。这是样本内的结果，V2 还是看着这些错误写的，仍然没有改善。多数误报条目 5 次判定全部相同（标记率 0 或 1），说明错误是 judge 的系统性偏差，不是采样噪声，用多数投票（judge_repeats）也压不下去。V2 没有替换主实验里的 V1，主实验的数字仍然是用 V1 判的。
   - 另外 judge 与 qwen agent 同属 Qwen 系，可能偏向同系回复，未单独检验。
7. **ground truth 与 structured 记忆同源**：事实清单和 fact_consistency 的标准答案都来自同一份角色卡的抽取，且 system prompt 里已有角色卡描述；fact_consistency 偏高、区分度低可能与此有关。
8. 样本只有 61 条对话，且主角由"事实最多的角色"启发式选出，不代表 PIPPA 全体。
9. **对话短，记忆压力小**：每条对话最多 20 个用户轮次，`raw_context` 的窗口是 20 轮（用户+角色），大部分对话里它几乎等于完整历史；因此这个实验测的是"短对话下三种上下文注入方式的差别"，没有测上百轮的长期记忆，不能推出"长期记忆方案更好/更差"。另外 system prompt 里本来就有完整角色卡，structured_memory 注入的角色事实与它高度重叠，出戏率下降更可能来自"再次提醒人设"，而不是"记住了新信息"，实验没有设计对照把这两者分开。

## 已修复的、会静默污染结果的问题

- `run_dialogue()` 把 user_turn 传给 `retrieve()`，structured_memory 按 speaker 查事实永远落空，整段对话拿到的记忆上下文为空（见 `docs/DATASET.md`）。
- `vector_memory` 在 SURF 环境里没装 sentence-transformers，后来又因 embedding 模型被放到 GPU 0 与 vLLM 抢显存而 OOM，两种情况都静默退化成词重叠相似度；现已固定在 CPU 上运行，日志中无退化提示。
- SURF 上若干文件版本落后于本地（metrics.py、vector_memory.py、tests/）。
