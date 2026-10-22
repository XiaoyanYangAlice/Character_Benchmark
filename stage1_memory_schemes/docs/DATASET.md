# 数据说明

## 对话数据（生成实验用）

- **主数据源**：[PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA)（约 26 万轮来自
  Character.AI 的真实用户对话，附角色卡）。筛选 ≥15 轮（指 PIPPA 原对话的消息总数）、角色设定信息丰富的片段（注意：`build_dialogues.py` 随后只取前 6 个用户发言作为脚本，见 RESULTS.md 限制 9），
  实际导出 90 段角色描述（`fetch_pippa.py --limit 90`）。**实测这批数据是纯英文**
  （角色卡本身就是英文写的），所以角色本体抽取统一用 `--lang en`（v4 英文版
  prompt），不使用中文版。
- **对照数据源（可选，暂未使用）**：SillyTavern 社区角色卡（character card v2
  格式，如从 Chub.ai 导出）。当前没有现成的角色卡，这条路径先跳过；如果后续要
  补中文数据（比如配合中文角色卡或 AO3 中文故事），再用 `cards_to_stories.py`
  转换、走 `--lang zh` 抽取，届时可以恢复中英文一致性对照。
- `data/processed/dialogues.jsonl` 由 `src/generate/build_dialogues.py` 产出
  （不需要重新下载数据，复用 `fetch_pippa.py` 已经建好的本地缓存，这次把
  `conversation` 字段里人类说的轮次也取出来，之前只用了 `bot_description`）。
  每行一条：
  ```json
  {"dialogue_id": "...", "character_card": {"name": "...", "description": "...", "greeting": "..."},
   "scripted_user_turns": ["...", "..."], "fact_labels": [...], "relationship_trajectory": [...]}
  ```
  `scripted_user_turns` 是从原始对话里取出的真实人类发言，固定住给三种记忆方案
  统一跑，保证对比公平（差异只来自记忆方案给了什么上下文，不会混进"用户聊的
  内容不一样"这个变量）。

## 角色事实 ground truth（`fact_labels` 字段）

不再用临时 prompt 做弱标注，改用 `src/ontology_extraction/` 里已经过验证的
biography/emotion/personality/physical 四类抽取 prompt（与 GOLEM 本体论文
`G17_Character_Feature` 子类对齐），对角色卡文本跑一次抽取，直接产出
`character_features_en.jsonl`（PIPPA 数据是纯英文，用 v4 英文版 prompt），
合并进 `fact_labels`。具体步骤见 [`SURF_GUIDE.md`](./SURF_GUIDE.md)。

## 已知局限：主角选择用的是启发式，不是精确方法

一个 PIPPA 故事的 `bot_description` 经常会提到好几个角色（主角 + 配角），但
`character_features_en.jsonl` 里是按 `story_id` 存了这个故事所有出现过的角色
的事实，不区分谁是这个 bot 真正在扮演的主角。`build_dialogues.py` 用
"事实条数最多的角色 = 主角"这个启发式来选（`pick_protagonist()`），依据是
"描述篇幅越大，抽取出来的事实自然越多"，抽样检查基本成立，但不是精确方法——
极端情况下（配角描述篇幅反而更长）可能选错。`--min-facts` 参数会把事实条数
太少的故事跳过，减少明显不合理的样本，但不保证每条都选对，跑完
`build_dialogues.py` 之后建议抽样核对几条 `character_card.name` 是否真的对应
这个故事该演的角色，再往下跑生成实验。

## 已知局限：ground truth 抽取本身的指代消歧误差

对 62 个故事跑完全量抽取（1009 条记录）后做质检发现两类问题，如实记录在这里，
不掩盖：

1. **角色指代消解不完全稳定**：抽样看到至少 1 个故事（`pippa_8TMl93F5tRQ...`，
   一个"军阀 Alastair + 公主 Sofiya"的角色卡）里，`biography` 类别正确把
   Sentence 1-2 的内容都归到了 `Alastair`，但 `personality`/`emotion`/
   `physical` 三个类别的调用里，指代同一个人的代词 `"he"` 没有被解析回
   `Alastair`，被当成了一个独立的伪角色 `"He"`；`Princess Sofiya`（长名）和
   `Sofiya`（短名，后续句子里退化出现）也没有被统一成同一个 canonical label。
   根因：四个特征类别是四次**独立**的模型调用，每次调用都要在自己的上下文里
   重新做一遍"角色注册表"，稳定性不如同一次调用内部的指代消解。
   `validate_extractions.py` 现在会把角色数超过 3 个的故事全部列出来，方便
   人工抽查，但**不会自动修正**——这是当前用纯 LLM in-context 做指代消解、
   没有专门指代消解模块的固有局限，也正是这个项目要拿三种记忆方案去对比评测
   的问题本身（结构化记忆方案如果要在实践中可用，就需要正视这类噪声）。
2. **modality 偶尔跑出 prompt 定义的枚举范围**：1009 条里出现了 2 条
   `"future"`（prompt 只定义了 `present`/`past`/`potential`），都是角色描述里
   将来时的威胁性语句。`extract_character_features.py` 的 span 自检不校验
   modality 取值，所以这类脏值会原样写进 jsonl；`structured_memory.py` 的
   `load_character_facts()` 现在会把不在枚举里的值统一兜底成 `present` 并打印
   提醒，避免脏值污染下游检索逻辑。

处理方式：考虑到 8 天时间预算，这里不做人工逐条修正或重新设计两阶段指代消解
流程，而是把这个误差率作为**已知局限**记录，在最终结果分析里说明"ground truth
本身并非 100% 干净，可能给三种记忆方案的评测分数带来共同的噪声下限"。如果之后
有时间，可以考虑的改进方向见 SURF_GUIDE.md 或 README 里冒烟测试环节讨论的
两阶段抽取方案。

## 已修复：第一人称角色卡的 canonical 标签是代词，生成 prompt 里会读不通

`build_dialogues.py` 真跑出来的 61 段对话抽查发现：8 条样本里有 5 条
（62.5%）的 `character_card.name` 不是真名，而是 `"I"`/`"Me"` 这种代词，或者
`"Harasser"`/`"Automaton"` 这种从描述里挑出来的词——根因是这些 PIPPA 角色卡
整段用第一人称写（比如 "I love to sing, play rhythm games..."），文本里压根
没出现过专有名词，`extract_character_features.py` 的四个抽取器找不到名字可用，
只能把代词本身当 canonical 标签写进 `character_features_en.jsonl`。

这对事实检索没有影响（同一个故事内部自洽，`fact_labels` 里的 `character`
字段也是同一个 `"I"`），但 `dialogue_runner.py` 的 `build_system_prompt()`
会把这个标签直接拼进生成指令：`"You are role-playing as I. ... Respond only
in first person as I."`——这句话语法本身是破的，可能拖累生成质量，而且是
均匀影响全部三种记忆方案（不偏向某一个方案，不破坏对比公平性，但会拉低
整体数据质量、给判断器打分引入额外噪声）。

修复：`build_dialogues.py` 新增 `pick_display_name()`，canonical 标签是代词
（`i`/`me`/`you`/`we`/`us`/`he`/`she`/`they`/`him`/`her`，大小写不敏感）且
PIPPA 自带 `bot_name` 字段非空时，取 `bot_name` 当 `character_card
["display_name"]`；否则 `display_name` 就还是 canonical 标签本身。
`dialogue_runner.py` 的 `build_system_prompt()` 改成优先用 `display_name`
拼生成文字，`character_card["name"]`（事实检索用的 canonical 标签）和
`run_dialogue()` 里角色轮次的 `speaker` 字段**不受影响**，仍然认 `name`——
两者用途不同，不能混用，混用会导致 `structured_memory.py` 查不到事实、
静默退化成"结构化记忆方案什么都没用上"。已加单元测试覆盖（`tests/
test_build_dialogues.py`、`tests/test_dialogue_runner.py`）。

⚠ 已经跑过一次的 `dialogues.jsonl` 不会自动带上 `display_name` 字段——
用了这个修复版的 `build_dialogues.py` 之后需要**重新跑一遍**才能生效。

实测规模：重新跑一遍后确认，61 段对话里有 **20 段（约 33%）** 的 canonical
标签是这类第一人称占位词，修复后都从 PIPPA 自带的 `bot_name` 拿到了正常的
`display_name`（Gawr Gura、Monika、Shirakami Fubuki 等），不是极端个例。

## 已修复：`structured_memory.py` 按同一份文件加载事实会跨故事串味

正是上面这个"20 个故事共用同一个 canonical 标签 `I`"的发现，牵出了
`structured_memory.py` 里一个更严重的潜在 bug：`load_character_facts()`
是把 `character_features_en.jsonl` **整份文件**按 `character` 字段分组导入
的，不区分 `story_id`。如果 `run_eval.py` 对每条 dialogue 都指向同一份文件
调用这个方法，`character_facts["I"]` 会把这 20 个互不相关故事（Gawr Gura、
Monika、Nao……）的事实全部合并在一起——检索时张冠李戴，且不会报错，是那种
会悄悄污染方案 C 全部评测结果、事后很难定位的 bug。

修复：新增 `load_character_facts_from_records(records)`，只导入内存里
已经按 `story_id` 过滤好的一份事实列表（也就是 `dialogue["fact_labels"]`），
不重新读整份文件。**`run_eval.py` 跑主实验矩阵必须用这个方法，不能用
`load_character_facts()`**——后者只留给"手动看看某份原始抽取文件里有什么"
的调试场景。已加回归测试（`tests/test_memory.py::
test_structured_memory_from_records_does_not_leak_other_stories_with_same_label`）
覆盖这个具体场景。

## 已修复：`run_dialogue()` 把 user_turn 传给 `retrieve()`，structured_memory 的证据检索静默退化成 0

全量实验（366 行，`results/summary.csv`）跑完后发现：两个模型在
`structured_memory` 方案下的 `evidence_traceability` 都精确等于 **0.000**，
61 个样本一个不剩（n=61 但均值 0.000），比 bug 的典型特征还典型——真实数据
几乎不可能这么整齐。

根因在 `src/generate/dialogue_runner.py` 的 `run_dialogue()`：每轮调用
`memory.retrieve(history, current_turn)` 时，传的 `current_turn` 就是
`user_turn`（`speaker` 字段永远是字符串 `"user"`）。这个参数在两种有状态
记忆方案里的用法完全不同：

- `vector_memory.py` 只读 `current_turn["text"]` 当检索 query——传
  `user_turn` 正好是对的（检索"跟用户刚说的话相关的记忆"）。
- `structured_memory.py` 只读 `current_turn["speaker"]`，拿去查
  `self.character_facts[speaker]`——传 `user_turn` 时 `speaker` 永远是
  `"user"`，这张表里从来没有叫 `"user"` 的角色，查表永远落空，`retrieve()`
  **不报错**，只是悄悄把检索结果拼成空字符串，`retrieval_log` 里
  `retrieved_fact_ids` 永远是 `[]`。`evidence_traceability_rate()` 看到
  log 非空但每条都没证据，算出来就是 0.000——不是"structured_memory 这个
  方案没用"，是从来没有真的跑过检索逻辑。

修复：`run_dialogue()` 改成传一个"即将要生成的这一轮"的占位 turn
（`speaker` 换成角色的 canonical name，`text` 仍然是用户刚说的话），同时
满足两种方案的需求，而不是直接复用 `user_turn`。已加回归测试
（`tests/test_dialogue_runner.py::
test_run_dialogue_passes_character_name_as_speaker_to_structured_memory`），
断言 `retrieval_log` 里的 `speaker` 是角色名、`retrieved_fact_ids` 非空。

**影响范围比最初判断的更大，不只是 `evidence_traceability` 这一列。**
`structured_memory.retrieve()` 拼三段内容：角色事实（卡在这个 bug 上）、
事件（`self.events`）、关系状态（`self.relationships`）。后两段靠
`on_new_turn()` 增量抽取填充，而 `on_new_turn()` 目前还是个 TODO 空实现
（见下一节），从一开始就永远是空列表，跟这个 bug 无关。三段全部拿不到
内容时，`retrieve()` 返回的就是空字符串——也就是说这次全量实验里，
`structured_memory` 方案下模型生成角色回复时实际拿到的"记忆上下文"
从头到尾都是空的，跟 `raw_context`/`vector_memory` 那两个方案（确实把
真实历史对话拼进了 prompt）不是同一个起跑线。`results/summary.csv` 里
`structured_memory` 这一行的全部四个指标（`fact_consistency`、
`ooc_rate`、`evidence_traceability`）都需要在修复后重跑，不只是
`evidence_traceability` 一列；`raw_context`/`vector_memory` 两行不受
影响，可以直接用。

## 关系状态轨迹（`relationship_trajectory` 字段）

用开源 LLM（如 Qwen2.5-72B）按固定 prompt 为每 3-5 轮对话打亲密度/信任度等级
（1-5 分），抽样 15% 人工复核修正，得到可用但廉价的弱标签。TODO：9/23 实现打分
脚本。

## 许可与合规

- PIPPA：遵循数据集本身的使用条款（学术研究用途，发布前需在 README 注明来源与
  局限性，不能直接重新分发原始用户对话文本，只发布处理后的统计结果/示例）。
- SillyTavern 角色卡：不同角色卡作者授权不同，抓取前逐个确认可否用于研究/是否
  允许再分发；仓库里只保留角色设定的结构化字段，不整段转载。
- 发布 GitHub 仓库前，`DATASET.md` 需要写清楚"数据不直接分发，需要用户自行下载
  并运行 `prepare_stories.py`/预处理脚本复现"。
