# Character Memory Eval

**角色扮演 AI 记忆方案对比：原始上下文 vs 向量记忆 vs 结构化事件记忆**

NetEase Fuxi 的 NPC 记忆系统、腾讯《猫仙玦》的静态/动态双记忆机制、Yumeloop
《祝星之诺》的结构化记忆引擎（44 个角色、175 段关系）都在说明同一件事：长期
角色扮演交互的记忆问题已经是工业界的公开难题，但学术界缺一个可复现、可量化的
开源对比实验。这个仓库做的就是这件事：在同一批真实角色扮演对话上，对比三种
记忆方案在角色事实一致性、关系状态准确率、OOC 率、证据可追溯率四个指标上的
表现，用两个开源模型（Qwen2.5、Llama-3.1）跑评测。

评测器设计参考 Fabula（Mirowski et al., 2026）从叙事理论教材提炼 63 条评估
准则、与人类偏好相关性达 0.83 的做法；结构化记忆方案的字段设计直接对齐已发表的
[GOLEM 本体](https://pure.rug.nl/ws/portalfiles/portal/1513732875/humanities-14-00193-v2.pdf)
（Pianzola et al. 2025, *Humanities*, 14(10), 193）——角色事实抽取用的就是这篇
论文里 `G17_Character_Feature` 子类对应的四个抽取器（biography/emotion/
personality/physical），证据溯源对应 `E13_Attribute_Assignment`。

## 当前进度

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 项目方案设计（数据、架构、指标、时间表） | ✅ 完成 | 见团队内部执行方案文档 |
| 角色本体抽取（biography/emotion/personality/physical） | ✅ 代码完成，已过语法/逻辑冒烟检查 | `src/ontology_extraction/`，四个 prompt 复用自已发表方法论，中英双语版本 |
| SURF 部署指南 | ✅ 完成 | `docs/SURF_GUIDE.md`，从登录到拿到结果的完整步骤 |
| 数据预处理（对话筛选、格式化） | 🔜 待做 | `data/processed/dialogues.jsonl`，见 `docs/DATASET.md` |
| 方案 A：原始上下文拼接 | ✅ 已实现 | `src/memory/raw_context.py`，单元测试通过 |
| 方案 B：向量记忆 | 🔜 骨架已搭，检索逻辑待接 embedding 模型 | `src/memory/vector_memory.py` |
| 方案 C：结构化事件记忆 | 🔜 骨架已搭，角色事实导入已实现，事件/关系增量抽取待做 | `src/memory/structured_memory.py` |
| 评测指标与判断器 prompt | 🔜 待做 | `src/eval/metrics.py`、`src/eval/judge_prompts.py` |
| 主实验入口 | 🔜 待做 | `src/eval/run_eval.py` |

## 仓库结构

```
character-memory-eval/
├── README.md
├── requirements.txt
├── configs/
│   └── experiment.yaml          # 模型、记忆方案参数、评测配置
├── data/
│   ├── raw/                     # 原始下载数据（不入库，见 .gitignore）
│   ├── raw_stories/             # 待抽取的角色卡/故事文本（不入库）
│   └── processed/               # 清洗后的 JSONL（不入库，体积大；结构见 docs/DATASET.md）
├── docs/
│   ├── DATASET.md               # 数据来源、筛选标准、许可说明
│   └── SURF_GUIDE.md            # SURF (Snellius) 逐步操作指南
├── src/
│   ├── ontology_extraction/     # 角色本体抽取（biography/emotion/personality/physical）
│   │   ├── prompts/             # 8 个抽取 prompt（中文 v2 / 英文 v4）
│   │   ├── prepare_stories.py   # 原始文本 → 编号句子 JSONL
│   │   ├── extract_character_features.py  # 调 vLLM 跑抽取，自带 span 校验
│   │   └── validate_extractions.py        # 质检 + 双语一致性对照
│   ├── memory/
│   │   ├── base.py              # 三种方案共用接口 BaseMemory.retrieve()
│   │   ├── raw_context.py       # 方案 A
│   │   ├── vector_memory.py     # 方案 B
│   │   └── structured_memory.py # 方案 C
│   ├── generate/
│   │   └── dialogue_runner.py   # 驱动角色 agent 跑完整段对话
│   └── eval/
│       ├── judge_prompts.py     # LLM-as-judge 的判断准则
│       ├── metrics.py           # 四个指标的计算函数
│       └── run_eval.py          # 主入口：跑完整实验矩阵
├── slurm/
│   ├── ontology_smoke_test.slurm
│   └── ontology_run_extraction.slurm
├── tests/
│   └── test_memory.py
└── results/
    └── raw_outputs/             # 每条对话的完整生成记录（不入库）
```

## 快速开始

本地/登录节点上：

```bash
pip install -r requirements.txt
python -m pytest tests/          # 目前只有 raw_context 的测试
```

在 SURF 上跑角色本体抽取（当前唯一已经打通、可以实际运行的完整链路）：
见 [`docs/SURF_GUIDE.md`](./docs/SURF_GUIDE.md)，从登录 Snellius 开始一步步来。

## 复现完整实验（记忆方案对比部分完成后）

```bash
python src/eval/run_eval.py --config configs/experiment.yaml
```

（`run_eval.py` 目前是 TODO 骨架，见上面的进度表。）

## 许可与数据声明

见 [`docs/DATASET.md`](./docs/DATASET.md)：对话数据不直接分发，需要按说明自行
下载并运行预处理脚本复现。
