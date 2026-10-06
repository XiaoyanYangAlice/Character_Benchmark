"""
方案 C：结构化事件记忆（GOLEM 式）。存储角色事实（来自
src/ontology_extraction 的抽取结果，对应 G17_Character_Feature）、事件三元组
（G5_Narrative_Event）和关系状态（G4_Social_Relationship），检索时按角色/时间/
事件类型查询，拼成结构化上下文。

TODO：9/25 实现事件与关系的增量抽取（对话进行中，而不只是开局导入角色事实）；
当前先支持从 src/ontology_extraction 产出的 character_features_*.jsonl 里
导入角色事实作为记忆库的初始状态（见 load_character_facts）。
"""
import json
from pathlib import Path

from .base import BaseMemory

# 抽取 prompt 唯一定义的三个合法 modality 值。实测跑过 62 个故事的全量抽取后
# 发现模型偶尔会自造出 prompt 里没定义的值（比如对将来时威胁性语句标了
# "future"），extract_character_features.py 的 verify_span() 只校验 span
# 是否逐字出现在句子里，不校验 modality 取值范围，所以这类脏数据会原样写进
# jsonl。这里在导入时兜底纠正，避免脏值传到下游的记忆检索/评测逻辑里。
VALID_MODALITIES = {"present", "past", "potential"}


class StructuredEventMemory(BaseMemory):
    name = "structured_memory"

    def __init__(self):
        self.character_facts: dict[str, list[dict]] = {}  # character -> [{feature_type, span, modality, evidence}]
        self.events: list[dict] = []       # {time, participants, event_type, content, emotion_salience, evidence_turn_id}
        self.relationships: list[dict] = []  # {character_a, character_b, dimension, delta, evidence_turn_id}
        # 每次 retrieve() 实际拼进 prompt 用到了哪些证据 id，供
        # src/eval/metrics.py::evidence_traceability_rate 计算。事实 id 用
        # "角色名#在该角色事实列表里的下标" 拼出来，够用来对照 character_facts
        # 定位到具体是哪条抽取记录（story_id/span/sentence_id）。
        self.retrieval_log: list[dict] = []

    def _ingest_fact(self, rec: dict) -> bool:
        """导入单条事实记录，非法 modality 兜底成 present。返回是否触发了兜底
        （调用方用来汇总打印提醒），不修改调用方传进来的原始 dict。"""
        rec = dict(rec)
        fixed = rec.get("modality") not in VALID_MODALITIES
        if fixed:
            # 兜底：非法 modality（比如 "future"）一律按 present 处理——
            # 这些基本上都是"角色断言的一个当前意图/倾向"，present 是最安全
            # 的默认值。数量应该很少（实测 1009 条里 2 条），调用方会打印汇总
            # 提醒，不代表数据整体质量有问题。
            rec["modality"] = "present"
        self.character_facts.setdefault(rec["character"], []).append(rec)
        return fixed

    def load_character_facts(self, features_jsonl_path: str | Path) -> None:
        """导入 src/ontology_extraction/extract_character_features.py 的完整
        输出（一份文件里所有故事、所有角色的事实）。

        ⚠ run_eval.py 跑主实验矩阵时不要用这个方法！这个方法会把文件里**全部
        故事**的事实都塞进同一个 self.character_facts，实测 61 段
        dialogues.jsonl 里有 20 段的 canonical 标签都是同一个占位词 "I"（角色
        卡用第一人称写、没给真名，见 docs/DATASET.md），如果每条 dialogue 都
        指向同一份 character_features_en.jsonl 调这个方法，会把 20 个互不
        相关的故事（Gawr Gura、Monika、Nao……）的事实全部合并到
        character_facts["I"] 下面，检索时张冠李戴且不会报错。跑单条 dialogue
        请用 load_character_facts_from_records(dialogue["fact_labels"])，
        只导入这一条 dialogue 自己、已经按 story_id 过滤好的事实。这个方法
        留给"只想看看某份原始抽取文件长什么样"的手动调试场景用。
        """
        path = Path(features_jsonl_path)
        n_fixed = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if self._ingest_fact(json.loads(line)):
                n_fixed += 1
        if n_fixed:
            print(f"[structured_memory] 提醒：{n_fixed} 条记录的 modality 不在 "
                  f"{sorted(VALID_MODALITIES)} 范围内，已按 present 处理。")

    def load_character_facts_from_records(self, records: list[dict]) -> None:
        """从内存里的事实记录列表导入（比如 dialogue["fact_labels"]），不重新
        读整份文件——run_eval.py 跑主实验矩阵时应该用这个方法，原因见
        load_character_facts() 的警告：一份 character_features_en.jsonl 里
        62 个故事共用同一份文件，不按 story_id 过滤会把不相关故事的同名角色
        事实（实测 "I" 这个标签横跨 20 个不同故事）混到一起。
        build_dialogues.py 产出的 dialogue["fact_labels"] 已经只保留了这一条
        dialogue 自己主角的事实，直接导入即可，不需要在这里再按 story_id 筛。
        """
        n_fixed = sum(self._ingest_fact(rec) for rec in records)
        if n_fixed:
            print(f"[structured_memory] 提醒：{n_fixed} 条记录的 modality 不在 "
                  f"{sorted(VALID_MODALITIES)} 范围内，已按 present 处理。")

    def on_new_turn(self, turn: dict) -> None:
        # TODO: 调用 LLM 抽取本轮新增的事件/关系变化，追加到 self.events / self.relationships
        # 复用 src/ontology_extraction 的抽取范式（few-shot prompt + 自检 span 校验）
        pass

    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        parts = []
        retrieved_fact_ids: list[str] = []

        # 角色事实：只带出当前轮涉及到的角色
        speaker = current_turn.get("speaker")
        if speaker in self.character_facts:
            used_facts = self.character_facts[speaker][:8]
            facts = "; ".join(f["span"] for f in used_facts)
            parts.append(f"[{speaker} 的已知事实] {facts}")
            retrieved_fact_ids = [f"{speaker}#{i}" for i in range(len(used_facts))]

        # 最近的相关事件（占位：取最近 5 条，TODO 按角色/事件类型过滤）
        if self.events:
            recent = self.events[-5:]
            ev_lines = [f"- {e['time']}: {e['content']}" for e in recent]
            parts.append("[相关事件]\n" + "\n".join(ev_lines))

        # 关系状态
        if self.relationships:
            rel_lines = [
                f"- {r['character_a']}-{r['character_b']}: {r['dimension']} {r['delta']:+d}"
                for r in self.relationships[-5:]
            ]
            parts.append("[关系状态]\n" + "\n".join(rel_lines))

        self.retrieval_log.append({"speaker": speaker, "retrieved_fact_ids": retrieved_fact_ids})

        return "\n\n".join(parts)
