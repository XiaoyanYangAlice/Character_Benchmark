"""
方案 C：结构化事件记忆（GOLEM 式）。存储角色事实（来自
src/ontology_extraction 的抽取结果，对应 G17_Character_Feature）、事件三元组
（G5_Narrative_Event）和关系状态（G4_Social_Relationship），检索时按角色/时间/
事件类型查询，拼成结构化上下文。

v2 变化：`on_new_turn()` 不再是空实现。传入 `event_extractor` 之后，每完成一个
「用户发言 + 角色回复」的来回，就调用一次抽取器，把这一轮新出现的事实/事件写进
记忆。这是 v1 最大的名实不符：当时 structured_memory 只是"开局导入一次角色卡
事实"，对话里新出现的信息一律不进记忆，所以它根本没有在测"记忆"，只是在测
"再贴一遍人设"。没有 extractor 时行为与 v1 相同（只有静态事实），方便单元测试
和消融对比。

检索预算：事实最多 `max_facts` 条、事件最多 `max_events` 条，避免 structured
方案因为上下文更长而占便宜——三种方案的实际上下文长度由 run_eval.py 记录到
per_turn.csv 的 context_chars 列，报告里要一起给出。
"""
import json
import re
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
    supports_evidence = True

    def __init__(self, event_extractor=None, max_facts: int = 8, max_events: int = 8,
                 retrieval: str = "recency", n_recent: int = 4):
        """event_extractor: 可调用对象 `(exchange, character_name) -> list[dict]`，
        返回本轮新增的事件记录（见 on_new_turn）。传 None 时退回 v1 行为（只有
        开局导入的静态角色卡事实），用于消融实验和不需要 LLM 的单元测试。
        """
        self.character_facts: dict[str, list[dict]] = {}  # character -> [{feature_type, span, modality, evidence}]
        self.events: list[dict] = []       # {turn_id, kind, content, who}
        self.relationships: list[dict] = []  # {character_a, character_b, dimension, delta, evidence_turn_id}
        self.event_extractor = event_extractor
        self.max_facts = max_facts
        self.max_events = max_events
        # retrieval: "recency" = v2 行为，取最近 max_events 条；
        #            "hybrid"  = 保留 n_recent 条最近的（角色需要知道刚刚发生了什么，
        #                        纯按相关度取会丢掉对话连贯性），其余名额按与当前
        #                        用户发言的实词重合度选，并在打平时优先用户相关的事件。
        # 诊断依据：一段对话抽出约 43 条事件，只给 8 条，而其中仅约 27% 是关于
        # 用户的——探针问的却全是用户说过的话。等于角色只记得用户说的两件事。
        self.retrieval = retrieval
        self.n_recent = n_recent
        self.extraction_failures = 0
        # 每次 retrieve() 实际拼进 prompt 用到了哪些证据，供
        # src/eval/metrics.py::evidence_use_rate 计算。事实 id 用
        # "角色名#在该角色事实列表里的下标" 拼出来，够用来对照 character_facts
        # 定位到具体是哪条抽取记录（story_id/span/sentence_id）。
        self.retrieval_log: list[dict] = []
        # 待抽取的一个来回（用户发言 + 角色回复）。on_new_turn 收到角色轮次时
        # 才触发抽取，因为"这一轮发生了什么"要两边都看到才说得清。
        self._pending_user_turn: dict | None = None

    # ---------------------------------------------------------------- 导入静态事实
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
        故事**的事实都塞进同一个 self.character_facts，实测 dialogues.jsonl 里
        有相当比例的 canonical 标签都是同一个占位词 "I"（角色卡用第一人称写、
        没给真名，见 docs/DATASET.md），如果每条 dialogue 都指向同一份
        character_features_en.jsonl 调这个方法，会把互不相关的故事（Gawr Gura、
        Monika、Nao……）的事实全部合并到 character_facts["I"] 下面，检索时
        张冠李戴且不会报错。跑单条 dialogue 请用
        load_character_facts_from_records(dialogue["fact_labels"])，只导入这一条
        dialogue 自己、已经按 story_id 过滤好的事实。这个方法留给"只想看看某份
        原始抽取文件长什么样"的手动调试场景用。
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
        load_character_facts() 的警告。
        """
        n_fixed = sum(self._ingest_fact(rec) for rec in records)
        if n_fixed:
            print(f"[structured_memory] 提醒：{n_fixed} 条记录的 modality 不在 "
                  f"{sorted(VALID_MODALITIES)} 范围内，已按 present 处理。")

    # ---------------------------------------------------------------- 增量抽取
    def on_new_turn(self, turn: dict) -> None:
        """每轮结束后调用。用户轮次先存起来，等到角色回复也到了，再把这一个
        来回交给 event_extractor 抽取"本轮新信息"。

        为什么按来回抽取而不是每轮抽一次：用户说"我妹妹叫 Anna"、角色回"Anna
        这名字真好听"——单看任何一边都容易丢掉指代，两边一起看抽出来的事件
        才是完整的。顺带把 LLM 调用次数减半。
        """
        if turn.get("speaker") == "user":
            self._pending_user_turn = turn
            return

        if self.event_extractor is None:
            self._pending_user_turn = None
            return

        exchange = {"user_turn": self._pending_user_turn, "character_turn": turn}
        self._pending_user_turn = None
        try:
            new_events = self.event_extractor(exchange) or []
        except Exception as e:  # 抽取失败不应该让整段对话白跑
            self.extraction_failures += 1
            print(f"[structured_memory] 事件抽取失败（turn {turn.get('turn_id')}）：{e}")
            return

        for ev in new_events:
            if not isinstance(ev, dict) or not str(ev.get("content", "")).strip():
                continue
            self.events.append({
                "turn_id": turn.get("turn_id"),
                "kind": str(ev.get("kind", "fact"))[:32],
                "who": str(ev.get("who", ""))[:64],
                "content": str(ev["content"]).strip()[:300],
            })

    # ---------------------------------------------------------------- 选取事件
    def _select_events(self, current_turn: dict) -> list[dict]:
        """从累计的事件里挑出这一轮要注入的那几条。

        "recency" 是 v2 的行为：取最近 max_events 条。实测它把探针要考的早期
        信息全部挤掉了——43 条事件只给 8 条，而且给的是最新的 8 条。

        "hybrid" 保留 n_recent 条最近的（角色总得知道刚刚发生了什么），剩下的
        名额按与当前用户发言的实词重合度来选。打分时给"关于用户"的事件一点
        加成，因为用户陈述的信息才是会被回头问起的那种。相关度用实词重合而不是
        句向量，是为了不引入新依赖、保持确定性；换成句向量是一个干净的后续消融。
        """
        if self.retrieval != "hybrid" or len(self.events) <= self.max_events:
            return self.events[-self.max_events:]

        n_recent = min(self.n_recent, self.max_events)
        recent = self.events[-n_recent:] if n_recent else []
        recent_ids = {id(e) for e in recent}
        pool = [e for e in self.events if id(e) not in recent_ids]

        query = content_tokens(str(current_turn.get("text") or ""))
        def score(e: dict) -> tuple:
            toks = content_tokens(e.get("content", ""))
            overlap = len(query & toks) / len(query) if query else 0.0
            is_user = 1 if str(e.get("who", "")).strip().lower() == "user" else 0
            # 先按重合度，再按是否关于用户，最后按时间近远（都相同时取较新的）
            return (overlap, is_user, e.get("turn_id") or 0)

        n_slots = self.max_events - len(recent)
        chosen = sorted(pool, key=score, reverse=True)[:n_slots]
        # 按时间排序后再拼进 prompt，读起来才像一条时间线
        return sorted(chosen + recent, key=lambda e: e.get("turn_id") or 0)

    # ---------------------------------------------------------------- 检索
    def retrieve(self, history: list[dict], current_turn: dict) -> str:
        parts = []
        retrieved_fact_ids: list[str] = []
        retrieved_spans: list[str] = []

        # 角色事实：只带出当前轮涉及到的角色
        speaker = current_turn.get("speaker")
        if speaker in self.character_facts:
            used_facts = self.character_facts[speaker][: self.max_facts]
            facts = "; ".join(f["span"] for f in used_facts)
            parts.append(f"[Known facts about {speaker}] {facts}")
            retrieved_fact_ids = [f"{speaker}#{i}" for i in range(len(used_facts))]
            retrieved_spans += [f["span"] for f in used_facts]

        # 对话中积累的新信息（on_new_turn 抽出来的）
        if self.events:
            recent = self._select_events(current_turn)
            ev_lines = [f"- (turn {e['turn_id']}) {e['content']}" for e in recent]
            parts.append("[Established in this conversation]\n" + "\n".join(ev_lines))
            retrieved_spans += [e["content"] for e in recent]

        # 关系状态
        if self.relationships:
            rel_lines = [
                f"- {r['character_a']}-{r['character_b']}: {r['dimension']} {r['delta']:+d}"
                for r in self.relationships[-5:]
            ]
            parts.append("[Relationship state]\n" + "\n".join(rel_lines))

        self.retrieval_log.append({
            "speaker": speaker,
            "retrieved_fact_ids": retrieved_fact_ids,
            "retrieved_spans": retrieved_spans,
            "n_events": len(self.events),
        })

        return "\n\n".join(parts)


# -------------------------------------------------------------------- LLM 抽取器
EVENT_EXTRACTION_PROMPT = """You are maintaining a memory log for a role-play character.

Below is one exchange from an ongoing conversation. Record new, concrete information
worth remembering later.

PRIORITY. Record these FIRST and never leave them out:
1. Anything the USER stated about themselves or their situation: names, numbers,
   places, relationships, stated plans, stated preferences, opinions they voiced.
2. Corrections — the user revising something they said earlier.
3. Commitments either side made.
Only after those, record at most one concrete thing that happened in the scene.

DO NOT record: the character's personality, appearance or backstory (those are already
in the character card), emotions described in the abstract, or anything not actually said.

PRESERVE EXACT WORDS for names, numbers, places and quoted phrases. A memory that
generalises away the specific is useless later.

  Good: {{"who": "user", "kind": "fact", "content": "The user's sister is named Anna."}}
  Bad:  {{"who": "user", "kind": "fact", "content": "The user talked about their family."}}

  Good: {{"who": "user", "kind": "fact", "content": "The user weighs around 150 lbs."}}
  Bad:  {{"who": "user", "kind": "fact", "content": "The user answered a question about weight."}}

  Good: {{"who": "user", "kind": "fact", "content": "The user said a fight involves two people."}}
  Bad:  {{"who": "Kasey", "kind": "event", "content": "took a bite of the user's lunch"}}
        (a passing character action carrying no lasting information)

If nothing new and concrete was established, return an empty list.

USER SAID:
{user_text}

CHARACTER REPLIED:
{character_text}

Return JSON only, in this exact form:
{{"events": [{{"who": "<'user' or the character's name>", "kind": "<fact|event|commitment>", "content": "<one short sentence>"}}]}}
"""


def make_llm_event_extractor(client, model: str, max_events_per_turn: int = 3,
                              temperature: float = 0.0):
    """返回一个可以传给 StructuredEventMemory(event_extractor=...) 的抽取器。

    用 temperature 0：记忆写入应该是确定性的，同一个来回每次抽出来的东西不该
    变。这跟 judge 不同（judge 用 0.3 是沿用主实验配置）。
    """
    from src.eval.metrics import as_list, call_judge  # 复用同一套 JSON 解析与重试

    def extract(exchange: dict) -> list[dict]:
        user_turn = exchange.get("user_turn") or {}
        char_turn = exchange.get("character_turn") or {}
        prompt = EVENT_EXTRACTION_PROMPT.format(
            user_text=(user_turn.get("text") or "")[:1500],
            character_text=(char_turn.get("text") or "")[:1500],
        )
        result = call_judge(client, model, prompt, temperature=temperature)
        return as_list(result, "events")[:max_events_per_turn]

    return extract


# -------------------------------------------------------------------- 证据重合度
_CONTENT_TOKEN_RE = re.compile(r"[一-鿿]|[A-Za-z][A-Za-z'-]{2,}")
# 英文高频虚词：出现在任何句子里都不说明"用上了这条证据"，必须剔除，否则
# overlap 会被 the/and/that 这类词刷到很高，指标又变成恒为 1 的废指标。
_STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "you", "your", "are", "was", "were",
    "have", "has", "had", "but", "not", "they", "them", "she", "her", "his", "him",
    "from", "what", "when", "who", "all", "can", "will", "would", "could", "about",
    "there", "their", "been", "its", "it's", "just", "like", "one", "out", "get",
    "now", "how", "then", "than", "into", "more", "some", "other", "any", "our",
}


def content_tokens(text: str) -> set[str]:
    """抽出"有信息量的词"：中文逐字、英文取长度≥3 的词，去掉高频虚词。"""
    return {t for t in (m.group(0).lower() for m in _CONTENT_TOKEN_RE.finditer(text))
            if t not in _STOPWORDS}
