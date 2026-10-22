"""
指标计算。判断器 prompt 在 judge_prompts.py。

v2 的两处结构性变化：

1. **逐轮判定会被保存下来**（`judge_turns()` 返回每一轮的判定，run_eval.py 写进
   `per_turn.csv`）。v1 只存了每条对话的汇总值，导致后来做 judge 审核时必须
   重新判一遍 600 轮——重判和当时那次不是同一批判定（judge 温度非 0），审核
   结论和主实验之间存在无法消除的缝隙。存逐轮判定之后，审核直接抽主实验那次
   真正用过的判定，缝隙消失，而且可以按"对话前段/后段"、按方案分层做分析。

2. **`evidence_traceability` 废弃，换成 `evidence_use_rate`**。前者只检查
   "这一轮检索结果是不是非空"，所以对 vector/structured 恒为 1.000，是个
   没有信息量的代理指标。后者检查"回复里是不是真的出现了检索到的内容词"，
   虽然仍是词面重合、不等于语义上用到了，但至少不是恒为 1。

judge 相关指标仍然走"重复 judge_repeats 次 + 多数投票"。v1 用的是
judge_repeats=1，v2 默认 3：审核发现 judge 的系统性误报压不下去，但多数投票
至少能压掉采样噪声那一部分，让"方案之间的差"少一层随机性。
"""
import json
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 只在类型检查时导入，不要求这个模块在没装 openai 包的环境（比如本地单元
    # 测试用假 client）里也能跑；真实调用时传进来的 client 只要实现
    # client.chat.completions.create(...) -> 带 .choices[0].message.content
    # 的对象就行，不要求一定是 openai.OpenAI 实例（tests/test_eval_metrics.py
    # 里的 FakeClient 就是鸭子类型的例子）。
    from openai import OpenAI
else:
    OpenAI = Any

from src.memory.structured_memory import content_tokens

from .judge_prompts import (
    FACT_CONSISTENCY_JUDGE_PROMPT,
    OOC_JUDGE_PROMPT,
    OOC_JUDGE_PROMPT_V2,
    PROBE_RECALL_JUDGE_PROMPT,
    RELATIONSHIP_STATE_JUDGE_PROMPT,
)

OOC_PROMPTS = {"v1": OOC_JUDGE_PROMPT, "v2": OOC_JUDGE_PROMPT_V2}


def _extract_json(raw: str) -> dict | None:
    """跟 ontology_extraction/extract_character_features.py 里的同名逻辑一致
    （判断器也偶尔会包一层 markdown 代码块），没有抽成共享模块是因为两边分属
    「数据准备」和「运行时评测」两个独立阶段，各自独立演进更清楚，代码量也不大，
    重复一份的维护成本低于引入跨模块耦合。"""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def as_list(result, key: str) -> list:
    """从 judge 的返回里取一个列表字段，容忍模型直接返回裸数组。

    实测 Qwen2.5-14B 有时会无视 prompt 里的 schema，直接返回 `[...]` 而不是
    `{"<key>": [...]}`。调用方如果直接 `result.get(key)` 就会抛
    AttributeError: 'list' object has no attribute 'get'——这正是第一次跑
    make_probes.py 时整个作业挂掉的原因。
    """
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        v = result.get(key)
        if isinstance(v, list):
            return v
    return []


def call_judge(client: OpenAI, model: str, prompt: str, max_retries: int = 3,
               temperature: float = 0.3, max_tokens: int = 384) -> dict | list | None:
    """通用 judge 调用：跑 prompt，解析 JSON，失败重试。

    temperature 默认 0.3（沿用主实验配置，配合多数投票）；需要确定性写入的
    场景（结构化记忆的事件抽取）显式传 0.0。max_tokens 比 v1 的 256 大一些，
    因为 V2 prompt 要求"先写理由再给结论"，输出更长，256 会把 JSON 截断，
    表现为解析失败率升高。
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = _extract_json(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  [judge retry {attempt + 1}] API error: {e}")
        time.sleep(1)
    return None


def _vote_bool(client: OpenAI, judge_model: str, prompt: str, key: str,
               judge_repeats: int) -> tuple[bool | None, list[bool], str]:
    """重复调用 judge_repeats 次取多数投票。

    返回 (多数结论, 每次的投票, 最后一次的理由)。多数结论为 None 表示
    judge_repeats 次全部失败（无法解析出合法 JSON）；调用方必须把这类样本
    单独统计、不计入分母，而不是悄悄当成 False——那样会系统性低估问题发生率。
    同时返回逐次投票，便于事后看"这条是 judge 稳定判定还是摇摆"：v1 的审核
    发现误报大多是 5 次全同，这正是判断"系统性偏差 vs 采样噪声"的依据。
    """
    votes: list[bool] = []
    reason = ""
    for _ in range(judge_repeats):
        result = call_judge(client, judge_model, prompt)
        # isinstance 检查不能省：模型偶尔返回裸数组，那时 `key in result` 会
        # 在 list 上做成员判断（永远 False），虽然不报错但会被误计成"判定失败"。
        if isinstance(result, dict) and key in result:
            votes.append(bool(result[key]))
            reason = str(result.get("reason", ""))[:300] or reason
    if not votes:
        return None, [], ""
    return sum(votes) > len(votes) / 2, votes, reason


# -------------------------------------------------------------- 规则型助手腔检测
# v1 的 judge 审核发现：真正抓到的出戏几乎全是"安全拒绝/说教腔"和模板残留。
# 既然真实出戏集中在这么窄的一类现象上，就可以用确定性规则直接数出来，作为
# LLM judge 的独立交叉验证——规则没有 judge 那种"把符合设定的行为当出戏"的
# 系统性误报，代价是只覆盖这一类、必然漏判其它类型。两个指标一起报告：
# 规则检测率给"至少这么多"的下界，judge 标记率给"至多这么多"的上界。
_LEAKAGE_PATTERNS = [
    # 元话语：自称 AI / 语言模型
    r"\bas an? (?:ai|artificial intelligence|language model|assistant)\b",
    r"\bi'?m (?:an? )?(?:ai|chatbot|language model)\b",
    r"\bi (?:can ?not|can't|am unable to) (?:continue|engage|assist|help) with (?:this|that)\b",
    # 安全说教腔
    r"\bsafety (?:always )?(?:comes|come) first\b",
    r"\b(?:remember|keep in mind)(?: that)?,? (?:consent|safety|communication) is\b",
    r"\bif you(?:'re| are) (?:ever )?(?:feeling|in) (?:unsafe|distress|crisis)\b",
    r"\bplease (?:seek|consider seeking) (?:professional )?help\b",
    r"\blet'?s take a (?:break|step back)\b",
    # 助手式服务用语
    r"\bis there anything else i can help you with\b",
    r"\bhere (?:are|is) (?:a few|some|several) (?:suggestions|tips|options|ideas)\b",
    r"\bi hope this helps\b",
    # 生成模板残留（聊天模板的角色标记漏进正文）
    r"(?:^|\n)\s*(?:assistant|system|user)\s*[:：]",
    r"<\|(?:im_start|im_end|eot_id|start_header_id)\|>",
]
_LEAKAGE_RE = [re.compile(p, re.IGNORECASE) for p in _LEAKAGE_PATTERNS]


def detect_assistant_leakage(text: str) -> list[str]:
    """返回命中的规则（可能多条）。空列表表示没检出。

    这是高精确、低召回的检测器：命中基本可以确定是出戏，没命中不代表没出戏。
    报告里必须这样表述，不能当成"真实出戏率"。
    """
    return [r.pattern for r in _LEAKAGE_RE if r.search(text or "")]


# -------------------------------------------------------------- 逐轮判定
def judge_turns(client: OpenAI, judge_model: str, turns: list[dict], character_card: str,
                character_facts: list[dict], judge_repeats: int = 3,
                judge_repeats_fact: int | None = None,
                ooc_prompt_version: str = "v2",
                retrieval_log: list[dict] | None = None,
                do_ooc: bool = True, do_fact: bool = True) -> list[dict]:
    """对每个角色轮次跑 OOC 判定和事实矛盾判定，返回逐轮记录。

    返回的每条记录：
      turn_id, turn_index（第几个角色轮次，从 1 开始）, is_probe,
      ooc（True/False/None=判定失败）, ooc_votes, ooc_reason,
      fact_contradicts（同上）, leakage（规则命中数）,
      context_chars（这一轮注入的记忆上下文长度）, reply_chars,
      evidence_overlap（回复与检索内容的内容词重合数，没有检索到就是 None）

    汇总值由 aggregate_per_turn() 从这份逐轮记录算出来，不在这里直接算——
    这样"原始判定"和"怎么汇总"分开，事后想按轮次位置分层、想换汇总口径，
    都不需要重新调用 judge。
    """
    # 事实一致性可以用更少的重复次数：这个指标接近天花板、区分度低（v1 三次
    # 重复的差异全部不显著），没必要和 OOC 一样投 3 次。judge 调用量是这次
    # 实验的主要成本，这里减半能直接砍掉近一半的 judge 时间。
    if judge_repeats_fact is None:
        judge_repeats_fact = judge_repeats
    prompt_tmpl = OOC_PROMPTS.get(ooc_prompt_version)
    if prompt_tmpl is None:
        raise ValueError(f"未知的 ooc_prompt_version: {ooc_prompt_version}，可选 {sorted(OOC_PROMPTS)}")

    facts_text = "\n".join(
        f"- [{f.get('feature_type', '?')}] {f['span']} (modality={f.get('modality', 'present')})"
        for f in character_facts
    ) or "（无已知事实）"

    character_turns = [t for t in turns if t.get("speaker") != "user"]
    if not character_turns:
        raise ValueError("turns 里没有角色轮次，无法判定")

    records = []
    for idx, t in enumerate(character_turns, 1):
        text = t.get("text", "")
        ooc, ooc_votes, ooc_reason = (None, [], "")
        if do_ooc:
            ooc_prompt = prompt_tmpl.format(character_card=character_card, turn_text=text)
            ooc, ooc_votes, ooc_reason = _vote_bool(client, judge_model, ooc_prompt, "ooc",
                                                     judge_repeats)

        contradicts, fact_votes = None, []
        if do_fact:
            fact_prompt = FACT_CONSISTENCY_JUDGE_PROMPT.format(facts=facts_text, turn_text=text)
            contradicts, fact_votes, _ = _vote_bool(client, judge_model, fact_prompt,
                                                     "contradicts_known_fact", judge_repeats_fact)

        rec = {
            "turn_id": t.get("turn_id"),
            "turn_index": idx,
            "is_probe": bool(t.get("is_probe")),
            "ooc": ooc,
            "ooc_votes": "".join("1" if v else "0" for v in ooc_votes),
            "ooc_reason": ooc_reason.replace("\n", " "),
            "fact_contradicts": contradicts,
            "fact_votes": "".join("1" if v else "0" for v in fact_votes),
            "leakage": len(detect_assistant_leakage(text)),
            "context_chars": t.get("context_chars"),
            "reply_chars": len(text),
            "evidence_overlap": None,
        }
        if retrieval_log is not None and idx - 1 < len(retrieval_log):
            entry = retrieval_log[idx - 1] or {}
            rec["evidence_overlap"] = _evidence_overlap(entry, text)
        records.append(rec)
    return records


def _evidence_overlap(entry: dict, reply_text: str) -> int | None:
    """这一轮检索到的内容与回复之间的内容词重合数；没检索到任何东西返回 None
    （区别于"检索到了但没用上"=0，这两件事不能混为一谈）。"""
    spans = entry.get("retrieved_spans")
    if spans is None:
        # vector_memory 的 log 里存的是 turn_id，没有文本；调用方（run_eval）
        # 负责把检索到的文本填进 retrieved_spans，填不了就返回 None。
        return None
    if not spans:
        return None
    ev_tokens = set()
    for s in spans:
        ev_tokens |= content_tokens(str(s))
    if not ev_tokens:
        return None
    return len(ev_tokens & content_tokens(reply_text))


# -------------------------------------------------------------- 汇总
def aggregate_per_turn(records: list[dict], late_from_fraction: float = 0.5) -> dict:
    """从逐轮判定算汇总值，并按"对话前段/后段"分层。

    late_from_fraction=0.5 表示后一半轮次算"后段"。记忆方案的差别应该主要体现
    在后段——前段几乎没有历史可记。v1 没有这个分层，所以无法判断观察到的差异
    是不是"记忆"带来的。
    """
    def rate(recs: list[dict], key: str) -> tuple[float | None, int, int]:
        judged = [r for r in recs if r.get(key) is not None]
        if not judged:
            return None, 0, len(recs)
        hits = sum(1 for r in judged if r[key])
        return hits / len(judged), len(judged), len(recs) - len(judged)

    non_probe = [r for r in records if not r.get("is_probe")]
    n = len(non_probe)
    cut = int(n * late_from_fraction)
    early, late = non_probe[:cut], non_probe[cut:]

    ooc_all, n_ooc_judged, n_ooc_fail = rate(non_probe, "ooc")
    fact_all, n_fact_judged, n_fact_fail = rate(non_probe, "fact_contradicts")
    overlaps = [r["evidence_overlap"] for r in non_probe if r.get("evidence_overlap") is not None]

    out = {
        "n_character_turns": n,
        "ooc_rate": ooc_all,
        "ooc_rate_early": rate(early, "ooc")[0],
        "ooc_rate_late": rate(late, "ooc")[0],
        "fact_consistency": None if fact_all is None else 1 - fact_all,
        "fact_consistency_early": (lambda v: None if v is None else 1 - v)(rate(early, "fact_contradicts")[0]),
        "fact_consistency_late": (lambda v: None if v is None else 1 - v)(rate(late, "fact_contradicts")[0]),
        "leakage_rate": (sum(1 for r in non_probe if r["leakage"]) / n) if n else None,
        "n_judge_fail_ooc": n_ooc_fail,
        "n_judge_fail_fact": n_fact_fail,
        "evidence_use_rate": (sum(1 for v in overlaps if v >= 2) / len(overlaps)) if overlaps else None,
        "mean_context_chars": _mean([r["context_chars"] for r in non_probe
                                     if isinstance(r.get("context_chars"), int)]),
    }
    return out


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# -------------------------------------------------------------- 记忆探针
def probe_recall_accuracy(client: OpenAI, judge_model: str, turns: list[dict],
                           probes: list[dict], judge_repeats: int = 3) -> tuple[float | None, list[dict]]:
    """记忆探针得分：对话后段问"我刚才跟你说过什么"，看角色答得对不对。

    这是 v2 新增的、唯一**直接**测记忆的指标。ooc_rate 和 fact_consistency 都
    只测"像不像这个角色"，不测"记不记得发生过什么"——一个完全没有记忆的模型
    只要人设稳就能拿高分。探针题目来自固定的用户发言（三种方案看到的用户台词
    完全相同），所以探针对所有方案是同一套题，可比。

    返回 (正确率, 逐题记录)。None 表示没有探针或全部判定失败。
    """
    probe_turns = [t for t in turns if t.get("is_probe") and t.get("speaker") != "user"]
    if not probe_turns or not probes:
        return None, []

    records = []
    for t, probe in zip(probe_turns, probes):
        prompt = PROBE_RECALL_JUDGE_PROMPT.format(
            question=probe.get("question", ""),
            expected=probe.get("expected_answer", ""),
            reply=t.get("text", ""),
        )
        correct, votes, reason = _vote_bool(client, judge_model, prompt, "correct", judge_repeats)
        records.append({
            "turn_id": t.get("turn_id"),
            "question": probe.get("question", ""),
            "expected_answer": probe.get("expected_answer", ""),
            "source_turn_index": probe.get("source_turn_index"),
            "correct": correct,
            "votes": "".join("1" if v else "0" for v in votes),
            "reason": reason.replace("\n", " "),
        })
    judged = [r for r in records if r["correct"] is not None]
    if not judged:
        return None, records
    return sum(1 for r in judged if r["correct"]) / len(judged), records


# -------------------------------------------------------------- v1 兼容封装
def character_fact_consistency(client: OpenAI, judge_model: str, turns: list[dict],
                                character_facts: list[dict], judge_repeats: int = 1) -> float:
    """v1 接口：只返回汇总的事实一致性。保留是为了让旧脚本和旧测试继续可用；
    主实验走 judge_turns() + aggregate_per_turn()，因为那条路会把逐轮判定存下来。
    """
    recs = judge_turns(client, judge_model, turns, "", character_facts,
                       judge_repeats=judge_repeats, ooc_prompt_version="v1", do_ooc=False)
    agg = aggregate_per_turn(recs)
    if agg["fact_consistency"] is None:
        raise RuntimeError("所有轮次的 judge 调用都失败了，检查 judge 模型服务是否正常")
    return agg["fact_consistency"]


def ooc_rate(client: OpenAI, judge_model: str, turns: list[dict], character_card: str,
             judge_repeats: int = 1, prompt_version: str = "v1") -> float:
    """v1 接口：只返回汇总的 OOC 标记率。注意命名——这是 **judge 的标记率**，
    审核显示精确率只有约 40%，不要当成真实出戏率（见 docs/JUDGE_AUDIT.md）。
    """
    recs = judge_turns(client, judge_model, turns, character_card, [],
                       judge_repeats=judge_repeats, ooc_prompt_version=prompt_version,
                       do_fact=False)
    agg = aggregate_per_turn(recs)
    if agg["ooc_rate"] is None:
        raise RuntimeError("所有轮次的 judge 调用都失败了，检查 judge 模型服务是否正常")
    return agg["ooc_rate"]


def relationship_state_accuracy(client: OpenAI, judge_model: str, turns: list[dict],
                                 trajectory: list[dict], window: int = 4,
                                 judge_repeats: int = 1) -> float:
    """关系状态准确率。relationship_trajectory 弱标签仍未实现（见
    docs/DATASET.md），所以这个函数对主实验数据必然抛 ValueError，由
    run_eval.py 捕获后在 summary.csv 里留空。保留实现是为了弱标签做出来之后
    可以直接接上。"""
    if not trajectory:
        raise ValueError("relationship_trajectory 为空，无法计算 relationship_state_accuracy")

    n_correct, n_judged = 0, 0
    for seg in trajectory:
        # 弱标签既可能给 (start_turn, end_turn) 区间，也可能只给一个 turn_id；
        # 后者取它往前 window 轮作为上下文。
        if "start_turn" in seg or "end_turn" in seg:
            lo, hi = seg.get("start_turn", 0), seg.get("end_turn", 0)
        else:
            hi = seg.get("turn_id", 0)
            lo = max(0, hi - window)
        seg_turns = [t for t in turns if lo <= (t.get("turn_id") or 0) <= hi]
        if not seg_turns:
            continue
        excerpt = "\n".join(f"{t['speaker']}: {t['text']}" for t in seg_turns)
        prompt = RELATIONSHIP_STATE_JUDGE_PROMPT.format(dialogue=excerpt)
        result = call_judge(client, judge_model, prompt)
        if isinstance(result, dict) and isinstance(result.get("score"), (int, float)):
            n_judged += 1
            if abs(float(result["score"]) - float(seg.get("score", 0))) <= 1:
                n_correct += 1
    if n_judged == 0:
        raise RuntimeError("关系状态判定全部失败")
    return n_correct / n_judged


def evidence_traceability_rate(memory) -> float:
    """**已废弃**：只检查"这一轮检索结果是不是非空"，对 vector/structured 恒为
    1.000，没有信息量。v1 报告里把它当结果列出来是不对的，v2 用
    evidence_use_rate（见 aggregate_per_turn）替代：看回复里是否真的出现了
    检索到的内容词。这个函数保留只为旧脚本不报错，不要用在新结果里。
    """
    if not getattr(memory, "supports_evidence", False):
        raise ValueError(f"{getattr(memory, 'name', type(memory).__name__)} 没有检索语义，"
                          f"不支持证据类指标")
    log = getattr(memory, "retrieval_log", None)
    if log is None:
        raise ValueError(f"{type(memory).__name__} 没有 retrieval_log，不支持这个指标")
    if not log:
        return 0.0

    def _has_evidence(entry: dict) -> bool:
        return bool(entry.get("retrieved_fact_ids") or entry.get("retrieved_turn_ids"))

    return sum(_has_evidence(e) for e in log) / len(log)
