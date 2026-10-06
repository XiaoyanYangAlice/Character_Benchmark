"""
四个指标的计算函数。判断器 prompt 在 judge_prompts.py。

三个需要 LLM-as-judge 的指标（character_fact_consistency / ooc_rate /
relationship_state_accuracy）都走同一套调用模式：每轮判断重复 judge_repeats
次、多数投票，减少单次判断的随机性；参数默认 judge_repeats=1（先用 1 跑通
整条链路、控制成本），正式出结果前建议按主项目文档"评测协议"一节的建议改成 3，
并在小样本上人工核对判断器的一致率。

evidence_traceability_rate 不调用 judge（见函数内文档字符串说明为什么，以及
这个代理指标的局限）。
"""
import json
import re
import time
from collections import Counter
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

from .judge_prompts import (
    FACT_CONSISTENCY_JUDGE_PROMPT,
    OOC_JUDGE_PROMPT,
    RELATIONSHIP_STATE_JUDGE_PROMPT,
)


def _extract_json(raw: str) -> dict | None:
    """跟 ontology_extraction/extract_character_features.py 里的同名逻辑一致
    （判断器也偶尔会包一层 markdown 代码块），没有抽成共享模块是因为两边分属
    「数据准备」和「运行时评测」两个独立阶段，各自独立演进更清楚，代码量也不大，
    重复一份的维护成本低于引入跨模块耦合。"""
    raw = raw.strip()
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


def call_judge(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> dict | None:
    """通用 judge 调用：跑 prompt，解析 JSON，失败重试。temperature=0 不是为了
    绝对确定性（judge_repeats 本来就假设有随机性可采样），而是减少"同一次调用
    内"格式跑偏的概率；跨次调用的随机性交给上层多次调用+多数投票。"""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            parsed = _extract_json(resp.choices[0].message.content)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"  [judge retry {attempt + 1}] API error: {e}")
        time.sleep(1)
    return None


def _majority_vote_bool(client: OpenAI, judge_model: str, prompt: str, key: str,
                         judge_repeats: int) -> bool | None:
    """对同一个 prompt 重复调用 judge_repeats 次，取多数投票。返回 None 表示
    judge_repeats 次全部调用失败（无法解析出合法 JSON），调用方应把这类样本
    单独统计、不计入分母，而不是悄悄当成 False——那样会系统性低估问题发生率。"""
    votes = []
    for _ in range(judge_repeats):
        result = call_judge(client, judge_model, prompt)
        if result is not None and key in result:
            votes.append(bool(result[key]))
    if not votes:
        return None
    return sum(votes) > len(votes) / 2


def character_fact_consistency(client: OpenAI, judge_model: str, turns: list[dict],
                                character_facts: list[dict], judge_repeats: int = 1) -> float:
    """1 - 与已知事实矛盾的轮次数 / 总轮次数（只看角色说的轮次，用户轮次不参与
    评判——用户说什么不代表角色"应该"知道或认同）。

    character_facts: structured_memory.load_character_facts() 产出的同一种
    dict 列表（{feature_type, span, sentence_id, modality, ...}）。
    """
    character_turns = [t for t in turns if t.get("speaker") != "user"]
    if not character_turns:
        raise ValueError("turns 里没有角色轮次，无法计算 character_fact_consistency")

    facts_text = "\n".join(
        f"- [{f.get('feature_type', '?')}] {f['span']} (modality={f.get('modality', 'present')})"
        for f in character_facts
    ) or "（无已知事实）"

    n_contradict, n_unjudged = 0, 0
    for t in character_turns:
        prompt = FACT_CONSISTENCY_JUDGE_PROMPT.format(facts=facts_text, turn_text=t["text"])
        vote = _majority_vote_bool(client, judge_model, prompt, "contradicts_known_fact", judge_repeats)
        if vote is None:
            n_unjudged += 1
            continue
        if vote:
            n_contradict += 1

    n_judged = len(character_turns) - n_unjudged
    if n_judged == 0:
        raise RuntimeError("所有轮次的 judge 调用都失败了，检查 judge 模型服务是否正常")
    if n_unjudged:
        print(f"[character_fact_consistency] 提醒：{n_unjudged}/{len(character_turns)} "
              f"轮的 judge 调用失败，已从分母里剔除，不是当成'不矛盾'处理。")
    return 1 - n_contradict / n_judged


def ooc_rate(client: OpenAI, judge_model: str, turns: list[dict], character_card: str,
             judge_repeats: int = 1) -> float:
    """OOC 轮次数 / 总轮次数（只看角色轮次）。"""
    character_turns = [t for t in turns if t.get("speaker") != "user"]
    if not character_turns:
        raise ValueError("turns 里没有角色轮次，无法计算 ooc_rate")

    n_ooc, n_unjudged = 0, 0
    for t in character_turns:
        prompt = OOC_JUDGE_PROMPT.format(character_card=character_card, turn_text=t["text"])
        vote = _majority_vote_bool(client, judge_model, prompt, "ooc", judge_repeats)
        if vote is None:
            n_unjudged += 1
            continue
        if vote:
            n_ooc += 1

    n_judged = len(character_turns) - n_unjudged
    if n_judged == 0:
        raise RuntimeError("所有轮次的 judge 调用都失败了，检查 judge 模型服务是否正常")
    if n_unjudged:
        print(f"[ooc_rate] 提醒：{n_unjudged}/{len(character_turns)} 轮的 judge 调用失败，已从分母里剔除。")
    return n_ooc / n_judged


def relationship_state_accuracy(client: OpenAI, judge_model: str, turns: list[dict],
                                 relationship_trajectory: list[dict], window: int = 4,
                                 judge_repeats: int = 1) -> float:
    """relationship_trajectory 是弱标签轨迹：[{"turn_id": int, "score": 1-5}, ...]
    （见 docs/DATASET.md，来自另一个 LLM 按固定 prompt 打分 + 抽样人工复核）。

    对每个锚点，把截止到该 turn_id 的最近 `window` 轮对话喂给 judge 重新打分，
    跟弱标签算 MAE，再线性映射成 0-1（1-5 分制下最大可能误差是 4 分）。

    返回的是"和弱标签的一致程度"，不是"关系状态判断的绝对准确率"——弱标签本身
    也是 LLM 打的、只抽样做了人工复核，这一点需要在报告里说清楚，不要把这个
    指标包装成比它实际能证明的更强的结论。
    """
    if not relationship_trajectory:
        raise ValueError("relationship_trajectory 为空，无法计算 relationship_state_accuracy")

    errors, n_unjudged = [], 0
    for anchor in relationship_trajectory:
        end_id = anchor["turn_id"]
        window_turns = [t for t in turns if t.get("turn_id", 0) <= end_id][-window:]
        if not window_turns:
            continue
        dialogue_text = "\n".join(f"{t['speaker']}: {t['text']}" for t in window_turns)

        scores = []
        for _ in range(judge_repeats):
            prompt = RELATIONSHIP_STATE_JUDGE_PROMPT.format(dialogue=dialogue_text)
            result = call_judge(client, judge_model, prompt)
            if result is not None and isinstance(result.get("score"), (int, float)):
                scores.append(result["score"])
        if not scores:
            n_unjudged += 1
            continue

        # 多次重复取众数（分数是有序离散值，众数比平均更抗离群判断）
        predicted = Counter(scores).most_common(1)[0][0]
        errors.append(abs(predicted - anchor["score"]))

    if not errors:
        raise RuntimeError("所有锚点的 judge 调用都失败了，检查 judge 模型服务是否正常")
    if n_unjudged:
        print(f"[relationship_state_accuracy] 提醒：{n_unjudged}/{len(relationship_trajectory)} "
              f"个锚点的 judge 调用失败，已跳过。")

    mae = sum(errors) / len(errors)
    return max(0.0, 1 - mae / 4)


def evidence_traceability_rate(memory) -> float:
    """只对方案 B（向量记忆）/ C（结构化记忆）有意义：这一轮生成时，记忆方案
    检索到的上下文是不是非空。

    这是一个**代理指标**，不是严格定义的"回复里的每个断言都能定位到具体证据"
    ——那需要 judge 模型逐句核对回复内容和被检索证据的对应关系，调用成本是
    这里的量级的好几倍（每轮至少多一次 judge 调用），不在当前 8 天预算内实现，
    作为后续改进方向记录在 docs/DATASET.md。

    当前版本能确定的是"下界"：如果这一轮检索结果是空的，那这一轮回复不可能有
    可追溯证据支撑（False 是可信的）；如果检索结果非空，只能说"有机会用上"，
    不保证真的用上了（True 不代表回复真的忠实于检索到的证据，可能只是恰好都
    非空）。写报告时这个区别务必讲清楚，不要把代理指标当成精确指标引用。

    memory: 一个记忆方案实例，需要有 retrieval_log 属性（vector_memory.py /
    structured_memory.py 都已经维护了这个字段，entry 里用 retrieved_turn_ids
    或 retrieved_fact_ids 记录本轮实际用到的证据 id）。raw_context 方案没有
    "检索"这个概念，不适用这个指标，调用前先用 memory.name 判断。
    """
    if memory.name == "raw_context":
        raise ValueError("raw_context 方案没有证据检索的概念，evidence_traceability_rate 不适用")

    log = getattr(memory, "retrieval_log", None)
    if log is None:
        raise ValueError(f"{memory.name} 没有维护 retrieval_log，先参考 vector_memory.py 补上再计算这个指标")
    if not log:
        return 0.0

    def _has_evidence(entry: dict) -> bool:
        return bool(entry.get("retrieved_turn_ids") or entry.get("retrieved_fact_ids"))

    n_traceable = sum(1 for entry in log if _has_evidence(entry))
    return n_traceable / len(log)
