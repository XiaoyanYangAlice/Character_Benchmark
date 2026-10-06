"""src/eval/metrics.py 的单元测试。不打真实 API，用一个假的 OpenAI 兼容客户端
模拟 judge 模型的响应，只验证「调用/解析/聚合」这条链路的逻辑是对的——判断器
本身的判断质量好不好，需要在真实模型上做人工一致率核对（见 judge_prompts.py
文件头的说明），单元测试证明不了这个。
"""
import json

from src.eval.metrics import (
    character_fact_consistency,
    evidence_traceability_rate,
    ooc_rate,
    relationship_state_accuracy,
)
from src.memory.raw_context import RawContextMemory
from src.memory.vector_memory import VectorMemory
from tests.fakes import FakeClient


def test_character_fact_consistency_counts_contradictions():
    turns = [
        {"turn_id": 1, "speaker": "user", "text": "你是做什么工作的"},
        {"turn_id": 2, "speaker": "character", "text": "我从来没学过医"},  # 应判矛盾
        {"turn_id": 3, "speaker": "character", "text": "今天天气不错"},     # 应判不矛盾
    ]
    facts = [{"feature_type": "biography", "span": "急诊科医生", "modality": "present"}]
    client = FakeClient([
        json.dumps({"contradicts_known_fact": True, "contradicted_fact": "急诊科医生", "reason": "..."}),
        json.dumps({"contradicts_known_fact": False, "contradicted_fact": "", "reason": "..."}),
    ])
    score = character_fact_consistency(client, "judge-model", turns, facts, judge_repeats=1)
    assert score == 0.5  # 2 条角色轮次里 1 条矛盾


def test_character_fact_consistency_excludes_user_turns():
    turns = [{"turn_id": 1, "speaker": "user", "text": "xxx"}]
    try:
        character_fact_consistency(FakeClient([]), "judge-model", turns, [])
    except ValueError as e:
        assert "角色轮次" in str(e)
    else:
        raise AssertionError("应该在没有角色轮次时报错")


def test_ooc_rate_majority_vote():
    turns = [{"turn_id": 1, "speaker": "character", "text": "作为一个AI，我不能..."}]
    # judge_repeats=3，2 票 True 1 票 False，多数投票应该判 True
    client = FakeClient([
        json.dumps({"ooc": True, "reason": "..."}),
        json.dumps({"ooc": True, "reason": "..."}),
        json.dumps({"ooc": False, "reason": "..."}),
    ])
    rate = ooc_rate(client, "judge-model", turns, character_card="一个古代侠客", judge_repeats=3)
    assert rate == 1.0


def test_relationship_state_accuracy_computes_mae_based_score():
    turns = [
        {"turn_id": 1, "speaker": "user", "text": "你好"},
        {"turn_id": 2, "speaker": "character", "text": "你好呀"},
    ]
    trajectory = [{"turn_id": 2, "score": 3}]
    client = FakeClient([json.dumps({"score": 3, "reason": "..."})])
    acc = relationship_state_accuracy(client, "judge-model", turns, trajectory, window=4, judge_repeats=1)
    assert acc == 1.0  # 预测和标签完全一致，MAE=0


def test_evidence_traceability_rate_uses_retrieval_log():
    mem = VectorMemory(top_k=1)
    mem.on_new_turn({"turn_id": 1, "speaker": "character", "text": "我喜欢弹吉他"})
    mem.retrieve([], {"speaker": "user", "text": "你喜欢弹吉他吗"})  # 命中，非空检索
    mem.retrieval_log.append({"query": "无关问题", "retrieved_turn_ids": []})  # 手动追加一条空检索模拟未命中

    rate = evidence_traceability_rate(mem)
    assert rate == 0.5  # 2 次 retrieve 里 1 次有证据


def test_evidence_traceability_rate_rejects_raw_context():
    mem = RawContextMemory()
    try:
        evidence_traceability_rate(mem)
    except ValueError as e:
        assert "raw_context" in str(e)
    else:
        raise AssertionError("raw_context 方案不该支持这个指标，应该报错")
