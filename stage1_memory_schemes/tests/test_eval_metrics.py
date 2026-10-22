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


def test_supplementary_stats_truncation_and_holm():
    from src.eval.supplementary_stats import dropped_per_turn, holm
    # 窗口 20：第 10 个用户轮次 history 长度 19，不截断；第 11 个长度 21，丢 1 轮
    d = dropped_per_turn(12, 20)
    assert d[9] == 0 and d[10] == 1 and d[11] == 3
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


# ---------------------------------------------------------------- v2 新增
def test_leakage_detector_fires_on_audited_real_ooc_patterns():
    """规则检测器的设计依据是 v1 审核的发现：真正抓到的出戏几乎全是安全说教腔、
    通用拒绝腔和模板残留。这几条是审核里真实出现过的形态。"""
    from src.eval.metrics import detect_assistant_leakage
    assert detect_assistant_leakage("As an AI language model, I can't do that.")
    assert detect_assistant_leakage("Safety always comes first, let's take a break.")
    assert detect_assistant_leakage("assistant: sure, here you go")


def test_leakage_detector_ignores_in_character_violence():
    """judge 的系统性误报是『把符合设定的行为当成出戏』。规则检测器不能重复
    这个错误——它只认助手腔，不认内容尺度。"""
    from src.eval.metrics import detect_assistant_leakage
    assert detect_assistant_leakage("I grabbed his collar and slammed him into the wall.") == []
    assert detect_assistant_leakage("My programming won't let me feel pain, human.") == []


def test_aggregate_splits_early_and_late_turns():
    """记忆方案的差别应该出现在对话后段。v1 没有这个分层，所以无法判断
    观察到的差异是不是记忆带来的。"""
    from src.eval.metrics import aggregate_per_turn
    recs = [{"turn_index": i, "is_probe": False, "ooc": i > 2, "fact_contradicts": False,
             "leakage": 0, "context_chars": 10, "evidence_overlap": None} for i in range(1, 5)]
    agg = aggregate_per_turn(recs, late_from_fraction=0.5)
    assert agg["ooc_rate"] == 0.5
    assert agg["ooc_rate_early"] == 0.0   # 前两轮都没标
    assert agg["ooc_rate_late"] == 1.0    # 后两轮都标了


def test_aggregate_excludes_probe_turns_from_ooc_denominator():
    """探针轮次是我们自己插的提问，不该算进出戏率的分母。"""
    from src.eval.metrics import aggregate_per_turn
    recs = [
        {"turn_index": 1, "is_probe": False, "ooc": False, "fact_contradicts": False,
         "leakage": 0, "context_chars": 1, "evidence_overlap": None},
        {"turn_index": 2, "is_probe": True, "ooc": True, "fact_contradicts": True,
         "leakage": 0, "context_chars": 1, "evidence_overlap": None},
    ]
    agg = aggregate_per_turn(recs)
    assert agg["n_character_turns"] == 1
    assert agg["ooc_rate"] == 0.0


def test_aggregate_does_not_count_judge_failures_as_passes():
    """judge 判定失败必须从分母里剔除，不能当成『没出戏』——那会系统性低估。"""
    from src.eval.metrics import aggregate_per_turn
    recs = [
        {"turn_index": 1, "is_probe": False, "ooc": True, "fact_contradicts": None,
         "leakage": 0, "context_chars": 1, "evidence_overlap": None},
        {"turn_index": 2, "is_probe": False, "ooc": None, "fact_contradicts": None,
         "leakage": 0, "context_chars": 1, "evidence_overlap": None},
    ]
    agg = aggregate_per_turn(recs)
    assert agg["ooc_rate"] == 1.0        # 1 条有效判定，标了 1 条
    assert agg["n_judge_fail_ooc"] == 1


def test_evidence_use_rate_ignores_stopword_overlap():
    """旧的 evidence_traceability 只看『检索结果非空』，恒为 1.000。新指标看
    回复里是否真的出现了检索到的内容词；虚词重合不算数，否则又会恒为 1。"""
    from src.eval.metrics import _evidence_overlap
    entry = {"retrieved_spans": ["She is a trained paramedic from Oslo"]}
    assert _evidence_overlap(entry, "and the that with for you are") == 0
    assert _evidence_overlap(entry, "I trained as a paramedic back in Oslo") >= 2
    # 没有检索内容时返回 None（区别于『检索到了但没用上』= 0）
    assert _evidence_overlap({"retrieved_spans": None}, "anything") is None


def test_probe_recall_accuracy_scores_only_probe_turns():
    from src.eval.metrics import probe_recall_accuracy
    turns = [
        {"turn_id": 1, "speaker": "Alice", "text": "普通回复", "is_probe": False},
        {"turn_id": 3, "speaker": "Alice", "text": "你妹妹叫 Anna", "is_probe": True},
    ]
    probes = [{"question": "我妹妹叫什么", "expected_answer": "Anna", "source_turn_index": 1}]
    client = FakeClient([json.dumps({"reason": "对上了", "correct": True})])
    acc, recs = probe_recall_accuracy(client, "judge-model", turns, probes, judge_repeats=1)
    assert acc == 1.0
    assert len(recs) == 1 and recs[0]["expected_answer"] == "Anna"


def test_judge_turns_can_use_fewer_repeats_for_fact_than_ooc():
    """judge 调用是实验的主要时间成本。OOC 受 judge 噪声影响大、保留 3 次投票；
    事实一致性接近天花板、区分度低，1 次即可。这里断言两者确实用了不同次数。"""
    from src.eval.metrics import judge_turns

    class CountingClient:
        def __init__(self):
            self.n_ooc = self.n_fact = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            prompt = kw["messages"][-1]["content"]
            if "出戏" in prompt:
                self.n_ooc += 1
                content = json.dumps({"reason": "r", "ooc": False})
            else:
                self.n_fact += 1
                content = json.dumps({"reason": "r", "contradicts_known_fact": False})

            class M:
                pass
            m = M()
            m.content = content

            class C:
                pass
            c = C()
            c.message = m

            class R:
                pass
            r = R()
            r.choices = [c]
            return r

    client = CountingClient()
    judge_turns(client, "judge", [{"turn_id": 1, "speaker": "A", "text": "x"}],
                "card", [], judge_repeats=3, judge_repeats_fact=1)
    assert client.n_ooc == 3
    assert client.n_fact == 1


def test_as_list_tolerates_bare_array_from_judge():
    """实测 Qwen2.5-14B 有时无视 schema，直接返回裸数组 [...] 而不是
    {"probes": [...]}。第一次跑 make_probes.py 时，一条这样的响应让
    ThreadPoolExecutor 抛 AttributeError，47 条对话全部白跑。"""
    from src.eval.metrics import as_list
    assert as_list([{"question": "q"}], "probes") == [{"question": "q"}]
    assert as_list({"probes": [{"question": "q"}]}, "probes") == [{"question": "q"}]
    assert as_list(None, "probes") == []
    assert as_list({"wrong_key": [1]}, "probes") == []
    assert as_list("not json at all", "probes") == []


def test_vote_bool_treats_bare_array_as_judge_failure_not_as_false():
    """judge 返回裸数组时，`key in result` 在 list 上是合法的成员判断（永远
    False），会被误计成"判定为 False"。必须当成判定失败从分母里剔除。"""
    from src.eval.metrics import judge_turns
    client = FakeClient([json.dumps([{"ooc": True}])])
    recs = judge_turns(client, "j", [{"turn_id": 1, "speaker": "A", "text": "x"}],
                       "card", [], judge_repeats=1, do_fact=False)
    assert recs[0]["ooc"] is None
