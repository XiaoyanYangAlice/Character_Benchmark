"""src/eval/run_eval.py 里不需要真实 vLLM/judge 服务的部分单独测试：
build_memory() 的三路分发对不对、character_card_text() 的 display_name 优先级、
load_dialogues() 能不能正确解析。run_one()/main() 需要真实（或至少假的）
OpenAI client 跑完整流程，那部分只能在装了 openai 包的环境（集群上）里用
--dry-run 或真跑一遍验证，不在这里测。"""
import json
import tempfile
from pathlib import Path

from src.eval.run_eval import build_memory, character_card_text, load_dialogues
from src.memory.raw_context import RawContextMemory
from src.memory.structured_memory import StructuredEventMemory
from src.memory.vector_memory import VectorMemory

DIALOGUE = {
    "dialogue_id": "pippa_test",
    "character_card": {"name": "I", "display_name": "Yuki", "description": "喜欢唱歌。", "greeting": "你好。"},
    "scripted_user_turns": ["你好", "你喜欢什么"],
    "fact_labels": [
        {"story_id": "pippa_test", "feature_type": "biography", "character": "I",
         "span": "喜欢唱歌", "sentence_id": 1, "modality": "present", "lang": "zh"},
    ],
    "relationship_trajectory": [],
}


def test_build_memory_raw_context_uses_configured_max_turns():
    mem = build_memory("raw_context", {"raw_context": {"max_turns": 5}}, DIALOGUE)
    assert isinstance(mem, RawContextMemory)
    assert mem.max_turns == 5


def test_build_memory_vector_memory_uses_configured_top_k():
    mem = build_memory("vector_memory", {"vector_memory": {"top_k": 3}}, DIALOGUE)
    assert isinstance(mem, VectorMemory)
    assert mem.top_k == 3


def test_build_memory_structured_memory_loads_only_this_dialogues_facts():
    other_dialogue_same_label = {
        "fact_labels": [
            {"story_id": "pippa_other", "feature_type": "biography", "character": "I",
             "span": "完全不相关的另一个故事", "sentence_id": 1, "modality": "present", "lang": "zh"},
        ],
    }
    mem_a = build_memory("structured_memory", {}, DIALOGUE)
    mem_b = build_memory("structured_memory", {}, other_dialogue_same_label)

    assert isinstance(mem_a, StructuredEventMemory)
    # 两个不同 dialogue 各自的 memory 实例互不串味，哪怕 canonical 标签一样都是 "I"
    assert [f["span"] for f in mem_a.character_facts["I"]] == ["喜欢唱歌"]
    assert [f["span"] for f in mem_b.character_facts["I"]] == ["完全不相关的另一个故事"]


def test_build_memory_rejects_unknown_scheme():
    try:
        build_memory("not_a_real_scheme", {}, DIALOGUE)
    except ValueError as e:
        assert "not_a_real_scheme" in str(e)
    else:
        raise AssertionError("未知记忆方案应该报错，而不是静默跑出无意义的结果")


def test_character_card_text_prefers_display_name_over_canonical_label():
    text = character_card_text(DIALOGUE["character_card"])
    assert "Yuki" in text
    assert "角色名：I" not in text  # 不应该把给事实检索用的占位标签暴露给 judge
    assert "喜欢唱歌" in text


def test_character_card_text_falls_back_to_name_when_no_display_name():
    text = character_card_text({"name": "姜遥", "description": "急诊科医生"})
    assert "姜遥" in text


def test_load_dialogues_parses_jsonl():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "dialogues.jsonl"
        path.write_text(
            json.dumps(DIALOGUE, ensure_ascii=False) + "\n" + json.dumps(DIALOGUE, ensure_ascii=False),
            encoding="utf-8",
        )
        dialogues = load_dialogues(path)
    assert len(dialogues) == 2
    assert dialogues[0]["dialogue_id"] == "pippa_test"


# ---------------------------------------------------------------- v2 新增
def _dialogue(did, facts, name="Alice"):
    return {"dialogue_id": did, "character_card": {"name": name, "description": "d"},
            "scripted_user_turns": ["x"], "fact_labels": facts, "relationship_trajectory": []}


def test_build_memory_supports_control_schemes():
    from src.eval.run_eval import build_memory
    d1 = _dialogue("d1", [{"character": "Alice", "span": "会弹吉他", "modality": "present"}])
    d2 = _dialogue("d2", [{"character": "Bob", "span": "是宇航员", "modality": "present"}], name="Bob")

    assert build_memory("no_memory", {}, d1, [d1, d2]).retrieve([], {"speaker": "Alice"}) == ""

    mem = build_memory("shuffled_facts", {}, d1, [d1, d2])
    ctx = mem.retrieve([], {"speaker": "Alice"})
    assert "宇航员" in ctx and "弹吉他" not in ctx  # 注入的是别人的事实
    assert mem.donor_dialogue_id == "d2"


def test_pick_donor_is_deterministic_and_never_self():
    """捐赠者必须确定性地挑：否则 shuffled 这一列的方差里会混进『这次恰好配到
    一个事实特别多/特别少的捐赠者』，与记忆方案无关。"""
    from src.eval.run_eval import pick_donor
    ds = [_dialogue(f"d{i}", [{"character": "C", "span": f"s{i}", "modality": "present"}] * (i + 1))
          for i in range(6)]
    first = pick_donor(ds[0], ds)
    assert first["dialogue_id"] != "d0"
    assert all(pick_donor(ds[0], ds)["dialogue_id"] == first["dialogue_id"] for _ in range(5))


def test_make_probes_validation_rejects_answers_the_user_never_said():
    """探针答案必须真的出现在它声称的那条用户发言里。模型很容易编出一个看起来
    合理但用户没说过的答案，那种题目无论角色怎么答都是错的，会系统性压低所有
    方案的探针分数。"""
    from src.generate.make_probes import validate_probe
    turns = ["我妹妹叫 Anna，她在鹿特丹念书", "今天有点累"]
    assert validate_probe({"source_turn_index": 1, "question": "我妹妹叫什么",
                           "expected_answer": "Anna"}, turns, 2)
    assert not validate_probe({"source_turn_index": 1, "question": "我妹妹叫什么",
                               "expected_answer": "Sophie"}, turns, 2)
    assert not validate_probe({"source_turn_index": 5, "question": "x",
                               "expected_answer": "Anna"}, turns, 2)


def test_probe_validation_rejects_answers_leaked_into_the_question():
    """实测第一次生成的 65 个探针里多数是这种退化形态：模型把用户原话截一段
    改成疑问句，"标准答案"就是那一段本身。照抄问题就能答对，完全不测记忆。
    下面前四条是真实生成出来的例子，后两条是同一批里合格的。"""
    from src.generate.make_probes import validate_probe
    turns = ["I'm going back to Liyue tomorrow and Dimo will arrive to Liyue too, "
             "maybe they think we are partner, do we need to ask someone to help us",
             "I am eating chikin nuggies and airport staff think I am a lost child"]

    def ok(q, a, idx=1):
        return validate_probe({"source_turn_index": idx, "question": q, "expected_answer": a},
                              turns, 2)

    assert not ok("Are you going back to Liyue?", "going back to Liyue")
    assert not ok("Did Dimo arrive to Liyue?", "arrive to Liyue")
    assert not ok("What if they think we are...hmmm...partner?", "partner")
    assert not ok("Do we need to ask someone to help us?", "someone to help us")
    assert ok("What am I eating right now?", "chikin nuggies", idx=2)
    assert ok("What do airport staff think I am when I walk around?", "a lost child", idx=2)
