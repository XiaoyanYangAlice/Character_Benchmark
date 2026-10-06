"""src/eval/run_eval.py 里不需要真实 vLLM/judge 服务的部分单独测试：
build_memory() 的三路分发对不对、character_card_text() 的 display_name 优先级、
load_dialogues() 能不能正确解析。run_one()/main() 需要真实（或至少假的）
OpenAI client 跑完整流程，那部分只能在装了 openai 包的环境（SURF）里用
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
