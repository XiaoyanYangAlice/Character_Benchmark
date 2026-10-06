"""最小单元测试：证明代码能跑，不只是截图。"""
from src.memory.raw_context import RawContextMemory
from src.memory.structured_memory import StructuredEventMemory
from src.memory.vector_memory import VectorMemory


def test_raw_context_truncates():
    mem = RawContextMemory(max_turns=2)
    history = [
        {"turn_id": 1, "speaker": "user", "text": "第一轮开场白"},
        {"turn_id": 2, "speaker": "character", "text": "第二轮回应"},
        {"turn_id": 3, "speaker": "user", "text": "今天天气怎么样"},
    ]
    ctx = mem.retrieve(history, {"speaker": "user", "text": "今天天气怎么样"})
    assert "第一轮开场白" not in ctx  # 第一轮应该被截掉
    assert "今天天气怎么样" in ctx
    assert mem.dropped_turns_log[-1] == 1


def test_raw_context_no_truncation_needed():
    mem = RawContextMemory(max_turns=20)
    history = [{"turn_id": 1, "speaker": "user", "text": "hi"}]
    mem.retrieve(history, {"speaker": "user", "text": "hi"})
    assert mem.dropped_turns_log[-1] == 0


def test_vector_memory_retrieves_lexically_relevant_turn():
    # 本地开发环境没装 sentence-transformers，这个测试实际跑的是词重叠退化路径
    # （_use_lexical_fallback=True），用词汇完全不重叠的三轮历史保证检索结果
    # 是确定性的，不依赖真实 embedding 模型也能验证 retrieve/on_new_turn 的逻辑对不对。
    mem = VectorMemory(top_k=1)
    mem.on_new_turn({"turn_id": 1, "speaker": "character", "text": "我最喜欢弹吉他和写歌"})
    mem.on_new_turn({"turn_id": 2, "speaker": "character", "text": "昨天去医院复查了一下过敏"})
    mem.on_new_turn({"turn_id": 3, "speaker": "character", "text": "周末打算去爬山看日出"})

    ctx = mem.retrieve([], {"speaker": "user", "text": "你复查过敏的结果怎么样"})
    assert "过敏" in ctx
    assert "吉他" not in ctx

    assert mem.retrieval_log[-1]["retrieved_turn_ids"] == [2]


def test_vector_memory_empty_store_returns_empty_context():
    mem = VectorMemory(top_k=3)
    ctx = mem.retrieve([], {"speaker": "user", "text": "hi"})
    assert ctx == ""
    assert mem.retrieval_log[-1]["retrieved_turn_ids"] == []


def test_structured_memory_load_and_retrieve_logs_fact_ids():
    import json
    import tempfile
    from pathlib import Path

    records = [
        {"story_id": "s1", "feature_type": "biography", "character": "姜遥",
         "span": "急诊科医生", "sentence_id": 1, "modality": "present", "lang": "zh"},
        {"story_id": "s1", "feature_type": "personality", "character": "姜遥",
         "span": "冷静", "sentence_id": 2, "modality": "present", "lang": "zh"},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "facts.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

        mem = StructuredEventMemory()
        mem.load_character_facts(path)
        assert len(mem.character_facts["姜遥"]) == 2

        ctx = mem.retrieve([], {"speaker": "姜遥"})
        assert "急诊科医生" in ctx
        assert mem.retrieval_log[-1]["retrieved_fact_ids"] == ["姜遥#0", "姜遥#1"]


def test_structured_memory_normalizes_invalid_modality():
    import json
    import tempfile
    from pathlib import Path

    records = [
        {"story_id": "s1", "feature_type": "emotion", "character": "I",
         "span": "will find out", "sentence_id": 1, "modality": "future", "lang": "en"},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "facts.jsonl"
        path.write_text(json.dumps(records[0]), encoding="utf-8")

        mem = StructuredEventMemory()
        mem.load_character_facts(path)
        assert mem.character_facts["I"][0]["modality"] == "present"


def test_structured_memory_from_records_does_not_leak_other_stories_with_same_label():
    # 实测发现的真实 bug 场景：61 段 dialogues.jsonl 里有 20 段 canonical 标签都
    # 是同一个占位词 "I"（第一人称角色卡没写真名，见 docs/DATASET.md）。如果每条
    # dialogue 都用 load_character_facts() 指向同一份 character_features_en.jsonl，
    # 这 20 个互不相关故事的事实会被合并进同一个 character_facts["I"]。
    # run_eval.py 应该用 load_character_facts_from_records() 只导入这一条
    # dialogue 自己的 fact_labels，这个测试证明这样不会把另一个故事的事实带进来。
    gura_facts = [
        {"story_id": "pippa_gura", "feature_type": "biography", "character": "I",
         "span": "a shark", "sentence_id": 1, "modality": "present", "lang": "en"},
    ]
    monika_facts = [
        {"story_id": "pippa_monika", "feature_type": "biography", "character": "I",
         "span": "president of the literature club", "sentence_id": 1, "modality": "present", "lang": "en"},
    ]

    mem = StructuredEventMemory()
    mem.load_character_facts_from_records(gura_facts)

    assert len(mem.character_facts["I"]) == 1
    assert mem.character_facts["I"][0]["story_id"] == "pippa_gura"
    assert "president of the literature club" not in str(mem.character_facts["I"])

    ctx = mem.retrieve([], {"speaker": "I"})
    assert "a shark" in ctx
    assert "literature club" not in ctx

    # 确认没有意外改到调用方传进来的原始 list（同一份 dict 被复用会引发更隐蔽的 bug）
    assert monika_facts[0]["character"] == "I"
