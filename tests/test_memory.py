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


# ---------------------------------------------------------------- v2 新增
def test_no_memory_returns_empty_context():
    from src.memory.no_memory import NoMemory
    mem = NoMemory()
    ctx = mem.retrieve([{"turn_id": 1, "speaker": "user", "text": "x"}], {"speaker": "Ayaka"})
    assert ctx == ""
    assert len(mem.retrieval_log) == 1


def test_shuffled_facts_relabels_donor_facts_so_retrieval_is_not_empty():
    """对照组必须真的注入内容。如果忘了把捐赠事实挂到当前角色名下，retrieve()
    按 speaker 查表会查不到任何东西、静默退化成空上下文——那 shuffled_facts
    就变成了 no_memory 的复制品，对照完全失效且不报错。这正是 v1 踩过的
    『传错 speaker 导致记忆全程为空』那个坑的同一形态。"""
    from src.memory.shuffled_facts import ShuffledFactsMemory
    donor = [{"character": "Bob", "span": "是一名宇航员", "modality": "present", "story_id": "s2"}]
    mem = ShuffledFactsMemory()
    mem.load_donor_facts(donor, character_name="Alice")

    ctx = mem.retrieve([], {"speaker": "Alice"})
    assert "宇航员" in ctx
    assert mem.retrieval_log[-1]["retrieved_fact_ids"] == ["Alice#0"]
    # 留痕：事后要能查出这条事实原本属于谁
    assert mem.character_facts["Alice"][0]["donor_story_id"] == "s2"


def test_structured_memory_on_new_turn_accumulates_events_from_extractor():
    """v1 的 on_new_turn 是空实现，结构化记忆只有开局导入的静态事实，对话里
    新出现的信息一律不进记忆——那测的就不是记忆。这里断言抽取器拿到的是
    『一个完整来回』，并且抽出来的事件会出现在后续检索结果里。"""
    from src.memory.structured_memory import StructuredEventMemory
    seen = []

    def fake_extractor(exchange):
        seen.append(exchange)
        return [{"who": "user", "kind": "fact", "content": "用户的妹妹叫 Anna"}]

    mem = StructuredEventMemory(event_extractor=fake_extractor)
    mem.on_new_turn({"turn_id": 1, "speaker": "user", "text": "我妹妹叫 Anna"})
    assert seen == []  # 只收到用户轮次时还不能抽取，要等角色回复
    mem.on_new_turn({"turn_id": 2, "speaker": "Alice", "text": "Anna 这名字真好听"})

    assert len(seen) == 1
    assert seen[0]["user_turn"]["text"] == "我妹妹叫 Anna"
    assert seen[0]["character_turn"]["text"] == "Anna 这名字真好听"
    assert "Anna" in mem.retrieve([], {"speaker": "Alice"})


def test_structured_memory_survives_extractor_failure():
    """抽取失败不能让整段对话白跑，但必须计数，否则会出现『事件记忆全程为空
    却没人发现』——这正是 v1 那类静默失败的形态。"""
    from src.memory.structured_memory import StructuredEventMemory

    def boom(exchange):
        raise RuntimeError("judge 服务挂了")

    mem = StructuredEventMemory(event_extractor=boom)
    mem.on_new_turn({"turn_id": 1, "speaker": "user", "text": "x"})
    mem.on_new_turn({"turn_id": 2, "speaker": "Alice", "text": "y"})
    assert mem.events == []
    assert mem.extraction_failures == 1


def test_structured_memory_respects_retrieval_budget():
    """检索预算必须生效，否则 structured 注入的上下文会比别的方案长很多，
    方案之间比的就变成了上下文长度。"""
    from src.memory.structured_memory import StructuredEventMemory
    facts = [{"character": "Alice", "span": f"fact{i}", "modality": "present"} for i in range(20)]
    mem = StructuredEventMemory(max_facts=3)
    mem.load_character_facts_from_records(facts)
    ctx = mem.retrieve([], {"speaker": "Alice"})
    assert "fact2" in ctx and "fact3" not in ctx


# ---------------------------------------------------------------- 混合检索
def test_hybrid_retrieval_keeps_relevant_early_events():
    """诊断发现：按时间近远取最近 8 条，会把探针要考的早期信息全部挤掉
    （一段对话约 43 条事件，只给 8 条）。hybrid 必须能把与当前问题相关的
    早期事件捞回来，同时保留几条最近的维持对话连贯。"""
    from src.memory.structured_memory import StructuredEventMemory

    mem = StructuredEventMemory(max_events=4, retrieval="hybrid", n_recent=2)
    mem.events = [
        {"turn_id": 3, "who": "user", "kind": "fact",
         "content": "The user's sister is named Anna."},                 # 早期但相关
        {"turn_id": 5, "who": "Kasey", "kind": "event", "content": "walked to class"},
        {"turn_id": 7, "who": "Kasey", "kind": "event", "content": "ate a sandwich"},
        {"turn_id": 9, "who": "Kasey", "kind": "event", "content": "opened the door"},
        {"turn_id": 11, "who": "Kasey", "kind": "event", "content": "sat down"},
        {"turn_id": 13, "who": "Kasey", "kind": "event", "content": "looked outside"},
    ]
    out = mem._select_events({"text": "What is my sister called?"})
    contents = [e["content"] for e in out]
    assert len(out) == 4
    assert "The user's sister is named Anna." in contents, \
        f"相关的早期事件应当被捞回来，实际取到：{contents}"
    assert contents[-1] == "looked outside", "最近的事件应当保留，且按时间排在最后"
    assert [e["turn_id"] for e in out] == sorted(e["turn_id"] for e in out), \
        "注入顺序必须按时间，否则读起来不像一条时间线"


def test_recency_retrieval_is_unchanged_by_default():
    """默认行为必须和 v2 一致，否则旧结果无法复现。"""
    from src.memory.structured_memory import StructuredEventMemory

    mem = StructuredEventMemory(max_events=2)
    mem.events = [{"turn_id": i, "who": "user", "content": f"e{i}"} for i in (1, 3, 5)]
    out = mem._select_events({"text": "e1"})
    assert [e["content"] for e in out] == ["e3", "e5"]


def test_hybrid_prefers_user_events_when_overlap_ties():
    """重合度打平时优先用户相关的事件——探针问的全是用户说过的话。"""
    from src.memory.structured_memory import StructuredEventMemory

    mem = StructuredEventMemory(max_events=1, retrieval="hybrid", n_recent=0)
    mem.events = [
        {"turn_id": 2, "who": "Kasey", "content": "zzz unrelated one"},
        {"turn_id": 4, "who": "user", "content": "zzz unrelated two"},
    ]
    out = mem._select_events({"text": "something entirely different"})
    assert out[0]["who"] == "user"
