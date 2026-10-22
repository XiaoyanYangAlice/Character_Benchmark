"""src/generate/dialogue_runner.py 的单元测试：验证"跑一段对话"这条粘合逻辑
本身是对的（轮次交替、turn_id 递增、memory 钩子在正确的时机被调用），不测生成
内容的质量——那需要真实模型。"""
from src.generate.dialogue_runner import build_system_prompt, run_dialogue
from src.memory.raw_context import RawContextMemory
from src.memory.vector_memory import VectorMemory
from src.memory.structured_memory import StructuredEventMemory
from tests.fakes import FakeClient


CHARACTER_CARD = {
    "name": "姜遥",
    "description": "急诊科医生，冷静专业。",
    "greeting": "你好，我是姜遥。",
}


def test_build_system_prompt_includes_name_and_description():
    prompt = build_system_prompt(CHARACTER_CARD)
    assert "姜遥" in prompt
    assert "急诊科医生" in prompt
    assert "AI" in prompt  # 明确要求不能暴露自己是 AI


def test_build_system_prompt_prefers_display_name_over_generic_pronoun_label():
    # 实测发现：PIPPA 第一人称角色卡抽取出来的 canonical 标签是代词（"I"/
    # "Me"），直接拼进 "You are role-playing as I" 语法是破的。有 display_name
    # 时优先用它拼文字，name 字段本身不受影响（fact 检索仍然认 name）。
    card = {"name": "I", "display_name": "Yuki", "description": "喜欢唱歌。"}
    prompt = build_system_prompt(card)
    assert "role-playing as Yuki" in prompt
    assert "as I." not in prompt
    assert card["name"] == "I"  # 确认这个字段没被函数改掉


def test_build_system_prompt_falls_back_to_name_when_no_display_name():
    # 没有 display_name 字段时（比如老数据、或 canonical 标签本来就是真名），
    # 照旧退回用 name，行为跟改动前一致。
    prompt = build_system_prompt(CHARACTER_CARD)
    assert "role-playing as 姜遥" in prompt


def test_run_dialogue_alternates_turns_and_uses_character_name_as_speaker():
    client = FakeClient(["（冷静地）有什么症状？", "先坐下，我看看。"])
    mem = RawContextMemory(max_turns=20)
    turns = run_dialogue(mem, client, "fake-model", CHARACTER_CARD,
                          scripted_user_turns=["我肚子疼", "已经疼了三天了"])

    assert len(turns) == 4  # 2 轮用户 + 2 轮角色
    assert [t["speaker"] for t in turns] == ["user", "姜遥", "user", "姜遥"]
    assert [t["turn_id"] for t in turns] == [1, 2, 3, 4]
    assert turns[1]["text"] == "（冷静地）有什么症状？"
    assert turns[3]["text"] == "先坐下，我看看。"


def test_run_dialogue_rejects_empty_scripted_turns():
    mem = RawContextMemory()
    try:
        run_dialogue(mem, FakeClient([]), "fake-model", CHARACTER_CARD, scripted_user_turns=[])
    except ValueError as e:
        assert "scripted_user_turns" in str(e)
    else:
        raise AssertionError("空的 scripted_user_turns 应该报错，而不是静默跑出空对话")


def test_run_dialogue_feeds_vector_memory_on_new_turn():
    client = FakeClient(["回复一", "回复二"])
    mem = VectorMemory(top_k=2)
    turns = run_dialogue(mem, client, "fake-model", CHARACTER_CARD,
                          scripted_user_turns=["第一句话", "第二句话"])

    # 4 轮（2 用户 + 2 角色）都应该被存进 memory，供后续检索
    assert len(mem._store) == 4
    # 每个用户轮次触发一次 retrieve()，一共 2 次
    assert len(mem.retrieval_log) == 2
    assert turns[-1]["speaker"] == "姜遥"


def test_run_dialogue_passes_character_name_as_speaker_to_structured_memory():
    # 回归测试：这个 bug 曾经导致 evidence_traceability_rate 对 structured_
    # memory 永远算出 0.000（全量实验 366 行跑完后发现的）。根因是 run_dialogue()
    # 把 user_turn（speaker 永远是 "user"）传给了 retrieve()，而
    # structured_memory.retrieve() 是按 current_turn["speaker"] 去查
    # self.character_facts 的——传错了就永远查不中任何角色的事实表，
    # retrieved_fact_ids 永远是空列表，retrieve() 并不报错，是静默退化。
    # 这里直接断言 character_facts 里登记过的事实真的被检索到了。
    client = FakeClient(["回复一", "回复二"])
    mem = StructuredEventMemory()
    mem.load_character_facts_from_records([
        {"character": "姜遥", "span": "急诊科医生", "modality": "present", "evidence": "..."},
    ])
    run_dialogue(mem, client, "fake-model", CHARACTER_CARD,
                 scripted_user_turns=["第一句话", "第二句话"])

    assert len(mem.retrieval_log) == 2
    for entry in mem.retrieval_log:
        assert entry["speaker"] == "姜遥"  # 不能是 "user"
        assert entry["retrieved_fact_ids"] != []  # 事实真的被查到了，不是静默退化成空


def test_run_dialogue_passes_seed_to_model_only_when_given():
    # 重复实验靠 seed 让每次采样独立可复现；不传 seed 时不能往请求里塞这个字段。
    class RecordingClient(FakeClient):
        def __init__(self, responses):
            super().__init__(responses)
            self.seen = []
            inner = self.chat.completions
            orig = inner.create

            def create(**kwargs):
                self.seen.append(kwargs)
                return orig(**kwargs)
            inner.create = create

    c1 = RecordingClient(["a"])
    run_dialogue(RawContextMemory(), c1, "m", CHARACTER_CARD, ["hi"], seed=7)
    assert c1.seen[0]["seed"] == 7

    c2 = RecordingClient(["a"])
    run_dialogue(RawContextMemory(), c2, "m", CHARACTER_CARD, ["hi"])
    assert "seed" not in c2.seen[0]


# ---------------------------------------------------------------- v2 新增
def test_run_dialogue_appends_probe_turns_and_marks_them():
    """探针提问接在脚本台词之后，并且被标记出来，指标那边才能把它们从
    出戏率的分母里排除、单独算记忆正确率。"""
    from src.memory.raw_context import RawContextMemory
    client = FakeClient(["回复1", "回复2", "你妹妹叫 Anna"])
    turns = run_dialogue(
        memory=RawContextMemory(),
        client=client, model="m",
        character_card={"name": "Alice", "description": "d"},
        scripted_user_turns=["你好", "今天怎么样"],
        memory_probes=[{"question": "我妹妹叫什么来着", "expected_answer": "Anna"}],
    )
    assert len(turns) == 6  # 2 轮脚本 + 1 轮探针，各含 user/character
    assert [t["text"] for t in turns if t["speaker"] == "user"][-1] == "我妹妹叫什么来着"
    probe_turns = [t for t in turns if t.get("is_probe")]
    assert len(probe_turns) == 2  # 探针的 user 轮和 character 轮都被标记
    assert all(not t.get("is_probe") for t in turns[:4])


def test_run_dialogue_records_context_length_per_turn():
    """每轮注入了多长的上下文必须记下来，否则无法回答『方案之间的差异是不是
    单纯因为某个方案塞进去的字更多』。"""
    from src.memory.no_memory import NoMemory
    from src.memory.raw_context import RawContextMemory
    card = {"name": "Alice", "description": "d"}

    empty = run_dialogue(memory=NoMemory(), client=FakeClient(["a", "b"]), model="m",
                         character_card=card, scripted_user_turns=["你好", "再见"])
    assert all(t["context_chars"] == 0 for t in empty if t["speaker"] != "user")

    raw = run_dialogue(memory=RawContextMemory(), client=FakeClient(["a", "b"]), model="m",
                       character_card=card, scripted_user_turns=["你好", "再见"])
    char_turns = [t for t in raw if t["speaker"] != "user"]
    assert char_turns[0]["context_chars"] > 0
    assert char_turns[1]["context_chars"] > char_turns[0]["context_chars"]
