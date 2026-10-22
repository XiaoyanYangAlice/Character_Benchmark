"""src/generate/build_dialogues.py 里不依赖网络/datasets 包的那部分逻辑
（事实分组、主角选择启发式）单独测试；实际拉取 PIPPA 数据的部分只能在装了
datasets 包且有 HF 缓存的环境（SURF）里跑，不在这里测。"""
import json
import tempfile
from pathlib import Path

from src.generate.build_dialogues import load_facts_by_story, pick_display_name, pick_protagonist


def test_load_facts_by_story_groups_by_story_and_character():
    records = [
        {"story_id": "pippa_a", "character": "Noelle Holiday", "feature_type": "biography", "span": "x"},
        {"story_id": "pippa_a", "character": "Noelle Holiday", "feature_type": "personality", "span": "y"},
        {"story_id": "pippa_a", "character": "Kris", "feature_type": "biography", "span": "z"},
        {"story_id": "pippa_b", "character": "Alex", "feature_type": "biography", "span": "w"},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "facts.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        by_story = load_facts_by_story(path)

    assert set(by_story.keys()) == {"pippa_a", "pippa_b"}
    assert len(by_story["pippa_a"]["Noelle Holiday"]) == 2
    assert len(by_story["pippa_a"]["Kris"]) == 1
    assert len(by_story["pippa_b"]["Alex"]) == 1


def test_pick_protagonist_returns_character_with_most_facts():
    char_to_facts = {
        "Kris": [{"span": "z"}],
        "Noelle Holiday": [{"span": "x"}, {"span": "y"}],
    }
    name, facts = pick_protagonist(char_to_facts)
    assert name == "Noelle Holiday"
    assert len(facts) == 2


def test_pick_display_name_falls_back_to_bot_name_for_generic_pronoun_label():
    # 第一人称角色卡（"I love to sing..."）抽取出来的 canonical 标签是代词，
    # 直接塞进生成 prompt 语法是破的，应该退回用 PIPPA 自带的 bot_name。
    assert pick_display_name("I", "Yuki") == "Yuki"
    assert pick_display_name("Me", "Yuki") == "Yuki"
    assert pick_display_name("i", "Yuki") == "Yuki"  # 大小写不敏感


def test_pick_display_name_keeps_canonical_label_when_not_generic():
    # 正常情况（角色卡里写了真名）：display_name 就是 canonical 标签本身，
    # 不应该被 bot_name 覆盖掉（哪怕两者不一样）。
    assert pick_display_name("Akane", "SomeUnrelatedBotName") == "Akane"


def test_pick_display_name_keeps_generic_label_when_bot_name_missing():
    # bot_name 本身是空的话没有更好的名字可用，只能保留代词标签。
    assert pick_display_name("I", "") == "I"
