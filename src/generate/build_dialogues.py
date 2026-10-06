#!/usr/bin/env python3
"""
build_dialogues.py

把 PIPPA 原始对话数据（真实人类和角色的对话轮次）跟已经跑出来的角色事实
（character_features_en.jsonl）关联起来，产出 dialogue_runner.py /
run_eval.py 需要的 data/processed/dialogues.jsonl。

不需要重新下载数据——走的是 fetch_pippa.py 已经建好的本地 HuggingFace 缓存，
只是这次把 `conversation` 字段里人类说的那些轮次（is_human=True）也取出来，
之前 fetch_pippa.py 只用了 bot_description，把这部分扔掉了。

⚠ 一个需要记录的设计决定：一个 PIPPA 故事的 bot_description 里经常会提到好几
个角色（比如"Noelle Holiday" + 她的朋友 Berdly/Kris/Susie），但这个故事真正
在角色扮演的"主角"只有一个（对应 PIPPA 的 bot_id 本身）。这里用一个启发式来
判断谁是主角：**事实条数最多的那个角色标签** —— 描述篇幅越大，抽取出来的事实
自然越多，这个假设在抽样检查里基本成立，但不是精确方法，极端情况下可能选错
（比如篇幅很长地描述配角、主角着墨反而少的故事）。`--min-facts` 参数会把
事实条数太少（可能是选错了主角，或者这个故事本来就不适合拿来做长对话生成）的
故事跳过，不是完美过滤，跑完建议抽样核对几条 character_card["name"] 是不是
真的对应这个故事该演的那个角色。

⚠ character_card["name"] 必须原样使用 character_features_en.jsonl 里的
canonical 标签（哪怕是 "I" 这种不太像名字的标签），不能换成 PIPPA 的
bot_name 字段——structured_memory.py 是按这个标签去 character_facts 字典里
查事实的，标签对不上就检索不到任何东西，参见 dialogue_runner.py 文件头的
说明。

relationship_trajectory 字段现在统一留空（[]）——关系状态弱标签打分脚本还
没做，是 docs/DATASET.md 里记录的 TODO，做出来之后再回填。

用法（在 SURF 登录节点，跟 fetch_pippa.py 同一个目录下跑）：
    python build_dialogues.py \
        --facts ../../data/processed/character_features_en.jsonl \
        --output ../../data/processed/dialogues.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

# datasets 是个不小的依赖，延迟到 main() 里再导入，这样 load_facts_by_story()/
# pick_protagonist() 这两个不需要它的纯逻辑函数可以在没装 datasets 的环境
# （比如本地跑单元测试）里单独测试，不用为了 import 这一个模块就装一整套。


def load_facts_by_story(facts_path: Path) -> dict[str, dict[str, list[dict]]]:
    """story_id -> {character_label: [fact_record, ...]}"""
    by_story: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in facts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        by_story[rec["story_id"]][rec["character"]].append(rec)
    return by_story


def pick_protagonist(char_to_facts: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    """启发式：事实条数最多的角色当作这个故事在扮演的主角，见文件头说明。"""
    return max(char_to_facts.items(), key=lambda kv: len(kv[1]))


# 实测抽查发现：PIPPA 不少角色卡整段用第一人称写（"I love to sing..."），完全
# 没写出真名，ontology 抽取器在这种文本里找不到专有名词可用，只能把代词本身
# 当 canonical 标签写进 character_features_en.jsonl（8 条抽查里出现了 "I"/
# "Me" 两种，另外还有 "Harasser"/"Automaton" 这类没给名字、抽取器只能用最
# 突出的描述词顶上的情况——这两种不在下面这个代词集合里，因为它们本身作为
# "名字"塞进句子里读着还算自然，不需要额外处理）。
_GENERIC_PRONOUN_LABELS = {"i", "me", "you", "we", "us", "he", "she", "they", "him", "her"}


def pick_display_name(canonical_name: str, bot_name: str) -> str:
    """给生成 prompt 用的"读起来自然的名字"，跟 canonical_name（存进
    character_card["name"]、用于匹配 fact_labels 的那个标签）分开算。

    canonical_name 是代词（比如 "I"/"Me"）时，直接塞进
    "You are role-playing as I" 这种句子里语法是破的，退回用 PIPPA 自带的
    bot_name（人写的、更像名字）；不是代词、或者 bot_name 本身是空的，就还是
    用 canonical_name 本身——大多数情况下都是这样（比如 "Akane"/"Madara"）。

    ⚠ 这个函数只负责算 display_name，不影响 character_card["name"] 本身：
    后者是 structured_memory.py 拿去匹配 character_features_en.jsonl 里的
    canonical 标签的，两者不能混用，见 dialogue_runner.py 文件头说明。
    """
    if canonical_name.strip().lower() in _GENERIC_PRONOUN_LABELS and bot_name.strip():
        return bot_name.strip()
    return canonical_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True, type=Path,
                     help="src/ontology_extraction/extract_character_features.py 的输出")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-user-turns", type=int, default=6,
                     help="每段对话最多取多少轮用户真实发言（轮数越多，生成+评测成本越高）")
    ap.add_argument("--min-facts", type=int, default=2,
                     help="主角事实条数少于这个数就跳过（大概率是主角选错了，或者素材太薄）")
    args = ap.parse_args()

    from datasets import load_dataset  # 见文件顶部说明：延迟导入

    by_story = load_facts_by_story(args.facts)
    wanted_bot_ids = {story_id.removeprefix("pippa_") for story_id in by_story}
    print(f"从 {args.facts} 读到 {len(by_story)} 个故事的事实。")

    print("加载 PIPPA 原始对话数据做关联（走本地缓存，fetch_pippa.py 已经下载过，不会重新联网下载）...")
    ds = load_dataset("PygmalionAI/pippa", data_files="pippa.jsonl", split="train")

    n_written = 0
    n_skip_too_few_facts = 0
    n_skip_no_human_turns = 0
    seen_bots = set()

    with open(args.output, "w", encoding="utf-8") as out:
        for row in ds:
            bot_id = row.get("bot_id")
            if bot_id not in wanted_bot_ids or bot_id in seen_bots:
                continue
            seen_bots.add(bot_id)

            story_id = f"pippa_{bot_id}"
            char_to_facts = by_story[story_id]
            protagonist_name, protagonist_facts = pick_protagonist(char_to_facts)
            if len(protagonist_facts) < args.min_facts:
                n_skip_too_few_facts += 1
                continue

            conv = row.get("conversation", []) or []
            human_turns = [m["message"] for m in conv if m.get("is_human") and m.get("message", "").strip()]
            scripted_user_turns = human_turns[: args.max_user_turns]
            if not scripted_user_turns:
                n_skip_no_human_turns += 1
                continue

            record = {
                "dialogue_id": story_id,
                "character_card": {
                    "name": protagonist_name,  # 必须跟 fact_labels 里的 character 标签一致，见文件头
                    "display_name": pick_display_name(protagonist_name, row.get("bot_name") or ""),
                    "description": (row.get("bot_description") or "").strip(),
                    "greeting": (row.get("bot_greeting") or "").strip(),
                },
                "scripted_user_turns": scripted_user_turns,
                "fact_labels": protagonist_facts,
                "relationship_trajectory": [],  # TODO：弱标签打分脚本还没做，见 docs/DATASET.md
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    n_unmatched = len(wanted_bot_ids) - len(seen_bots)
    print(f"\n写出 {n_written} 段对话到 {args.output}")
    print(f"跳过：主角事实条数 < {args.min_facts} 的 {n_skip_too_few_facts} 个，"
          f"没有人类发言轮次的 {n_skip_no_human_turns} 个，"
          f"在 pippa.jsonl 里没匹配到 bot_id 的 {n_unmatched} 个"
          f"（正常情况下应该是 0，不是就说明数据集版本变了）")
    print("下一步：抽样核对几条 character_card.name 是不是真的对应这个故事该演的角色，"
          "再跑 src/eval/run_eval.py。")


if __name__ == "__main__":
    main()
