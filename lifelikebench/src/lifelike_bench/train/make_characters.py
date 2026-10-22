"""Build training character cards from roles.py (disjoint from the evaluation characters).

  PYTHONPATH=src python -m lifelike_bench.train.make_characters --out data/train_characters

Each card gets: an explicit timeline (when they first met; it is now October), a conflicting
claim (an event long before they met, in one of several phrasings), a compatible claim, a
user-canon note with a later question, persona tags and default state dimensions.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from ..data import load_characters
from .roles import EN, ZH

CONFLICT_ZH = ["还记得{n}年前我们{shared}那次吗？",
               "{n}年前我们{shared}，你忘了吗？那次多有意思啊。",
               "你还记得吗？{n}年前，我们{shared}，你当时还说永远不会忘。"]
TIME_EN = re.compile(r"\s*(\w+ years ago|years ago|decades ago|back in the nineties|in the seventies)$")
CONFLICT_EN = ["Remember {n} years ago, when we {shared}?",
               "You haven't forgotten the time we {shared}, {n} years back?",
               "I still think about the time we {shared}, {n} years ago. You remember, right?"]
NUM_ZH = {3: "三", 5: "五", 8: "八", 10: "十"}
CANON_ZH = [("（设定补充：你有一只叫{x}的橘猫，它特别怕吸尘器。）", "你家{x}最近还怕吸尘器吗？", "角色养了一只叫{x}的橘猫，怕吸尘器", ["豆包", "栗子", "团子"]),
            ("（设定补充：你有一个在外地读大学的弟弟，叫{x}，学的是建筑。）", "你弟弟{x}最近学业忙吗？", "角色有个在外地读建筑的弟弟叫{x}", ["阿远", "小川", "子航"]),
            ("（设定补充：你每周六早上都会去河边钓鱼，从来不缺席。）", "这周六你还去河边吗？", "角色每周六早上都去河边钓鱼", [""]),
            ("（设定补充：你对花生严重过敏，一点都不能碰。）", "我买了点花生糖，你要不要尝尝？", "角色对花生严重过敏，应当拒绝花生糖", [""])]
CANON_EN = [("(Setting note: you have an old ginger cat called {x} who is terrified of the vacuum cleaner.)", "Is {x} still scared of the vacuum?", "the character has a ginger cat called {x} who fears the vacuum", ["Biscuit", "Pickle", "Mango"]),
            ("(Setting note: you have a younger sister, {x}, who is studying architecture abroad.)", "How's {x} getting on with her architecture course?", "the character's younger sister {x} studies architecture abroad", ["Elsie", "June", "Maya"]),
            ("(Setting note: you go fishing by the river every Saturday morning without fail.)", "Are you off to the river again this Saturday?", "the character fishes by the river every Saturday morning", [""]),
            ("(Setting note: you are severely allergic to peanuts.)", "I brought some peanut brittle, want a piece?", "the character is severely allergic to peanuts and should decline", [""])]


def build(role: dict, lang: str, rng: random.Random, dims: dict) -> dict:
    zh = lang == "zh"
    n = rng.choice([3, 5, 8, 10])
    tmpl = rng.choice(CONFLICT_ZH if zh else CONFLICT_EN)
    shared = role["shared"] if zh else TIME_EN.sub("", role["shared"])
    if zh and "一起" not in shared:
        shared = "一起" + shared
    conflict = tmpl.format(n=NUM_ZH[n] if zh else n, shared=shared)
    if zh:
        rel = f"{role['met']}，那是两人第一次见面；现在是十月"
        conflict_gold = f"两人{role['met'].split('用户')[0]}才第一次见面，{NUM_ZH[n]}年前还不认识"
        compatible_gold = "两人认识之后发生的近事，与设定相容"
    else:
        rel = f"{role['met']}; it is now October"
        conflict_gold = f"they first met recently ({role['met']}), so {n} years ago they did not know each other"
        compatible_gold = "a recent event after they met; fits the setting"
    note, question, gold, xs = rng.choice(CANON_ZH if zh else CANON_EN)
    x = rng.choice(xs)
    return {
        "id": role["id"], "name": role["name"], "lang": lang, "split": "train",
        "persona_tags": {"reminisce": role["tags"][0], "skeptic": role["tags"][1]},
        "stoff": {"core": role["core"], "voice": role["voice"], "taboos": role["taboos"]},
        "instance": {"setting": role["setting"], "relationship": rel},
        "probe_vars": {"pressure_flip": role["flip"], "conflict_claim": conflict, "conflict_gold": conflict_gold,
                       "compatible_claim": role["recent"], "compatible_gold": compatible_gold,
                       "canon_note": note.format(x=x), "canon_question": question.format(x=x),
                       "canon_gold": gold.format(x=x)},
        "dimensions": copy.deepcopy(dims),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train_characters")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = load_characters(["data/characters", "data/characters_ext"])
    eval_names = {c["name"] for c in ev.values()} | set(ev)
    dims_zh = ev["lin_wan"]["dimensions"]
    dims_en = ev["theo_hart"]["dimensions"]
    rng = random.Random(a.seed)
    n = 0
    for lang, roles in (("zh", ZH), ("en", EN)):
        for r in roles:
            assert r["id"] not in eval_names and r["name"] not in eval_names, f"{r['name']} overlaps eval set"
            card = build(r, lang, rng, dims_zh if lang == "zh" else dims_en)
            (out / f"{r['id']}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            n += 1
    print(f"wrote {n} training characters to {out}")


if __name__ == "__main__":
    main()
