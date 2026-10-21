#!/usr/bin/env python3
"""
按记忆方案分层的 OOC judge 审核抽样。取代 v1 的 ooc_audit_sample.py。

## 与 v1 审核的两个关键区别

1. **不重判。** v1 只存了每条对话的汇总 ooc_rate，没存逐轮判定，所以审核时
   必须用同一套配置重新判 600 轮。judge 温度 0.3，重判和主实验那次不是同一批
   判定，审核结论和主实验之间永远隔着一层。v2 的 run_eval.py 把每一轮的判定
   （含逐次投票和理由）写进 per_turn.csv，这个脚本直接从那里抽样——审核的就是
   主实验真正用过的那些判定。顺带也不需要 GPU 了，登录节点就能跑。

2. **按方案分层。** v1 的 40 条是从全体里抽的，被标出戏的 20 条里 raw 6 条、
   vector 4 条、structured 2 条，样本太小，无法回答最关键的那个问题：judge 的
   误报率是不是跟记忆方案有关？如果 judge 对 raw_context 的回复更容易误报
   （比如 raw 的回复更长、更容易出现被误判的句式），那"structured 的标记率更低"
   就可能完全是 judge 的偏差，而不是生成质量的差别。分层抽样才能检验这件事。

## 用法

    python src/eval/ooc_audit_stratified.py \\
        --per-turn results/per_turn.csv results_seed1/per_turn.csv \\
        --raw-dirs results/raw_outputs results_seed1/raw_outputs \\
        --dialogues data/processed/dialogues.jsonl \\
        --model qwen2.5-7b --n-flagged 20 --n-unflagged 20 \\
        --out-dir results/ooc_audit_v2

每个方案抽 n_flagged 条被标 + n_unflagged 条未标。5 个方案 x 40 条 = 200 条，
人工标注量不小；可以先只审 --schemes raw_context structured_memory 两个最关键的。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUBRIC = """判断准则（出戏 = 符合以下任一条）：
1. 角色自称/暗示自己是 AI、语言模型、聊天机器人，或说"作为一个AI我不能..."这类元话语；
2. 语气、用词风格与角色设定明显不符，且没有情节上的合理转折；
3. 说出与设定的知识范围、时代背景明显冲突的内容；
4. 纯粹的助手式服务用语（"我可以帮你查一下"、"以下是几点建议"之类列点体），脱离角色扮演口吻；
5. 角色"知道"了它不该知道的信息（没有合理的剧情内信息来源）。

不算出戏：
- 角色性格本来就直率/毒舌/温柔，用符合设定的方式表达强烈情绪或拒绝；
- 角色做出符合设定但你不喜欢的选择（喜好不归你判）；
- 使用了设定里明确允许的能力/知识。

⚠ 标注时请忽略"这条回复好不好"、"写得通不通顺"——只判是否脱离了角色设定。
"""


def load_per_turn(paths: list[str], model: str) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("model") != model or r.get("is_probe") in ("True", "true", "1"):
                    continue
                if r.get("ooc") in ("", "None", None):
                    continue  # judge 判定失败的轮次不进审核池
                r["_source"] = Path(p).parent.name
                rows.append(r)
    return rows


def load_reply_text(raw_dirs: list[str], model: str, scheme: str, dialogue_id: str,
                     turn_id: str) -> tuple[str, str]:
    """从 raw_outputs 里取回这一轮的角色回复原文和上一句用户发言。"""
    for d in raw_dirs:
        path = Path(d) / f"{model}__{scheme}__{dialogue_id}.json"
        if not path.exists():
            continue
        turns = json.loads(path.read_text(encoding="utf-8"))
        for i, t in enumerate(turns):
            if str(t.get("turn_id")) == str(turn_id) and t.get("speaker") != "user":
                prev = turns[i - 1]["text"] if i > 0 and turns[i - 1].get("speaker") == "user" else ""
                return t.get("text", ""), prev
    return "", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-turn", nargs="+", required=True)
    ap.add_argument("--raw-dirs", nargs="+", required=True)
    ap.add_argument("--dialogues", default="data/processed/dialogues.jsonl")
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--schemes", nargs="*", default=None,
                     help="只审这些方案；不传就审 per_turn.csv 里出现的所有方案")
    ap.add_argument("--n-flagged", type=int, default=20)
    ap.add_argument("--n-unflagged", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/ooc_audit_v2")
    args = ap.parse_args()

    rows = load_per_turn(args.per_turn, args.model)
    if not rows:
        raise SystemExit(f"per_turn.csv 里没有 model={args.model} 的有效判定")
    dialogues = {d["dialogue_id"]: d for d in
                 (json.loads(l) for l in Path(args.dialogues).read_text(encoding="utf-8").splitlines() if l.strip())}

    by_scheme: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"flagged": [], "unflagged": []})
    for r in rows:
        flagged = r["ooc"] in ("True", "true", "1")
        by_scheme[r["memory_scheme"]]["flagged" if flagged else "unflagged"].append(r)

    schemes = args.schemes or sorted(by_scheme)
    rng = random.Random(args.seed)
    chosen: list[dict] = []
    print(f"{args.model} 共 {len(rows)} 轮有效判定")
    for scheme in schemes:
        buckets = by_scheme.get(scheme)
        if not buckets:
            print(f"  {scheme}: per_turn.csv 里没有数据，跳过")
            continue
        fl, un = buckets["flagged"], buckets["unflagged"]
        rate = len(fl) / max(len(fl) + len(un), 1)
        print(f"  {scheme:<20}标记 {len(fl):>5} / 共 {len(fl) + len(un):>5}（{rate:.1%}）")
        if len(fl) < args.n_flagged:
            print(f"    ⚠ 被标记的只有 {len(fl)} 条，少于要求的 {args.n_flagged}；"
                  f"该方案的精确率区间会很宽，报告里要写清样本量。")
        rng.shuffle(fl)
        rng.shuffle(un)
        chosen += fl[: args.n_flagged] + un[: args.n_unflagged]

    rng.shuffle(chosen)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "judge_ooc", "ooc_votes", "leakage_rule", "scheme",
                    "dialogue_id", "turn_id", "turn_index", "source", "judge_reason"])
        for n, r in enumerate(chosen, 1):
            w.writerow([n, 1 if r["ooc"] in ("True", "true", "1") else 0, r.get("ooc_votes", ""),
                        r.get("leakage", ""), r["memory_scheme"], r["dialogue_id"],
                        r["turn_id"], r.get("turn_index", ""), r.get("_source", ""),
                        r.get("ooc_reason", "")])

    with open(out_dir / "labels_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "human_ooc", "note"])
        for n in range(1, len(chosen) + 1):
            w.writerow([n, "", ""])

    lines = ["# OOC judge 分层审核清单（盲评）", "",
             "逐条判断下面每个「角色回复」是否出戏，在 labels_template.csv 里对应 item_id 一行的 "
             "human_ooc 列填 1（出戏）或 0（没出戏），拿不准填 ?。**请先全部判完再看 key.csv。**",
             "", "清单里混合了不同记忆方案、被标记和未被标记的样本，顺序已打乱；",
             "标注时无法（也不应该）知道某一条来自哪个方案、judge 怎么判的。", "",
             RUBRIC,
             "注：judge 只看到「角色设定 + 角色回复」。「用户上一句」仅供你理解语境，judge 没看到。",
             "", "---", ""]
    n_missing = 0
    for n, r in enumerate(chosen, 1):
        text, prev = load_reply_text(args.raw_dirs, args.model, r["memory_scheme"],
                                      r["dialogue_id"], r["turn_id"])
        if not text:
            n_missing += 1
        card = dialogues.get(r["dialogue_id"], {}).get("character_card", {})
        name = card.get("display_name") or card.get("name", "")
        desc = (card.get("description") or "").strip().replace("\n", " ")
        lines += [f"## item {n}", f"**角色**：{name}",
                  f"**角色设定**：{desc[:700]}{'…' if len(desc) > 700 else ''}",
                  f"**用户上一句（仅参考）**：{prev or '（无）'}", "",
                  f"**角色回复**：{text or '（raw_outputs 里没找到这一轮，检查 --raw-dirs）'}",
                  "", "---", ""]
    (out_dir / "sheet.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {"model": args.model, "schemes": schemes, "n_items": len(chosen),
            "n_flagged_per_scheme": args.n_flagged, "n_unflagged_per_scheme": args.n_unflagged,
            "seed": args.seed, "n_missing_text": n_missing,
            "flag_rate_by_scheme": {s: len(by_scheme[s]["flagged"]) /
                                    max(len(by_scheme[s]["flagged"]) + len(by_scheme[s]["unflagged"]), 1)
                                    for s in schemes if s in by_scheme}}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n共抽 {len(chosen)} 条，写到 {out_dir}/：sheet.md（给你读）、"
          f"labels_template.csv（你填）、key.csv（答案，判完再看）、meta.json")
    if n_missing:
        print(f"⚠ {n_missing} 条在 raw_outputs 里没找到原文，检查 --raw-dirs 是否和 --per-turn 对应")
    print("标完之后：python src/eval/ooc_audit_score.py --audit-dir " + str(out_dir) +
          " --labels <你的labels.csv>  （会按方案分别给出精确率）")


if __name__ == "__main__":
    main()
