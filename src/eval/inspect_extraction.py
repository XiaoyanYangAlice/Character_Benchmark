#!/usr/bin/env python3
"""inspect_extraction.py —— 把 structured_memory 的召回失败拆成两种原因。

诊断脚本已经告诉我们：探针答案有 85–90% 的时候不在注入的上下文里。但"不在"有
两个完全不同的原因，要用完全不同的办法修：

  A. **从来没抽出来**  → 抽取器的问题。改 prompt，或者微调一个抽取器。
  B. **抽出来了但被挤掉** → 检索预算的问题。改成按相关度取，不是按时间近远取。

现有数据就能分开这两种，不需要 GPU，也不需要重跑。关键观察：
`memory_context` 是每一个角色轮次都存了的，而 structured_memory 每次注入的是
**最近 max_events 条**事件。一条事件刚被抽出来的那一刻必定是最新的，所以它一定
会出现在紧接着的那个窗口里。**把一条对话所有轮次的 memory_context 里的事件行取
并集，就重建出了这条对话抽取过的（几乎）全部事件。**

（唯一漏掉的是最后一个来回抽出的事件——它后面没有轮次了，所以从没进过任何窗口。
影响一条对话最多一两条，不改变结论量级。）

有了并集，就能对每道探针问：答案在"曾经抽到过的全部事件"里吗？
  - 不在 → 原因 A
  - 在，但不在探针那一刻的上下文里 → 原因 B

用法：
    python src/eval/inspect_extraction.py \
        --raw-dirs results_diag/raw_outputs \
        --dialogues data/processed/dialogues.jsonl \
        --scheme structured_memory \
        --samples 12 \
        --out docs/EXTRACTION_AUDIT.txt
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

# structured_memory.retrieve() 写出来的事件行长这样：  - (turn 17) 内容
EVENT_LINE = re.compile(r"^-\s*\(turn\s+(\d+)\)\s*(.+)$", re.M)
# 角色卡事实那一段：[Known facts about X] a; b; c
FACTS_LINE = re.compile(r"^\[Known facts about [^\]]*\]\s*(.*)$", re.M)


def answer_in(text: str, answer: str) -> bool:
    """与 make_probes.validate_probe / probe_diagnosis 同一套朴素匹配。
    宁可漏判也不要误判，所以这个函数给出的是下界。"""
    if not text or not answer:
        return False
    t = " ".join(text.lower().split())
    a = " ".join(answer.lower().split())
    if a in t:
        return True
    words = [w for w in a.split() if len(w) > 2]
    return bool(words) and all(w in t for w in words)


def parse_events(ctx: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2).strip()) for m in EVENT_LINE.finditer(ctx or "")]


def parse_facts(ctx: str) -> str:
    m = FACTS_LINE.search(ctx or "")
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dirs", nargs="+", required=True)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--scheme", default="structured_memory")
    ap.add_argument("--samples", type=int, default=12,
                    help="人工眼看的样本数：随机抽这么多道答错的探针，逐条打印"
                         "「题目 / 标准答案 / 用户原话 / 当时注入了什么 / 曾经抽到过什么」")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    probes_by_did: dict[str, list[dict]] = {}
    turns_by_did: dict[str, list[str]] = {}
    for line in Path(args.dialogues).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            probes_by_did[d["dialogue_id"]] = d.get("memory_probes") or []
            turns_by_did[d["dialogue_id"]] = d.get("scripted_user_turns") or []

    stats = defaultdict(lambda: defaultdict(int))
    cases: list[dict] = []
    n_files = 0

    for rd in args.raw_dirs:
        for f in sorted(Path(rd).glob("*.json")):
            # dialogue_id 自身可能含 "__"（PIPPA 的 ID 是 base64 风格），
            # 所以限制成切两次，否则那条对话会被整条静默丢掉。
            parts = f.stem.split("__", 2)
            if len(parts) != 3:
                continue
            model, scheme, did = parts
            if scheme != args.scheme:
                continue
            plist = probes_by_did.get(did) or []
            if not plist:
                continue
            n_files += 1
            turns = json.loads(f.read_text(encoding="utf-8"))

            # 1) 重建"曾经抽到过的全部事件"
            # ⚠ 必须按 (轮次, 内容) 去重，不能只按轮次：同一个来回可以抽出多条
            # 事件（make_llm_event_extractor 的 max_events_per_turn 默认是 3）。
            # 第一版用 turn_id 当字典键，每轮只留了一条，重建出来的事件数只有
            # 真实值的六成，结果是把"被挤掉"误判成了"从没抽出来"。
            all_events: list[tuple[int, str]] = []
            seen_ev: set[tuple[int, str]] = set()
            for t in turns:
                for tid, content in parse_events(t.get("memory_context") or ""):
                    key = (tid, content)
                    if key not in seen_ev:
                        seen_ev.add(key)
                        all_events.append(key)
            all_events.sort()
            ever_text = " || ".join(c for _, c in all_events)
            # 这些事件里有多少是在记用户说的话？探针考的全是用户发言，所以这个
            # 比例低本身就是一个发现。
            n_user_ev = sum(1 for _, c in all_events
                            if re.search(r"\buser\b|\bthe user's\b", c, re.I))
            card_facts = ""
            for t in turns:
                card_facts = parse_facts(t.get("memory_context") or "") or card_facts

            # 2) 逐道探针判定
            k = 0
            for t in turns:
                if t.get("speaker") == "user" or not t.get("is_probe"):
                    continue
                if k >= len(plist):
                    break
                probe = plist[k]; k += 1
                ans = str(probe.get("expected_answer", ""))
                ctx = t.get("memory_context")
                if ctx is None:
                    stats[(model, scheme)]["no_ctx"] += 1
                    continue
                in_now = answer_in(ctx, ans)
                in_ever = answer_in(ever_text, ans)
                in_card = answer_in(card_facts, ans)
                s = stats[(model, scheme)]
                s["n"] += 1
                s["n_events_total"] += len(all_events)
                s["n_user_events"] += n_user_ev
                if in_now:
                    s["in_context"] += 1
                elif in_ever:
                    s["extracted_but_evicted"] += 1      # 原因 B
                else:
                    s["never_extracted"] += 1            # 原因 A
                    if in_card:
                        s["never_extracted_but_in_card"] += 1
                if not in_now:
                    src = probe.get("source_turn_index")
                    user_turns = turns_by_did.get(did) or []
                    src_text = user_turns[int(src) - 1] if (
                        isinstance(src, int) and 1 <= src <= len(user_turns)) else ""
                    cases.append({
                        "model": model, "did": did,
                        "question": probe.get("question", ""), "answer": ans,
                        "src_turn": src, "src_text": src_text,
                        "ctx": ctx, "ever": all_events,
                        "reason": "extracted_but_evicted" if in_ever else "never_extracted",
                    })

    L = [f"读到 {n_files} 个 {args.scheme} 的 raw_outputs 文件。"]
    if not stats:
        L.append("没有可分析的数据。检查 --scheme 拼写，以及 raw_outputs 是不是"
                 "打了 memory_context 补丁之后跑的。")
        print("\n".join(L)); return

    L.append("")
    L.append("失败归因：答案不在上下文里，是从没抽出来，还是抽出来了被挤掉？")
    L.append(f"  {'model':<14}{'探针':>6}{'在上下文':>10}{'抽过但被挤掉':>14}"
             f"{'从没抽出来':>12}{'卡里有':>8}{'平均事件数':>12}{'其中关于用户':>14}")
    for (model, scheme), s in sorted(stats.items()):
        n = s["n"] or 1
        ev = s["n_events_total"] or 1
        L.append(f"  {model:<14}{s['n']:6d}{s['in_context']/n:10.1%}"
                 f"{s['extracted_but_evicted']/n:14.1%}{s['never_extracted']/n:12.1%}"
                 f"{s['never_extracted_but_in_card']:8d}"
                 f"{s['n_events_total']/n:12.1f}{s['n_user_events']/ev:14.1%}")

    L += [
        "",
        "怎么读这张表 —— 它直接决定下一步做什么：",
        "- 「抽过但被挤掉」占大头 → 瓶颈是检索预算。先改成按相关度取，不要急着微调。",
        "- 「从没抽出来」占大头   → 瓶颈是抽取器。先改抽取 prompt；改不动再考虑微调。",
        "- 两者都不小             → 两个都要修，但先修便宜的那个（检索）。",
        "- 「平均事件数」远大于 max_events（配置里是 8）本身只说明有挤占，",
        "  不说明挤占是主因——要看第二列。",
        "- 重建出来的事件数应当与 summary.csv 的 n_events_extracted 大致吻合；",
        "  明显偏低说明重建漏了事件，「从没抽出来」会被高估，结论要打折扣。",
        "- 「其中关于用户」低 → 抽取器在记角色做了什么，而探针问的全是用户说了",
        "  什么。这是方向性错误，改 prompt 比改检索更关键。",
        "",
        "注意：匹配是朴素子串/实词匹配，会漏判语义改写，所以「从没抽出来」这一列",
        "是**上界**（真实值可能更低，因为有些是抽到了但换了说法）。下面的人工样本",
        "就是用来判断这个高估有多严重的——必须看。",
    ]

    rng = random.Random(args.seed)
    rng.shuffle(cases)
    L.append("")
    L.append("=" * 78)
    L.append(f"人工核对样本（随机 {args.samples} 道答案不在上下文里的探针）")
    L.append("重点看：标准答案在「用户原话」里是具体的，但在「曾经抽到过的事件」里")
    L.append("变成了什么？如果变成了泛化的概括（'用户提到了家人'），那就是抽取 prompt")
    L.append("的问题，微调之前先改 prompt。")
    L.append("=" * 78)
    for c in cases[: args.samples]:
        L.append("")
        L.append(f"[{c['reason']}] {c['model']} / {c['did']}")
        L.append(f"  探针问题    : {c['question']}")
        L.append(f"  标准答案    : {c['answer']}")
        L.append(f"  出自第 {c['src_turn']} 轮用户原话 : {c['src_text'][:300]}")
        ev = c["ever"]
        L.append(f"  曾经抽到过的事件（共 {len(ev)} 条，全列）:")
        for tid, content in ev:
            L.append(f"      (turn {tid}) {content[:160]}")
        L.append(f"  探针那一刻注入的上下文:")
        for ln in (c["ctx"] or "").splitlines():
            if ln.strip():
                L.append(f"      {ln[:160]}")

    body = "\n".join(L)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\n已写到 {args.out}")


if __name__ == "__main__":
    main()
