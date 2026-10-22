#!/usr/bin/env python3
"""probe_diagnosis.py —— 把探针的失败拆成两种，这两种要用完全不同的办法修。

探针答错有两种可能，汇总指标分不开：

  A. 答案根本不在注入的上下文里  → 写入或检索失败。
     structured 的话，要么抽取器没把这条信息写进事件，要么写进去了但被
     max_events 的时间近远预算挤出了检索结果。
  B. 答案在上下文里，角色还是答错了  → 生成失败。
     记忆是对的，模型没用上。

这个脚本读 raw_outputs/ 里每个探针轮次实际注入的上下文（需要先打了
dialogue_runner.py 保存 memory_context 的补丁），对照 dialogues.jsonl 里
那道探针的标准答案，输出一张 2x2：答案在不在 × 答没答对。

判定"答案在上下文里"用的是和 make_probes.py::validate_probe 同一套朴素匹配
（整体子串，或多词答案的每个实词都出现），宁可漏判也不要误判。

用法：
    python src/eval/probe_diagnosis.py \\
        --raw-dirs results_seed1/raw_outputs results_seed2/raw_outputs \\
        --dialogues data/processed/dialogues.jsonl \\
        --probe-csv results_seed1/probe_results.csv results_seed2/probe_results.csv \\
        --out docs/PROBE_DIAGNOSIS.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BOOL = {"true": True, "false": False, "1": True, "0": False}


def as_bool(v):
    if v is None or v == "":
        return None
    s = str(v).strip().lower()
    if s in _BOOL:
        return _BOOL[s]
    try:
        return float(s) >= 0.5
    except ValueError:
        return None


def answer_in(text: str, answer: str) -> bool:
    """答案是否出现在这段文本里。与 make_probes.validate_probe 同一套规则。"""
    if not text or not answer:
        return False
    t = " ".join(text.lower().split())
    a = " ".join(answer.lower().split())
    if a in t:
        return True
    words = [w for w in a.split() if len(w) > 2]
    return bool(words) and all(w in t for w in words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dirs", nargs="+", required=True)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--probe-csv", nargs="+", default=[])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # dialogue_id -> [探针]
    probes: dict[str, list[dict]] = {}
    for line in Path(args.dialogues).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            probes[d["dialogue_id"]] = d.get("memory_probes") or []

    # (model, scheme, dialogue_id, turn_id) -> 是否答对
    correct: dict[tuple, bool] = {}
    for p in args.probe_csv:
        pth = Path(p)
        if not pth.exists():
            continue
        with pth.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                c = as_bool(r.get("correct"))
                if c is not None:
                    correct[(r["model"], r["memory_scheme"], r["dialogue_id"],
                             str(r.get("turn_id")))] = c

    cells: dict = defaultdict(lambda: defaultdict(int))
    n_files = n_noctx = 0
    for rd in args.raw_dirs:
        for f in sorted(Path(rd).glob("*.json")):
            # 文件名格式：<model>__<scheme>__<dialogue_id>.json
            parts = f.stem.split("__")
            if len(parts) != 3:
                continue
            model, scheme, did = parts
            n_files += 1
            turns = json.loads(f.read_text(encoding="utf-8"))
            plist = probes.get(did, [])
            if not plist:
                continue
            # 探针轮次按出现顺序和 memory_probes 一一对应
            k = 0
            for t in turns:
                if t.get("speaker") == "user" or not t.get("is_probe"):
                    continue
                if k >= len(plist):
                    break
                ans = str(plist[k].get("expected_answer", ""))
                ctx = t.get("memory_context")
                k += 1
                if ctx is None:
                    n_noctx += 1
                    continue
                here = answer_in(ctx, ans)
                got = correct.get((model, scheme, did, str(t.get("turn_id"))))
                key = (model, scheme)
                cells[key]["n"] += 1
                cells[key]["in_ctx" if here else "not_in_ctx"] += 1
                if got is None:
                    cells[key]["no_label"] += 1
                else:
                    cells[key][("in" if here else "out") + ("_right" if got else "_wrong")] += 1

    L = [f"读到 {n_files} 个 raw_outputs 文件。"]
    if n_noctx:
        L.append(f"⚠ {n_noctx} 个探针轮次没有 memory_context 字段——这些是打补丁之前")
        L.append("  跑出来的结果，无法诊断。重跑时请确认用的是新版 dialogue_runner.py。")
    if not any(cells.values()):
        L.append("没有可诊断的探针轮次。检查 --dialogues 是不是带探针的那份，")
        L.append("以及 raw_outputs 是不是打补丁之后跑的。")
        print("\n".join(L))
        return

    L.append("")
    L.append("答案是否出现在注入的上下文里 × 是否答对")
    L.append(f"  {'model':<13}{'scheme':<19}{'探针数':>7}{'答案在上下文':>13}"
             f"{'在·答对':>9}{'在·答错':>9}{'不在·答对':>11}{'不在·答错':>11}")
    for (model, scheme), c in sorted(cells.items()):
        n = c["n"] or 1
        L.append(f"  {model:<13}{scheme:<19}{c['n']:7d}{c['in_ctx']/n:13.1%}"
                 f"{c['in_right']:9d}{c['in_wrong']:9d}{c['out_right']:11d}{c['out_wrong']:11d}")
    L += [
        "",
        "读法：",
        "- 「答案在上下文」这一列低 → 写入或检索失败。对 structured 来说，要么抽取器",
        "  没写进去，要么写进去了但被 max_events 的时间近远预算挤出了检索结果。",
        "- 「在·答错」多 → 生成失败：记忆给对了，模型没用上。这跟记忆方案无关，",
        "  是提示词或模型能力的问题，换检索策略解决不了。",
        "- 「不在·答对」是模型猜对的，或答案恰好能从角色卡推出来。这一格大的话，",
        "  探针本身的难度要重新检查。",
        "- 匹配是朴素子串/实词匹配，会漏判语义等价的改写（'我妹妹' vs 'his sister'），",
        "  所以「答案在上下文」这一列是**下界**。",
    ]
    body = "\n".join(L)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\n已写到 {args.out}")


if __name__ == "__main__":
    main()
