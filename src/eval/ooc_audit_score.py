"""给 OOC judge 的人工审核打分。读 ooc_audit_sample.py 产出的 key.csv / meta.json
和你填好的 labels.csv（item_id, human_ooc：1=出戏 0=没出戏），报告：

- 精确率：被 judge 标为出戏的里面，人工也认为出戏的比例（Wilson 95% CI）；
- 漏判率：judge 没标的里面，人工认为其实出戏的比例（Wilson 95% CI）；
- 估计召回率：用抽样时 judge 的总体标记率把上面两个率合起来估计，样本很小，
  区间会很宽，只作量级参考。

只用标准库。用法：
    python src/eval/ooc_audit_score.py [--dir results/ooc_audit] [--labels results/ooc_audit/labels.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ooc_audit")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    labels_path = Path(args.labels) if args.labels else d / "labels.csv"

    key = {int(r["item_id"]): int(r["judge_ooc"]) for r in csv.DictReader(open(d / "key.csv", encoding="utf-8"))}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    human = {}
    for r in csv.DictReader(open(labels_path, encoding="utf-8")):
        v = (r.get("human_ooc") or "").strip()
        if v in ("0", "1"):
            human[int(r["item_id"])] = int(v)
    missing = sorted(set(key) - set(human))
    if missing:
        print(f"还有 {len(missing)} 条没填 human_ooc（item {missing[:10]}{'…' if len(missing) > 10 else ''}），"
              f"只统计已填的。")

    fl = [i for i in human if key[i] == 1]
    un = [i for i in human if key[i] == 0]
    tp = sum(human[i] for i in fl)
    fn = sum(human[i] for i in un)

    prec = tp / len(fl) if fl else float("nan")
    miss = fn / len(un) if un else float("nan")
    pl, ph = wilson(tp, len(fl))
    ml, mh = wilson(fn, len(un))
    print(f"被 judge 标为出戏：{len(fl)} 条，人工同意 {tp} 条 → 精确率 {prec:.0%}  (95% CI {pl:.0%}–{ph:.0%})")
    print(f"judge 没标出戏：  {len(un)} 条，人工认为其实出戏 {fn} 条 → 漏判率 {miss:.0%}  (95% CI {ml:.0%}–{mh:.0%})")

    f_rate = meta["n_flagged_total"] / meta["n_judged"] if meta["n_judged"] else float("nan")
    if fl and un and f_rate == f_rate:
        true_pos_rate = f_rate * prec
        true_total = true_pos_rate + (1 - f_rate) * miss
        recall = true_pos_rate / true_total if true_total > 0 else float("nan")
        print(f"\nJudge 在随机抽样中的总体标记率：{f_rate:.1%}（{meta['n_flagged_total']}/{meta['n_judged']}）")
        print(f"据此估计：真实出戏率约 {true_total:.1%}，judge 召回率约 {recall:.0%}"
              f"（样本小，仅作量级参考，区间很宽）")
    print("\n读法：精确率低 → judge 过度标记，实验里的 ooc_rate 绝对值被高估；"
          "漏判率高 → 低估。各方案/模型之间的相对差异只在 judge 的错误跟记忆方案无关时才可信。")


if __name__ == "__main__":
    main()
