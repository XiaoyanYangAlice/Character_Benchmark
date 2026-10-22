"""给 OOC judge 的人工审核打分。读审核目录里的 key.csv / meta.json 和你填好的
labels.csv（item_id, human_ooc：1=出戏 0=没出戏，? 或留空=拿不准），报告：

- 精确率：被 judge 标为出戏的里面，人工也认为出戏的比例（Wilson 95% CI）；
- 漏判率：judge 没标的里面，人工认为其实出戏的比例（Wilson 95% CI）；
- 估计召回率：用抽样时 judge 的总体标记率把上面两个率合起来估计，样本很小，
  区间会很宽，只作量级参考；
- **按记忆方案分别给出精确率**（v2 新增，需要 key.csv 里有 scheme 列）。这是
  整个审核最关键的一张表：如果 judge 对不同方案的精确率明显不同，那么方案之间
  "出戏标记率"的差异就不能解释成生成质量的差异——judge 的误报本身就跟方案相关。
- **与规则检测器的交叉表**（v2 新增）：规则命中、judge 标记、人工标注三者的
  一致情况。规则检测高精确低召回，可以用来独立佐证人工标注。

同时支持 v1（ooc_audit_sample.py）和 v2（ooc_audit_stratified.py）产出的目录。

只用标准库。用法：
    python src/eval/ooc_audit_score.py --audit-dir results/ooc_audit_v2 \\
        --labels results/ooc_audit_v2/labels_user.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
    ap.add_argument("--audit-dir", "--dir", dest="audit_dir", default="results/ooc_audit_v2")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()
    d = Path(args.audit_dir)
    labels_path = Path(args.labels) if args.labels else d / "labels.csv"

    key_rows = list(csv.DictReader(open(d / "key.csv", encoding="utf-8")))
    key = {int(r["item_id"]): int(r["judge_ooc"]) for r in key_rows}
    scheme_of = {int(r["item_id"]): r.get("scheme", "") for r in key_rows}
    leak_of = {int(r["item_id"]): int(r.get("leakage_rule") or r.get("leakage") or 0)
               for r in key_rows}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    human, unsure = {}, []
    for r in csv.DictReader(open(labels_path, encoding="utf-8")):
        v = (r.get("human_ooc") or "").strip()
        if v in ("0", "1"):
            human[int(r["item_id"])] = int(v)
        elif v:
            unsure.append(int(r["item_id"]))
    missing = sorted(set(key) - set(human) - set(unsure))
    if missing:
        print(f"还有 {len(missing)} 条没填 human_ooc（item {missing[:10]}{'…' if len(missing) > 10 else ''}），"
              f"只统计已填的。")
    if unsure:
        print(f"{len(unsure)} 条标为「拿不准」，不计入分母（item {sorted(unsure)[:10]}）。"
              f"边界样本比例高本身就是结论的一部分，报告里要写出来。")

    fl = [i for i in human if key[i] == 1]
    un = [i for i in human if key[i] == 0]
    tp = sum(human[i] for i in fl)
    fn = sum(human[i] for i in un)

    prec = tp / len(fl) if fl else float("nan")
    miss = fn / len(un) if un else float("nan")
    pl, ph = wilson(tp, len(fl))
    ml, mh = wilson(fn, len(un))
    print(f"\n== 总体 ==")
    print(f"被 judge 标为出戏：{len(fl)} 条，人工同意 {tp} 条 → 精确率 {prec:.0%}  (95% CI {pl:.0%}–{ph:.0%})")
    print(f"judge 没标出戏：  {len(un)} 条，人工认为其实出戏 {fn} 条 → 漏判率 {miss:.0%}  (95% CI {ml:.0%}–{mh:.0%})")

    # ---------------------------------------------------------------- 按方案
    by_scheme: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in human:
        s = scheme_of.get(i, "")
        by_scheme[s]["n"] += 1
        if key[i] == 1:
            by_scheme[s]["flagged"] += 1
            by_scheme[s]["tp"] += human[i]
        else:
            by_scheme[s]["unflagged"] += 1
            by_scheme[s]["fn"] += human[i]
    if len(by_scheme) > 1:
        print("\n== 按记忆方案（这是分层审核的核心产出）==")
        print("scheme\t被标n\t人工同意\t精确率\t95%CI\t未标n\t漏判")
        for s in sorted(by_scheme):
            b = by_scheme[s]
            p = b["tp"] / b["flagged"] if b["flagged"] else float("nan")
            lo, hi = wilson(b["tp"], b["flagged"])
            print(f"{s or '(未知)'}\t{b['flagged']}\t{b['tp']}\t"
                  f"{'n/a' if p != p else f'{p:.0%}'}\t"
                  f"{'n/a' if p != p else f'{lo:.0%}–{hi:.0%}'}\t{b['unflagged']}\t{b['fn']}")
        print("读法：各方案的精确率区间如果互相重叠，就没有证据说明 judge 的误报与方案有关，"
              "方案间的标记率差异可以继续按'生成差异'解释（但仍受总体精确率低的限制）；"
              "如果某个方案的精确率明显更低，那它的标记率就被系统性高估了，"
              "方案间的比较不能直接用。")

    # ---------------------------------------------------------------- 规则交叉表
    if any(leak_of.values()):
        print("\n== 规则检测器 vs 人工 ==")
        cells = defaultdict(int)
        for i in human:
            cells[(leak_of.get(i, 0) > 0, human[i] == 1)] += 1
        rl_tp, rl_fp = cells[(True, True)], cells[(True, False)]
        rl_fn, rl_tn = cells[(False, True)], cells[(False, False)]
        rp = rl_tp / (rl_tp + rl_fp) if (rl_tp + rl_fp) else float("nan")
        rr = rl_tp / (rl_tp + rl_fn) if (rl_tp + rl_fn) else float("nan")
        print(f"规则命中且人工认为出戏 {rl_tp}；规则命中但人工认为没出戏 {rl_fp}；"
              f"规则没命中但人工认为出戏 {rl_fn}；都认为没出戏 {rl_tn}")
        print(f"规则检测器：精确率 {'n/a' if rp != rp else f'{rp:.0%}'}，"
              f"召回率 {'n/a' if rr != rr else f'{rr:.0%}'}")
        print("预期：规则精确率应该明显高于 judge，召回率明显低于 judge。如果成立，"
              "报告里可以用规则命中率作为真实出戏率的**下界**，judge 标记率作为上界。")

    # ---------------------------------------------------------------- 召回估计
    f_rate = None
    if "n_flagged_total" in meta and meta.get("n_judged"):
        f_rate = meta["n_flagged_total"] / meta["n_judged"]
    elif meta.get("flag_rate_by_scheme"):
        rates = list(meta["flag_rate_by_scheme"].values())
        f_rate = sum(rates) / len(rates) if rates else None
    if fl and un and f_rate:
        true_pos_rate = f_rate * prec
        true_total = true_pos_rate + (1 - f_rate) * miss
        recall = true_pos_rate / true_total if true_total > 0 else float("nan")
        print(f"\nJudge 的总体标记率：{f_rate:.1%}")
        print(f"据此估计：真实出戏率约 {true_total:.1%}，judge 召回率约 {recall:.0%}"
              f"（样本小，仅作量级参考，区间很宽）")

    print("\n读法：精确率低 → judge 过度标记，实验里的 ooc_rate 绝对值被高估；"
          "漏判率高 → 低估。各方案之间的相对差异只在 judge 的错误跟记忆方案无关时才可信，"
          "这正是上面「按记忆方案」那张表要回答的问题。")


if __name__ == "__main__":
    main()
