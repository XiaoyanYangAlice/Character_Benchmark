"""两份人工标注（比如 Claude 的 labels.csv 和你自己的 labels_user.csv）的一致性，
并分别算 judge 对着每一份的精确率/漏判率。只用标准库。

标注文件格式：item_id,human_ooc,note；human_ooc 填 1（出戏）/ 0（没出戏）/ ?（拿不准）。
"?" 不参与 kappa 和精确率计算，但会单独统计条数——灰区有多大本身就是一个结果。

用法：
    python src/eval/ooc_label_agreement.py --a results/ooc_audit/labels.csv \\
        --b results/ooc_audit/labels_user.csv [--key results/ooc_audit/key.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_labels(path: Path) -> dict[int, str]:
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        v = (r.get("human_ooc") or "").strip()
        if v in ("0", "1", "?"):
            out[int(r["item_id"])] = v
    return out


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def judge_vs(labels: dict[int, str], judge: dict[int, int]) -> str:
    ids = [i for i in labels if labels[i] in ("0", "1") and i in judge]
    fl = [i for i in ids if judge[i] == 1]
    un = [i for i in ids if judge[i] == 0]
    tp = sum(labels[i] == "1" for i in fl)
    fn = sum(labels[i] == "1" for i in un)
    prec = f"{tp}/{len(fl)} = {tp / len(fl):.0%}" if fl else "n/a"
    miss = f"{fn}/{len(un)} = {fn / len(un):.0%}" if un else "n/a"
    return f"精确率 {prec}；漏判率 {miss}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="标注 A（例如 Claude 的 labels.csv）")
    ap.add_argument("--b", required=True, help="标注 B（例如你自己的 labels_user.csv）")
    ap.add_argument("--key", default=None, help="key.csv，给了就同时算 judge 对着每份标注的精确率")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    args = ap.parse_args()

    a, b = read_labels(Path(args.a)), read_labels(Path(args.b))
    common = sorted(set(a) & set(b))
    print(f"{args.name_a} 填了 {len(a)} 条，{args.name_b} 填了 {len(b)} 条，两边都填了 {len(common)} 条")
    for name, lab in ((args.name_a, a), (args.name_b, b)):
        print(f"  {name}：标出戏 {sum(v == '1' for v in lab.values())}，没出戏 {sum(v == '0' for v in lab.values())}，"
              f"拿不准 {sum(v == '?' for v in lab.values())}")

    both = [i for i in common if a[i] in ("0", "1") and b[i] in ("0", "1")]
    pairs = [(int(a[i]), int(b[i])) for i in both]
    if pairs:
        agree = sum(x == y for x, y in pairs) / len(pairs)
        print(f"\n在两边都有明确判断的 {len(both)} 条上：一致率 {agree:.0%}，Cohen's kappa {cohen_kappa(pairs):.2f}")
        print("（参考：kappa < 0.4 一般认为一致性弱，0.4–0.6 中等，> 0.6 较好；样本只有 40 条，kappa 本身误差很大）")
    diff = [i for i in both if a[i] != b[i]]
    unsure = [i for i in common if "?" in (a[i], b[i])]
    print(f"\n判断不一致的条目 ({len(diff)})：{diff}")
    for i in diff:
        print(f"  item {i}: {args.name_a}={a[i]}  {args.name_b}={b[i]}")
    if unsure:
        print(f"至少一方拿不准的条目 ({len(unsure)})：{unsure}")

    if args.key:
        judge = {int(r["item_id"]): int(r["judge_ooc"]) for r in csv.DictReader(open(args.key, encoding="utf-8"))}
        print("\njudge（主实验那次判定，key.csv）对着各份标注：")
        print(f"  {args.name_a}：{judge_vs(a, judge)}")
        print(f"  {args.name_b}：{judge_vs(b, judge)}")
        print("注：这 40 条是按 judge 的判定分层抽的（20 标 + 20 未标），精确率/漏判率分别在两层内算，"
              "不是 judge 在全体回复上的表现。")


if __name__ == "__main__":
    main()
