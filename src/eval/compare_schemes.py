"""对 results/summary.csv 做配对比较：同一个模型、同一条对话，两种记忆方案的
指标差值（按 dialogue_id 配对），报告均值差、bootstrap 95% 置信区间、
胜/平/负条数。

为什么要配对：scripted_user_turns 在三种方案间是固定的，同一条对话在不同方案下
只有记忆上下文不同，配对差能把"对话本身难不难"这个方差去掉，比直接比两个均值
的差更有说服力。只用标准库，SURF 上直接跑。

置信区间不含 0 才说明差异在这批 61 条对话上比较稳定；含 0 就不要写"提升"。
这只是对这 61 条对话的重采样，不代表对 PIPPA 全体角色卡泛化。

多次重复实验（不同 seed）：--csv 可以传多个文件。同一 (模型, 方案, 对话) 在各次
重复里的指标先取平均，再按对话做配对 bootstrap——对话才是重采样的单位，seed 只是
降低每条对话上的采样噪声。另外会按每次重复单独打印均值差，看方向是否一致。

用法：python src/eval/compare_schemes.py [--csv results/summary.csv [更多csv ...]]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

METRICS = ["fact_consistency", "ooc_rate"]  # evidence_traceability 是代理指标，不做比较
PAIRS = [
    ("structured_memory", "raw_context"),
    ("vector_memory", "raw_context"),
    ("structured_memory", "vector_memory"),
]


def _to_float(v):
    if v is None or v == "" or str(v).lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def bootstrap_ci(diffs: list[float], n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["results/summary.csv"])
    args = ap.parse_args()

    # runs[k][(model, metric)][scheme][dialogue_id] = value
    runs = []
    for path in args.csv:
        with Path(path).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        run: dict = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            for m in METRICS:
                v = _to_float(r.get(m))
                if v is not None:
                    run[(r["model"], m)][r["memory_scheme"]][r["dialogue_id"]] = v
        runs.append(run)
    print(f"读入 {len(runs)} 次重复：{', '.join(args.csv)}")

    # data[(model, metric)][scheme][dialogue_id] = 各次重复的平均值
    data: dict = defaultdict(lambda: defaultdict(dict))
    keys = {k for run in runs for k in run}
    for key in keys:
        for scheme in {sc for run in runs for sc in run.get(key, {})}:
            ids = {i for run in runs for i in run.get(key, {}).get(scheme, {})}
            for i in ids:
                vals = [run[key][scheme][i] for run in runs
                        if i in run.get(key, {}).get(scheme, {})]
                data[key][scheme][i] = sum(vals) / len(vals)

    print("model\tmetric\tA vs B\tn\tmean(A-B)\t95%CI\tA<B / tie / A>B")
    for (model, metric) in sorted(data):
        by_scheme = data[(model, metric)]
        for a, b in PAIRS:
            ids = sorted(set(by_scheme.get(a, {})) & set(by_scheme.get(b, {})))
            if not ids:
                continue
            diffs = [by_scheme[a][i] - by_scheme[b][i] for i in ids]
            mean = sum(diffs) / len(diffs)
            lo, hi = bootstrap_ci(diffs)
            lt = sum(d < -1e-9 for d in diffs)
            gt = sum(d > 1e-9 for d in diffs)
            eq = len(diffs) - lt - gt
            print(f"{model}\t{metric}\t{a} vs {b}\t{len(ids)}\t{mean:+.3f}\t[{lo:+.3f}, {hi:+.3f}]\t{lt}/{eq}/{gt}")

    if len(runs) > 1:
        print("\n每次重复单独的均值差（方向是否一致）：")
        for (model, metric) in sorted(data):
            for a, b in PAIRS:
                per_run = []
                for run in runs:
                    sa = run.get((model, metric), {}).get(a, {})
                    sb = run.get((model, metric), {}).get(b, {})
                    ids = set(sa) & set(sb)
                    per_run.append(sum(sa[i] - sb[i] for i in ids) / len(ids) if ids else float("nan"))
                print(f"{model}\t{metric}\t{a} vs {b}\t" + "  ".join(f"{d:+.3f}" for d in per_run))

    print("\n读法：fact_consistency 越高越好、ooc_rate 越低越好。"
          "CI 含 0 → 这批对话上看不出稳定差异。")


if __name__ == "__main__":
    main()
