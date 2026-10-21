"""汇总 run_eval.py 跑出来的 results/summary.csv，按 (model, memory_scheme) 分组，
打印每个指标的均值、以及有效（非空）样本数。

为什么要看"有效样本数"：四个指标里 fact_consistency / ooc_rate / evidence_traceability
几乎每条对话都算得出来，但 relationship_accuracy 目前预期大概率是全空的——因为
dialogues.jsonl 里 relationship_trajectory 字段还是空列表（弱标注脚本还没写），
run_eval.py 在没有真值可比对时会把这一项记成 None 并跳过，而不是硬凑一个 0/1。
所以这里单独打印每个指标的非空计数，而不是只看均值，免得把"全是 None 导致
均值=NaN"误读成"模型表现很差"。

用法（在仓库根目录下）：
    python src/eval/summarize_results.py
    # 或指定别的路径：
    python src/eval/summarize_results.py --csv results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_COLS = [
    "fact_consistency",
    "ooc_rate",
    "relationship_accuracy",
    "evidence_traceability",
]


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float_or_none(value: str):
    if value is None or value == "" or value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 run_eval.py 的 summary.csv")
    parser.add_argument("--csv", default="results/summary.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = load_rows(csv_path)
    print(f"读到 {len(rows)} 行（不含表头）来自 {csv_path}")

    # group[(model, scheme)][metric] = list[float]
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {m: [] for m in METRIC_COLS}
    )
    for row in rows:
        key = (row["model"], row["memory_scheme"])
        for metric in METRIC_COLS:
            val = to_float_or_none(row.get(metric))
            if val is not None:
                groups[key][metric].append(val)

    # 固定顺序打印，方便三种记忆方案横向比较
    models = sorted({k[0] for k in groups})
    schemes_order = ["raw_context", "vector_memory", "structured_memory"]

    header = ["model", "memory_scheme"] + [f"{m}(mean)" for m in METRIC_COLS] + [f"{m}(n)" for m in METRIC_COLS]
    print("\t".join(header))
    for model in models:
        for scheme in schemes_order:
            key = (model, scheme)
            if key not in groups:
                continue
            means = []
            counts = []
            for metric in METRIC_COLS:
                vals = groups[key][metric]
                counts.append(len(vals))
                means.append(f"{statistics.mean(vals):.3f}" if vals else "NaN")
            print("\t".join([model, scheme] + means + [str(c) for c in counts]))

    n_total_rows = len(rows)
    n_rel_nonnull = sum(len(g["relationship_accuracy"]) for g in groups.values())
    if n_rel_nonnull == 0:
        print(
            "\n提示：relationship_accuracy 全部为空（0/"
            f"{n_total_rows}），这是预期中的——dialogues.jsonl 的 "
            "relationship_trajectory 弱标注脚本还没实现，所以这一项目前对所有对话"
            "都跳过了，不代表模型或记忆方案在这项指标上表现差。"
        )


if __name__ == "__main__":
    main()
