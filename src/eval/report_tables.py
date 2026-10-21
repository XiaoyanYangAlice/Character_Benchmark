#!/usr/bin/env python3
"""report_tables.py —— 把三次重复的结果汇总成报告需要的几张表。不需要 GPU。

产出四块：
  A. 各指标的绝对水平（三次分别 + 平均）——报告里现在只有差值，读者无法判断
     "+36 个百分点"是从 6 到 42 还是从 60 到 96。
  B. 结构化记忆到底攒了多少事件 ——如果远超 max_events=8，"预算把早期信息挤掉了"
     这个解释就有了直接证据。
  C. 探针按「出处轮次」分层的正确率 ——这是关键的一张表。如果 vector 不管问题出自
     第几轮都答得差不多，而 structured 只在靠后的轮次答得对，那"相关度 vs 时间近远"
     就从假设变成了结果。
  D. 诊断列（judge 解析失败、证据重合度、抽取失败）。

用法：
    python src/eval/report_tables.py \
        --runs run0=results_all_summary.csv seed1=results_seed1 seed2=results_seed2 \
        --out docs/REPORT_TABLES.txt
每个 --runs 条目可以是 summary.csv 的路径，也可以是结果目录（会自动找目录下的
summary.csv / probe_results.csv / per_turn.csv）。
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

SCHEMES = ["no_memory", "raw_context", "vector_memory", "structured_memory",
           "structured_hybrid", "shuffled_facts"]
LEVEL_METRICS = ["probe_accuracy", "ooc_rate", "ooc_rate_late", "fact_consistency",
                 "fact_consistency_late", "leakage_rate", "evidence_use_rate",
                 "mean_context_chars"]


_BOOL = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0, "1": 1.0, "0": 0.0}


def f(v):
    """CSV 里的数值。注意 correct / ooc / leakage 这些列是 Python bool 被 csv
    模块写出去的，长相是 "True"/"False"，float() 解析不了——第一次跑 C 表全空
    就是这个原因。"""
    if v is None or v == "" or str(v).lower() == "none":
        return None
    s = str(v).strip()
    if s.lower() in _BOOL:
        return _BOOL[s.lower()]
    try:
        return float(s)
    except ValueError:
        return None


def rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def resolve(spec: str) -> tuple[str, Path, Path, Path]:
    tag, _, loc = spec.partition("=")
    p = Path(loc or tag)
    if p.is_dir():
        return tag, p / "summary.csv", p / "probe_results.csv", p / "per_turn.csv"
    return tag, p, p.parent / "probe_results.csv", p.parent / "per_turn.csv"


def fmt(v, nd=3):
    return "   n/a" if v is None else f"{v:{nd+4}.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag=路径，路径可以是 summary.csv 或结果目录")
    ap.add_argument("--out", default="")
    ap.add_argument("--early-cut", type=int, default=4,
                    help="探针分层：出处轮次 <= 这个值算'早期'，> 这个值算'靠后'")
    args = ap.parse_args()

    runs = [resolve(s) for s in args.runs]
    # 同一个 tag 可以指多个目录（比如第一次跑是 Qwen / Llama 两个目录，但属于同一次
    # 重复）。按出现顺序去重，表格里就只出现一列，数值由各目录按 (模型, 方案) 填入，
    # 不会互相覆盖——两个目录的模型是互斥的。
    tags = list(dict.fromkeys(t for t, _, _, _ in runs))
    L = []

    # ---------------------------------------------------------------- A 绝对水平
    L.append("A. 各方案的绝对水平（每条对话一个值，再对对话取平均）")
    L.append("   报告里必须给这张表：差值不说明水平。probe_accuracy 越高越好；")
    L.append("   ooc_rate / leakage_rate 越低越好；ooc_rate 是裁判标记率，不是真实出戏率。")
    lvl: dict = defaultdict(dict)
    for tag, summ, _, _ in runs:
        per = defaultdict(list)
        for r in rows(summ):
            for m in LEVEL_METRICS:
                v = f(r.get(m))
                if v is not None:
                    per[(r["model"], r["memory_scheme"], m)].append(v)
        for k, v in per.items():
            lvl[k][tag] = sum(v) / len(v)

    models = sorted({k[0] for k in lvl})
    for metric in LEVEL_METRICS:
        if not any(k[2] == metric for k in lvl):
            continue
        nd = 0 if metric == "mean_context_chars" else 3
        L.append(f"\n  [{metric}]")
        L.append("  " + f"{'model':<13}{'scheme':<19}" +
                 "".join(f"{t:>9}" for t in tags) + f"{'mean':>9}{'spread':>9}")
        for model in models:
            for sc in SCHEMES:
                k = (model, sc, metric)
                if k not in lvl:
                    continue
                vals = [lvl[k].get(t) for t in tags]
                got = [v for v in vals if v is not None]
                mean = sum(got) / len(got) if got else None
                spread = (max(got) - min(got)) if len(got) > 1 else None
                L.append("  " + f"{model:<13}{sc:<19}" +
                         "".join(fmt(v, nd) + "  " for v in vals) +
                         fmt(mean, nd) + "  " + fmt(spread, nd))
        L.append("  ↑ spread = 三次重复里的最大值减最小值。spread 和 mean 同量级时，"
                 "这个方案的绝对水平不稳定，报告里只能报差值，不能报水平。")

    # ---------------------------------------------------------------- B 事件预算
    L.append("\n\nB. 结构化方案实际攒了多少事件（检索时只带出 max_events 条）")
    L.append("   如果累计条数远大于 max_events，那么'早期信息被挤出预算'就有了直接证据，")
    L.append("   不再只是推测。")
    ev = defaultdict(list)
    for tag, summ, _, _ in runs:
        for r in rows(summ):
            if r["memory_scheme"] not in ("structured_memory", "structured_hybrid",
                                          "shuffled_facts"):
                continue
            v = f(r.get("n_events_extracted"))
            if v is not None:
                ev[(r["model"], r["memory_scheme"])].append(v)
    if any(ev.values()):
        L.append("  " + f"{'model':<13}{'scheme':<19}{'median':>8}{'mean':>8}{'min':>7}{'max':>7}{'n':>6}")
        for k in sorted(ev):
            v = ev[k]
            L.append("  " + f"{k[0]:<13}{k[1]:<19}{statistics.median(v):8.1f}"
                            f"{sum(v)/len(v):8.1f}{min(v):7.0f}{max(v):7.0f}{len(v):6d}")
    else:
        L.append("  （summary.csv 里没有 n_events_extracted 列，跳过）")

    # ---------------------------------------------------------------- C 探针分层
    L.append(f"\n\nC. 探针正确率，按「答案出自第几轮用户发言」分层"
             f"（<= {args.early_cut} 轮算早期）")
    L.append("   这是检验'相关度 vs 时间近远'的直接证据：")
    L.append("   - vector 在早/晚两层都差不多 → 它是按相关度取的，不受位置影响")
    L.append("   - structured 只在靠后那层答得对 → 早期事件被时间近远的预算挤掉了")
    L.append("   - raw_context 只在靠后那层答得对 → 窗口截断，符合预期（健全性检查）")
    strat: dict = defaultdict(lambda: [0, 0])     # (model, scheme, 层) -> [对, 总]
    by_idx: dict = defaultdict(lambda: [0, 0])    # (scheme, 轮次) -> [对, 总]
    n_probe_rows = 0
    for tag, _, probe, _ in runs:
        for r in rows(probe):
            n_probe_rows += 1
            c = f(r.get("correct"))
            idx = f(r.get("source_turn_index"))
            if c is None or idx is None:
                continue
            layer = "early" if idx <= args.early_cut else "late"
            cell = strat[(r["model"], r["memory_scheme"], layer)]
            cell[0] += int(c >= 0.5); cell[1] += 1
            cell2 = by_idx[(r["memory_scheme"], int(idx))]
            cell2[0] += int(c >= 0.5); cell2[1] += 1
    if strat:
        L.append("  " + f"{'model':<13}{'scheme':<19}{'early acc':>11}{'n':>6}"
                        f"{'late acc':>11}{'n':>6}{'late − early':>14}")
        for model in models:
            for sc in SCHEMES:
                e = strat.get((model, sc, "early")); l = strat.get((model, sc, "late"))
                if not e and not l:
                    continue
                ea = e[0] / e[1] if e and e[1] else None
                la = l[0] / l[1] if l and l[1] else None
                d = (la - ea) if (ea is not None and la is not None) else None
                L.append("  " + f"{model:<13}{sc:<19}{fmt(ea):>11}{(e[1] if e else 0):6d}"
                                f"{fmt(la):>11}{(l[1] if l else 0):6d}{fmt(d):>14}")
        L.append("\n  按出处轮次逐格（三次重复合并，各方案跨模型合并；格子小，只看趋势）：")
        idxs = sorted({i for _, i in by_idx})
        L.append("  " + f"{'scheme':<19}" + "".join(f"{'t'+str(i):>7}" for i in idxs))
        for sc in SCHEMES:
            cells = []
            for i in idxs:
                c = by_idx.get((sc, i))
                cells.append("     ." if not c or c[1] == 0 else f"{c[0]/c[1]:7.2f}")
            if any(x.strip() != "." for x in cells):
                L.append("  " + f"{sc:<19}" + "".join(cells))
        L.append("  （'.' = 这一轮没有探针。每格下面的样本量很小，不要单看一格下结论。）")
    elif n_probe_rows:
        L.append(f"  ⚠ 读到了 {n_probe_rows} 行探针记录，但没有一行能同时解析出 "
                 f"correct 和 source_turn_index。下面是第一行的实际内容，看是哪一列不对：")
        for _, _, probe, _ in runs:
            rr = rows(probe)
            if rr:
                for k, v in list(rr[0].items())[:6]:
                    L.append(f"      {k} = {v!r}")
                break
    else:
        L.append("  （没找到 probe_results.csv，跳过。它和 summary.csv 在同一个结果目录下。）")

    # ---------------------------------------------------------------- D 诊断
    L.append("\n\nD. 诊断列（报告的'可信度'一节要用）")
    diag = defaultdict(float)
    nturns = defaultdict(int)
    for tag, summ, _, pt in runs:
        for r in rows(summ):
            for c in ("n_judge_fail_ooc", "n_judge_fail_fact", "n_extraction_failures"):
                v = f(r.get(c))
                if v:
                    diag[(tag, c)] += v
            v = f(r.get("n_turns"))
            if v:
                nturns[tag] += int(v)
    L.append("  " + f"{'run':<9}{'judge解析失败(OOC)':>20}{'judge解析失败(事实)':>21}"
                    f"{'抽取失败':>11}{'总轮次':>10}")
    for t in tags:
        L.append("  " + f"{t:<9}{diag[(t,'n_judge_fail_ooc')]:20.0f}"
                        f"{diag[(t,'n_judge_fail_fact')]:21.0f}"
                        f"{diag[(t,'n_extraction_failures')]:11.0f}{nturns[t]:10d}")
    L.append("  失败数相对总轮次应当很小；占比超过百分之几就要在局限里写明。")

    body = "\n".join(L)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\n已写到 {args.out}")


if __name__ == "__main__":
    main()
