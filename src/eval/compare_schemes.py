"""对 summary.csv 做配对比较：同一个模型、同一条对话，两种记忆方案的指标差值
（按 dialogue_id 配对），报告均值差、bootstrap 95% 置信区间、Holm 校正后的 p、
胜/平/负条数。

为什么要配对：scripted_user_turns 在各方案间是固定的，同一条对话在不同方案下
只有记忆上下文不同，配对差能把"对话本身难不难"这个方差去掉，比直接比两个均值
的差更有说服力。只用标准库，SURF 上直接跑。

置信区间不含 0 才说明差异在这批对话上比较稳定；含 0 就不要写"提升"。
这只是对这批对话的重采样，不代表对 PIPPA 全体角色卡泛化。

多次重复实验（不同 seed）：--csv 可以传多个文件。同一 (模型, 方案, 对话) 在各次
重复里的指标先取平均，再按对话做配对 bootstrap——对话才是重采样的单位，seed 只是
降低每条对话上的采样噪声。另外会按每次重复单独打印均值差，看方向是否一致。

v2 变化：
  - 指标扩展到 ooc_rate / ooc_rate_late / fact_consistency / probe_accuracy /
    leakage_rate。**probe_accuracy 是唯一直接测记忆的指标**，优先看它。
  - 比较对里加入 no_memory 和 shuffled_facts 两个对照（见 configs/experiment.yaml
    里的说明），用来判断效应是"记忆内容"还是"又贴了一遍人设"。
  - 内置 Holm 多重比较校正。v1 报告"12 个比较未校正"，是事后才补的。
  - 打印每个方案的平均上下文长度，用来检查"差异是不是单纯因为塞进去的字更多"。

用法：python src/eval/compare_schemes.py --csv results/summary.csv [更多csv ...]
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# 顺序即优先级：报告里先看 probe_accuracy（直接测记忆），再看后段 ooc，
# 最后才是整段 ooc。leakage_rate 是规则检测，不依赖 judge，作为交叉验证。
METRICS = ["probe_accuracy", "ooc_rate_late", "ooc_rate", "fact_consistency_late",
           "fact_consistency", "leakage_rate"]
HIGHER_IS_BETTER = {"probe_accuracy", "fact_consistency", "fact_consistency_late"}

# 每个比较都有明确的解释用途，不是穷举所有组合——穷举会把多重比较的代价
# 白白抬上去。
PAIRS = [
    ("raw_context", "no_memory"),            # 给历史原文有用吗
    ("vector_memory", "no_memory"),          # 检索有用吗
    ("structured_memory", "no_memory"),      # 结构化事实有用吗
    ("structured_memory", "raw_context"),    # v1 的主结论
    ("vector_memory", "raw_context"),
    ("structured_memory", "vector_memory"),
    ("structured_memory", "shuffled_facts"),  # 关键对照：内容对不对要紧吗
    ("shuffled_facts", "no_memory"),          # 错的事实比没有事实好还是坏
    # v2 修复后新增。structured_hybrid 与 structured_memory 共用同一个抽取器，
    # 只有检索策略不同（recency -> hybrid），所以下面第一条单独量化"检索预算"
    # 这一个因素的贡献，第二条量化修完检索之后离向量检索还差多少。
    ("structured_hybrid", "structured_memory"),  # 换检索策略贡献了多少
    ("structured_hybrid", "no_memory"),          # 修完之后比没有记忆强了吗
    ("structured_hybrid", "vector_memory"),      # 剩下的差距
    ("structured_hybrid", "shuffled_facts"),     # 同样的内容对照
]


def _to_float(v):
    if v is None or v == "" or str(v).lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def boot_means(diffs: list[float], n_boot: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means


def ci_at(boots: list[float], alpha: float) -> tuple[float, float]:
    n = len(boots)
    return boots[max(0, int((alpha / 2) * n))], boots[min(n - 1, int((1 - alpha / 2) * n) - 1)]


def boot_p(boots: list[float], n_boot: int) -> float:
    """双侧 bootstrap p：重采样均值落在 0 另一侧的比例（加 1 平滑，避免报 p=0）。
    下限约 2/(n_boot+1)，比这更小的 p 是重采样精度的伪精确，不要报告。"""
    le = sum(1 for x in boots if x <= 0)
    ge = sum(1 for x in boots if x >= 0)
    return min(1.0, 2 * (min(le, ge) + 1) / (n_boot + 1))


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni 逐步校正。比 Bonferroni 的功效高，同样控制 FWER。"""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["results/summary.csv"])
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out", default="", help="把表格同时写到这个文件")
    args = ap.parse_args()

    runs = []
    for path in args.csv:
        with Path(path).open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        run: dict = defaultdict(lambda: defaultdict(dict))
        ctx: dict = defaultdict(list)
        for r in rows:
            for m in METRICS:
                v = _to_float(r.get(m))
                if v is not None:
                    run[(r["model"], m)][r["memory_scheme"]][r["dialogue_id"]] = v
            c = _to_float(r.get("mean_context_chars"))
            if c is not None:
                ctx[(r["model"], r["memory_scheme"])].append(c)
        runs.append((run, ctx))
    lines = [f"读入 {len(runs)} 次重复：{', '.join(args.csv)}"]

    # 上下文预算：先检查公平性，再看结果。预算差太多（比如某方案是别人的 3 倍）
    # 时，任何方案间差异都可能只是"上下文长度"的效应。
    lines.append("\n各方案平均注入上下文长度（字符，用来检查预算是否可比）：")
    ctx_all: dict = defaultdict(list)
    for _, ctx in runs:
        for k, v in ctx.items():
            ctx_all[k] += v
    for (model, scheme), vals in sorted(ctx_all.items()):
        lines.append(f"  {model:<14}{scheme:<20}{sum(vals) / len(vals):8.0f}")

    # data[(model, metric)][scheme][dialogue_id] = 各次重复的平均值
    data: dict = defaultdict(lambda: defaultdict(dict))
    keys = {k for run, _ in runs for k in run}
    for key in keys:
        for scheme in {sc for run, _ in runs for sc in run.get(key, {})}:
            ids = {i for run, _ in runs for i in run.get(key, {}).get(scheme, {})}
            for i in ids:
                vals = [run[key][scheme][i] for run, _ in runs
                        if i in run.get(key, {}).get(scheme, {})]
                data[key][scheme][i] = sum(vals) / len(vals)

    results = []
    for (model, metric) in sorted(data):
        by_scheme = data[(model, metric)]
        for a, b in PAIRS:
            ids = sorted(set(by_scheme.get(a, {})) & set(by_scheme.get(b, {})))
            if not ids:
                continue
            diffs = [by_scheme[a][i] - by_scheme[b][i] for i in ids]
            boots = boot_means(diffs, args.n_boot)
            results.append({
                "model": model, "metric": metric, "pair": f"{a} vs {b}", "n": len(ids),
                "mean": sum(diffs) / len(diffs), "boots": boots,
                "p": boot_p(boots, args.n_boot),
                "lt": sum(d < -1e-9 for d in diffs), "gt": sum(d > 1e-9 for d in diffs),
            })

    if not results:
        lines.append("\n没有可比较的数据（检查 summary.csv 里的列名和方案名）")
        print("\n".join(lines))
        return

    # 多重比较的"族"按 (模型, 指标) 划分，在族内做 Holm 校正，不是对全部
    # 几十个比较一起校正。理由：这几个指标回答的是不同的问题（记忆准确率、
    # 出戏率、事实一致性），把它们合成一个族会让校正过度保守，连主指标都
    # 检不出来。这个划分必须**在看结果之前**定好并写进文档，否则就是看完
    # 结果再挑一个显著的族，等于没校正。本项目的主指标是 probe_accuracy
    # （唯一直接测记忆的指标），其余指标按探索性分析对待。
    adj = [0.0] * len(results)
    fam: dict = defaultdict(list)
    for i, r in enumerate(results):
        fam[(r["model"], r["metric"])].append(i)
    for idxs in fam.values():
        for i, a in zip(idxs, holm([results[i]["p"] for i in idxs])):
            adj[i] = a
    fam_sizes = sorted({len(v) for v in fam.values()})
    lines.append(f"\n共 {len(results)} 个比较，bootstrap {args.n_boot} 次。"
                 f"Holm 校正在 (模型 × 指标) 族内做，共 {len(fam)} 个族，"
                 f"每族 {'/'.join(map(str, fam_sizes))} 个比较"
                 f"（族的划分是事先定的，主指标是 probe_accuracy）。差值 = A − B。")
    lines.append("model\tmetric\tA vs B\tn\tmean(A-B)\t95%CI\tp\tHolm p(族内)\t校正后显著\tA<B/tie/A>B\t方向")
    for r, p_adj in zip(results, adj):
        lo, hi = ci_at(r["boots"], 0.05)
        better_high = r["metric"] in HIGHER_IS_BETTER
        if p_adj >= 0.05:
            direction = "—"
        elif (r["mean"] > 0) == better_high:
            direction = "A 更好"
        else:
            direction = "A 更差"
        tie = r["n"] - r["lt"] - r["gt"]
        lines.append(f"{r['model']}\t{r['metric']}\t{r['pair']}\t{r['n']}\t{r['mean']:+.3f}\t"
                     f"[{lo:+.3f}, {hi:+.3f}]\t{r['p']:.4f}\t{p_adj:.4f}\t"
                     f"{'是' if p_adj < 0.05 else '否'}\t{r['lt']}/{tie}/{r['gt']}\t{direction}")

    if len(runs) > 1:
        lines.append("\n每次重复单独的均值差（方向是否一致；不一致就不要下结论）：")
        for (model, metric) in sorted(data):
            for a, b in PAIRS:
                per_run = []
                for run, _ in runs:
                    sa = run.get((model, metric), {}).get(a, {})
                    sb = run.get((model, metric), {}).get(b, {})
                    ids = set(sa) & set(sb)
                    per_run.append(sum(sa[i] - sb[i] for i in ids) / len(ids) if ids else float("nan"))
                if any(p == p for p in per_run):  # 至少一个不是 nan
                    lines.append(f"{model}\t{metric}\t{a} vs {b}\t" +
                                 "  ".join(f"{d:+.3f}" for d in per_run))

    lines += [
        "",
        "读法：",
        "- probe_accuracy 越高越好，是唯一直接测记忆的指标；ooc_rate / leakage_rate 越低越好。",
        "- ooc_rate 是 **judge 的标记率**，审核显示精确率约 40%（docs/JUDGE_AUDIT.md），"
        "不要当成真实出戏率。leakage_rate 是规则检测，高精确低召回，给的是下界。",
        "- 先看 structured vs shuffled_facts：不显著就说明效应来自'又贴了一遍人设'，"
        "不是记忆内容，这时 structured vs raw 的差异不能解释成记忆更好。",
        "- CI 含 0 或 Holm p >= 0.05 → 这批对话上看不出稳定差异，不要写'提升'。",
        "- Holm 在 (模型 × 指标) 族内校正。族的划分是事先定的，不是看完结果再挑的；"
        "报告里要写明这一点，否则多重比较校正没有意义。",
        f"- bootstrap p 的下限约 {2 / (args.n_boot + 1):.0e}，比这更小是伪精确。",
    ]
    body = "\n".join(lines)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\n已写到 {args.out}")


if __name__ == "__main__":
    main()
