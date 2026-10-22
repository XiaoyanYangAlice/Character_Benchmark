"""补充统计（不需要 GPU、不需要 judge 服务，只用标准库，几秒跑完）。

为"向老师汇报"准备的四组数字：
  A. 对话长度分布，以及 raw_context 窗口实际截断了多少（回答"raw 是不是几乎等于完整历史"）
  B. 实际生成/判定的角色回复数（总体规模）
  C. judge 调用失败（解析失败）次数：从作业日志里数 "提醒：a/b 轮的 judge 调用失败"
  D. 12 个配对比较的多重比较校正（Holm）和 Bonferroni 校正后的置信区间

截断的精确算法（跟 dialogue_runner.run_dialogue 对齐）：第 k 个用户轮次时，传给
retrieve() 的 history 已经包含前面 2(k-1) 个轮次加上当前这句用户发言，长度 2k-1；
RawContextMemory 在长度超过窗口（config 里 max_turns，默认 20）时丢掉最早的轮次。
所以窗口 20 时，第 11 个用户轮次起开始截断。

用法（在仓库根目录）：
    python src/eval/supplementary_stats.py \\
        --dialogues data/processed/dialogues.jsonl \\
        --csv results/summary.csv results_seed1/summary.csv results_seed2/summary.csv \\
        --logs "logs/*.out" --out results/supplementary_stats.txt
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ["fact_consistency", "ooc_rate"]
PAIRS = [
    ("structured_memory", "raw_context"),
    ("vector_memory", "raw_context"),
    ("structured_memory", "vector_memory"),
]
WARN_RE = re.compile(r"\[(ooc_rate|character_fact_consistency)\] 提醒：(\d+)/(\d+)")
SKIP_RE = re.compile(r"\[跳过 (ooc_rate|fact_consistency)\]")


# ---------------------------------------------------------------- A. 对话长度
def dropped_per_turn(n_user_turns: int, window: int) -> list[int]:
    """第 k 个用户轮次时 raw_context 丢掉的轮数（k 从 1 开始）。"""
    return [max(0, (2 * k - 1) - window) for k in range(1, n_user_turns + 1)]


def length_stats(dialogues_path: str, max_user_turns: int, window: int) -> list[str]:
    lens = []
    with open(dialogues_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                lens.append(min(len(d["scripted_user_turns"]), max_user_turns))
    if not lens:
        return ["（dialogues.jsonl 为空）"]
    lens_sorted = sorted(lens)
    n = len(lens)

    def pct(p: float) -> float:
        return lens_sorted[min(n - 1, int(p * n))]

    first_trunc = window // 2 + 1  # 窗口 20 -> 第 11 个用户轮次
    any_trunc = sum(1 for k in lens if k >= first_trunc)
    total_turns = sum(lens)
    trunc_turns = sum(sum(1 for x in dropped_per_turn(k, window) if x > 0) for k in lens)
    dropped_all = [x for k in lens for x in dropped_per_turn(k, window)]
    heavy = sum(1 for k in lens if k >= first_trunc + 4)
    out = [
        f"对话数 n = {n}（用户轮次上限 {max_user_turns}，窗口 {window} 轮）",
        f"用户轮次：最小 {lens_sorted[0]}，25% 分位 {pct(0.25)}，中位数 {statistics.median(lens)}，"
        f"75% 分位 {pct(0.75)}，最大 {lens_sorted[-1]}，均值 {statistics.mean(lens):.1f}",
        "分布：" + "，".join(f"{k}轮×{lens.count(k)}" for k in sorted(set(lens))),
        f"至少有一次被截断的对话（用户轮次 ≥ {first_trunc}）：{any_trunc}/{n}（{any_trunc / n:.0%}）",
        f"截断较多的对话（用户轮次 ≥ {first_trunc + 4}，即至少丢掉 9 轮以上）：{heavy}/{n}（{heavy / n:.0%}）",
        f"发生截断的角色回复：{trunc_turns}/{total_turns}（{trunc_turns / total_turns:.0%}）；"
        f"全部回复平均丢掉 {statistics.mean(dropped_all):.2f} 轮，最多一次丢 {max(dropped_all)} 轮",
        "读法：比例低 -> 'raw 几乎等于完整历史' 成立；比例高 -> 该说法只对部分对话成立，要改口。",
    ]
    return out


# ---------------------------------------------------------------- B. 规模
def read_csv_rows(paths: list[str]) -> list[list[dict]]:
    runs = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            runs.append(list(csv.DictReader(f)))
    return runs


def scale_stats(runs: list[list[dict]]) -> list[str]:
    out = []
    per_cell = defaultdict(lambda: [0, 0])  # (model, scheme) -> [角色回复数, 行数]
    for rows in runs:
        for r in rows:
            try:
                n_turns = int(float(r["n_turns"]))
            except (KeyError, ValueError):
                continue
            c = per_cell[(r["model"], r["memory_scheme"])]
            c[0] += n_turns // 2  # n_turns 含用户轮次，角色回复约为一半
            c[1] += 1
    tot = sum(v[0] for v in per_cell.values())
    out.append(f"读入 {len(runs)} 次重复，共 {sum(v[1] for v in per_cell.values())} 行；角色回复约 {tot} 条")
    for (m, s), (n_rep, n_rows) in sorted(per_cell.items()):
        out.append(f"  {m:<14}{s:<20}行数 {n_rows:<4}角色回复约 {n_rep}")
    miss = defaultdict(int)
    for rows in runs:
        for r in rows:
            for m in METRICS:
                if r.get(m) in (None, "", "None"):
                    miss[m] += 1
    out.append("指标留空的行数（算不出来，不是 0 分）：" + (str(dict(miss)) if miss else "无"))
    return out


# ---------------------------------------------------------------- C. judge 失败次数
def log_stats(patterns: list[str]) -> list[str]:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        return ["没有匹配到日志文件（--logs 的路径/通配符是否写对？）"]
    out = ["说明：下面统计的是传入的**每个日志文件**里出现的失败提醒；日志里包含已作废的旧运行和冒烟测试，"
           "汇报时只引用最终三次主实验对应的日志。"]
    grand = defaultdict(lambda: [0, 0])
    for f in files:
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        per = defaultdict(lambda: [0, 0, 0])  # name -> [失败轮数, 涉及总轮数, 出现次数]
        for m in WARN_RE.finditer(text):
            p = per[m.group(1)]
            p[0] += int(m.group(2))
            p[1] += int(m.group(3))
            p[2] += 1
        skips = len(SKIP_RE.findall(text))
        if per or skips:
            desc = "；".join(f"{k}: {v[0]} 轮失败/{v[1]} 轮（{v[2]} 个对话）" for k, v in sorted(per.items()))
            out.append(f"  {Path(f).name}: {desc or '无逐轮失败'}" + (f"；整项指标被跳过 {skips} 次" if skips else ""))
            for k, v in per.items():
                grand[k][0] += v[0]
                grand[k][1] += v[1]
    if not grand:
        out.append("  所有日志里都没有出现 judge 调用失败提醒。")
    else:
        out.append("合计（所有传入日志）：" + "；".join(f"{k}: {v[0]} 轮失败/{v[1]} 轮" for k, v in sorted(grand.items())))
    return out


# ---------------------------------------------------------------- D. 多重比较校正
def _to_float(v):
    if v is None or v == "" or str(v).lower() == "none":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def boot_means(diffs: list[float], n_boot: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    out = []
    for _ in range(n_boot):
        out.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out


def comparisons(runs: list[list[dict]], n_boot: int) -> list[dict]:
    per_run = []
    for rows in runs:
        run = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            for m in METRICS:
                v = _to_float(r.get(m))
                if v is not None:
                    run[(r["model"], m)][r["memory_scheme"]][r["dialogue_id"]] = v
        per_run.append(run)
    keys = {k for run in per_run for k in run}
    avg = defaultdict(lambda: defaultdict(dict))
    for key in keys:
        for scheme in {s for run in per_run for s in run.get(key, {})}:
            ids = {i for run in per_run for i in run.get(key, {}).get(scheme, {})}
            for i in ids:
                vals = [run[key][scheme][i] for run in per_run if i in run.get(key, {}).get(scheme, {})]
                avg[key][scheme][i] = sum(vals) / len(vals)
    results = []
    for (model, metric) in sorted(avg):
        for a, b in PAIRS:
            by = avg[(model, metric)]
            ids = sorted(set(by.get(a, {})) & set(by.get(b, {})))
            if not ids:
                continue
            diffs = [by[a][i] - by[b][i] for i in ids]
            n = len(diffs)
            boots = boot_means(diffs, n_boot, seed=0)
            mean = sum(diffs) / n
            le = sum(1 for x in boots if x <= 0)
            ge = sum(1 for x in boots if x >= 0)
            p = min(1.0, 2 * (min(le, ge) + 1) / (n_boot + 1))
            results.append({"model": model, "metric": metric, "pair": f"{a} vs {b}", "n": n, "mean": mean,
                            "boots": boots, "p": p})
    return results


def holm(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def ci_at(boots: list[float], alpha: float) -> tuple[float, float]:
    n = len(boots)
    lo = boots[max(0, int((alpha / 2) * n))]
    hi = boots[min(n - 1, int((1 - alpha / 2) * n) - 1)]
    return lo, hi


def multiple_comparison_stats(runs: list[list[dict]], n_boot: int) -> list[str]:
    res = comparisons(runs, n_boot)
    if not res:
        return ["没有可比较的数据"]
    m = len(res)
    ps = [r["p"] for r in res]
    adj_all = holm(ps)
    ooc_idx = [i for i, r in enumerate(res) if r["metric"] == "ooc_rate"]
    adj_ooc = holm([ps[i] for i in ooc_idx]) if ooc_idx else []
    ooc_adj_map = {i: adj_ooc[k] for k, i in enumerate(ooc_idx)}
    out = [f"{m} 个比较，bootstrap {n_boot} 次；p 值是双侧 bootstrap p；Holm 对全部 {m} 个比较校正；"
           f"'Holm(仅 ooc)' 只在 {len(ooc_idx)} 个 ooc 比较内校正；Bonf.CI 是 Bonferroni 校正后的 "
           f"{100 * (1 - 0.05 / m):.2f}% 置信区间（按 {m} 个比较）。",
           "model\tmetric\tA vs B\tmean\t95%CI\tBonf.CI\tp\tHolm(全部)\tHolm(仅ooc)\t校正后是否仍显著(全部)"]
    for i, r in enumerate(res):
        lo, hi = ci_at(r["boots"], 0.05)
        blo, bhi = ci_at(r["boots"], 0.05 / m)
        h_all = adj_all[i]
        h_ooc = f"{ooc_adj_map[i]:.4f}" if i in ooc_adj_map else "-"
        sig = "是" if h_all < 0.05 else "否"
        out.append(f"{r['model']}\t{r['metric']}\t{r['pair']}\t{r['mean']:+.3f}\t[{lo:+.3f}, {hi:+.3f}]\t"
                   f"[{blo:+.3f}, {bhi:+.3f}]\t{r['p']:.4f}\t{h_all:.4f}\t{h_ooc}\t{sig}")
    out.append("读法：Bonf.CI 含 0 或 Holm >= 0.05，说明校正后不能再说该差异'显著'；bootstrap p 在 "
               f"p<{2 / (n_boot + 1):.0e} 附近已到重采样精度上限，不要报告更小的 p。")
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="data/processed/dialogues.jsonl")
    ap.add_argument("--csv", nargs="+", default=["results/summary.csv"])
    ap.add_argument("--logs", nargs="*", default=[], help="日志文件或通配符，如 \"logs/*.out\"")
    ap.add_argument("--max-user-turns", type=int, default=20)
    ap.add_argument("--window", type=int, default=20, help="raw_context 窗口（config 里 max_turns）")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    runs = read_csv_rows(args.csv)
    sections = [
        ("A. 对话长度与 raw_context 截断", length_stats(args.dialogues, args.max_user_turns, args.window)),
        ("B. 规模与留空", scale_stats(runs)),
        ("C. judge 调用失败（来自日志）", log_stats(args.logs) if args.logs else ["未传 --logs，跳过"]),
        ("D. 多重比较校正", multiple_comparison_stats(runs, args.n_boot)),
    ]
    text = []
    for title, lines in sections:
        text += [f"== {title} ==", *lines, ""]
    body = "\n".join(text)
    print(body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"已写到 {args.out}")


if __name__ == "__main__":
    main()
