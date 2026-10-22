"""OOC judge 的 prompt 回归测试：拿 results/ooc_audit 里已经标好的 40 条
（sheet.md + key.csv + labels.csv），分别用 V1 和 V2 prompt 重判，每条重复 N 次，
对着标注算 TP/FP/FN/TN、精确率、召回率，并列出判定变了的条目。

重要的局限（写进结论前必须记得）：
1. V2 是看着这 40 条的错误写的，在它们上面变好是"样本内"的，会偏乐观；真正的检验是
   之后的分层审核（新抽的、没看过的样本）。
2. 这 40 条是按旧 judge 的判定分层抽的（20 标 + 20 未标），不是随机样本，所以这里的
   精确率/召回率只用来比较两个 prompt，不能当 judge 在全体回复上的表现。
3. 标注者是 Claude，不是人工；边界样本（item 1、17、23、35）的标注本身有争议，
   报告里同时给出"去掉这几条"的结果。

为了拿到每条的完整角色设定（sheet.md 里被截断到 700 字），按"回复原文 + dialogue_id +
turn_id"去 raw_outputs 里找回原始轮次；找不到才退回用被截断的设定并提示。

需要 judge 的 vLLM 服务已起好（见 slurm/ooc_regress.slurm）。用法：
    python src/eval/ooc_judge_regress.py --config configs/experiment.yaml \\
        --audit-dir results/ooc_audit \\
        --raw-dirs results/raw_outputs results_seed1/raw_outputs results_seed2/raw_outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BORDERLINE = {1, 17, 23, 35}  # 标注本身有争议的条目


def parse_sheet(path: Path) -> dict[int, dict]:
    """从 sheet.md 取出每个 item 的角色名、（截断的）设定、回复原文。"""
    text = path.read_text(encoding="utf-8")
    items = {}
    parts = re.split(r"\n## item (\d+)\n", text)
    for i in range(1, len(parts), 2):
        n, body = int(parts[i]), parts[i + 1]
        m = re.search(r"\*\*角色回复\*\*：(.*?)\n\n---", body + "\n\n---", re.S)
        card = re.search(r"\*\*角色设定\*\*：(.*?)\n\*\*用户上一句", body, re.S)
        items[n] = {"reply": m.group(1).strip() if m else "",
                    "card_truncated": card.group(1).strip() if card else ""}
    return items


def find_card(item_key: dict, reply: str, raw_dirs: list[str], dialogues: dict):
    """用 key.csv 的 dialogue_id/scheme/turn_id 在 raw_outputs 里核对回复原文，找到则返回完整设定。"""
    did = item_key["dialogue_id"]
    if did not in dialogues:
        return None
    for d in raw_dirs:
        for path in Path(d).glob(f"qwen2.5-7b__{item_key['scheme']}__{did}.json"):
            for t in json.loads(path.read_text(encoding="utf-8")):
                if str(t.get("turn_id")) == str(item_key["turn_id"]) and t.get("text", "").strip() == reply:
                    return dialogues[did]["character_card"]
    return None


def judge_once(client, judge_model, prompt_tpl, card_text, reply) -> bool | None:
    from src.eval.metrics import call_judge
    r = call_judge(client, judge_model, prompt_tpl.format(character_card=card_text, turn_text=reply))
    if r is None or "ooc" not in r:
        return None
    return bool(r["ooc"])


def confusion(flag_rate: dict[int, float], human: dict[int, int], ids) -> dict:
    """flag_rate[i] 是该条在 N 次重复里被判出戏的比例；用期望计数，避免多数投票的随意性。"""
    tp = sum(flag_rate[i] for i in ids if human[i] == 1)
    fp = sum(flag_rate[i] for i in ids if human[i] == 0)
    fn = sum(1 - flag_rate[i] for i in ids if human[i] == 1)
    tn = sum(1 - flag_rate[i] for i in ids if human[i] == 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--judge-port", default="8002")
    ap.add_argument("--base-url-template", default="http://localhost:{port}/v1")
    ap.add_argument("--audit-dir", default="results/ooc_audit")
    ap.add_argument("--raw-dirs", nargs="+", default=["results/raw_outputs"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="qwen14b", help="区分不同 judge 的输出文件名")
    ap.add_argument("--out", default=None, help="逐条结果 csv，默认 <audit-dir>/regress_per_item_<tag>.csv")
    args = ap.parse_args()

    import yaml
    from openai import OpenAI
    from src.eval.judge_prompts import OOC_JUDGE_PROMPT, OOC_JUDGE_PROMPT_V2
    from src.eval.run_eval import character_card_text, load_dialogues

    d = Path(args.audit_dir)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dialogues = {x["dialogue_id"]: x for x in load_dialogues(config["data"]["dialogues_path"])}
    sheet = parse_sheet(d / "sheet.md")
    key = {int(r["item_id"]): r for r in csv.DictReader(open(d / "key.csv", encoding="utf-8"))}
    human = {int(r["item_id"]): int(r["human_ooc"])
             for r in csv.DictReader(open(d / "labels.csv", encoding="utf-8")) if (r["human_ooc"] or "").strip() in ("0", "1")}
    ids = sorted(human)

    cards, fallback = {}, []
    for i in ids:
        card = find_card(key[i], sheet[i]["reply"], args.raw_dirs, dialogues)
        if card is None:
            fallback.append(i)
            cards[i] = sheet[i]["card_truncated"]
        else:
            cards[i] = character_card_text(card)
    if fallback:
        print(f"警告：{len(fallback)} 条没在 raw_outputs 里找回完整设定，用 sheet.md 里被截断的版本：{fallback}")

    client = OpenAI(base_url=args.base_url_template.format(port=args.judge_port), api_key="EMPTY")
    judge_model = config["judge_model"]["served_model_name"]
    prompts = {"V1": OOC_JUDGE_PROMPT, "V2": OOC_JUDGE_PROMPT_V2}

    jobs = [(v, i, r) for v in prompts for i in ids for r in range(args.repeats)]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda j: judge_once(client, judge_model, prompts[j[0]], cards[j[1]], sheet[j[1]]["reply"]), jobs))

    flag_rate = {v: {} for v in prompts}
    failed = 0
    for v in prompts:
        for i in ids:
            votes = [x for (jv, ji, _), x in zip(jobs, results) if jv == v and ji == i and x is not None]
            failed += args.repeats - len(votes)
            flag_rate[v][i] = sum(votes) / len(votes) if votes else 0.0
    print(f"{len(jobs)} 次判定，解析失败 {failed} 次（按未标记计）")

    def report(title, subset):
        print(f"\n== {title}（n={len(subset)}，人工标为出戏 {sum(human[i] for i in subset)} 条；每条 {args.repeats} 次取期望）==")
        for v in prompts:
            c = confusion(flag_rate[v], human, subset)
            print(f"{v}: TP {c['tp']:.1f}  FP {c['fp']:.1f}  FN {c['fn']:.1f}  TN {c['tn']:.1f}  "
                  f"精确率 {c['precision']:.0%}  召回率 {c['recall']:.0%}")

    report("全部 40 条", ids)
    report("去掉有争议的标注 (item 1/17/23/35)", [i for i in ids if i not in BORDERLINE])
    orig = {i: int(key[i]["judge_ooc"]) for i in ids}
    print(f"\n参照：主实验当时那次判定（key.csv）对着这 40 条：精确率 "
          f"{sum(1 for i in ids if orig[i] and human[i]) / max(sum(orig.values()), 1):.0%}")

    print("\n判定变化明显的条目（|V2 标记率 − V1 标记率| ≥ 0.4）：")
    print("item  人工  V1标记率  V2标记率  方案")
    for i in ids:
        if abs(flag_rate["V2"][i] - flag_rate["V1"][i]) >= 0.4:
            print(f"{i:>4}  {human[i]:>4}  {flag_rate['V1'][i]:>8.1f}  {flag_rate['V2'][i]:>8.1f}  {key[i]['scheme']}")

    print("\nV2 仍然判错的条目（标记率与人工差 ≥ 0.5）：")
    for i in ids:
        if abs(flag_rate["V2"][i] - human[i]) >= 0.5:
            print(f"  item {i}: 人工 {human[i]}，V2 标记率 {flag_rate['V2'][i]:.1f}")

    out = Path(args.out) if args.out else d / f"regress_per_item_{args.tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "human_ooc", "v1_flag_rate", "v2_flag_rate", "scheme", "dialogue_id"])
        for i in ids:
            w.writerow([i, human[i], flag_rate["V1"][i], flag_rate["V2"][i], key[i]["scheme"], key[i]["dialogue_id"]])
    print(f"judge tag = {args.tag}")
    print(f"\n逐条结果写到 {out}")
    print("提醒：V2 是看着这 40 条的错误写的，上面的提升是样本内的；留出集检验是下一步的分层审核。")


if __name__ == "__main__":
    main()
