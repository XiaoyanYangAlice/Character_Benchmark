"""抽样审核 OOC judge：从 qwen 的生成结果里随机抽一批角色回复，用跟主实验同一个
judge 配置（同一个 prompt、temperature、judge_repeats=1）重新判一遍，然后抽出
"被判出戏"和"没被判出戏"各 N 条，打乱顺序后导出一份**盲评清单**（不显示 judge 的
判定），由人工逐条判断真的出不出戏；答案另存 key.csv，评分用
src/eval/ooc_audit_score.py。

为什么重判而不是读已有结果：主实验只存了每条对话的汇总 ooc_rate，没有存逐轮的
判定；而 judge 只看"角色设定 + 这一句回复"，跟对话上下文无关，所以对已有的
raw_outputs 重判是可行的。judge 温度 0.3 有随机性，重判结果不会跟当时那次逐轮
一致——这里审核的是"这个 judge 判得准不准"，不是复原某一次具体的判定。

为什么要抽"没被判出戏"的对照：只看被标出戏的，只能算精确率；真正出戏却没被
标出来的（漏判）只有从未标记的里面抽才估得出来。

需要 judge 的 vLLM 服务已经起好（见 slurm/ooc_audit.slurm）。用法：
    python src/eval/ooc_audit_sample.py --config configs/experiment.yaml \\
        --judge-port 8002 --raw-dirs results/raw_outputs results_seed1/raw_outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RUBRIC = """判断准则（出戏 = 符合以下任一条）：
1. 角色自称/暗示自己是 AI、语言模型、聊天机器人，或说"作为一个AI我不能..."这类元话语；
2. 语气、用词风格与角色设定明显不符，且没有情节上的合理转折；
3. 说出与设定的知识范围、时代背景明显冲突的内容；
4. 纯粹的助手式服务用语（"我可以帮你查一下"、"以下是几点建议"之类列点体），脱离角色扮演口吻；
5. 角色"知道"了它不该知道的信息（没有合理的剧情内信息来源）。

不算出戏：
- 角色性格本来就直率/毒舌/温柔，用符合设定的方式表达强烈情绪或拒绝；
- 角色做出符合设定但你不喜欢的选择（喜好不归你判）；
- 使用了设定里明确允许的能力/知识。
"""


def collect_turns(raw_dirs: list[str], model: str, dialogues: dict) -> list[dict]:
    """收集 model 的所有角色轮次。文件名格式 {model}__{scheme}__{dialogue_id}.json。"""
    items = []
    for d in raw_dirs:
        for path in sorted(Path(d).glob(f"{model}__*.json")):
            _, scheme, dialogue_id = path.stem.split("__", 2)
            if dialogue_id not in dialogues:
                continue
            turns = json.loads(path.read_text(encoding="utf-8"))
            for i, t in enumerate(turns):
                if t.get("speaker") == "user":
                    continue
                prev_user = turns[i - 1]["text"] if i > 0 and turns[i - 1].get("speaker") == "user" else ""
                items.append({
                    "source": f"{Path(d).parent.name}/{path.name}",
                    "scheme": scheme, "dialogue_id": dialogue_id,
                    "turn_id": t.get("turn_id"), "text": t["text"], "prev_user": prev_user,
                })
    return items


def judge_one(client, judge_model: str, item: dict, dialogues: dict) -> dict:
    from src.eval.judge_prompts import OOC_JUDGE_PROMPT
    from src.eval.metrics import call_judge
    from src.eval.run_eval import character_card_text

    card = dialogues[item["dialogue_id"]]["character_card"]
    prompt = OOC_JUDGE_PROMPT.format(character_card=character_card_text(card), turn_text=item["text"])
    result = call_judge(client, judge_model, prompt)
    if result is None or "ooc" not in result:
        return {**item, "judge_ooc": None, "judge_reason": ""}
    return {**item, "judge_ooc": bool(result["ooc"]), "judge_reason": str(result.get("reason", ""))}


def write_outputs(out_dir: Path, chosen: list[dict], dialogues: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "judge_ooc", "judge_reason", "scheme", "dialogue_id", "turn_id"])
        for n, it in enumerate(chosen, 1):
            w.writerow([n, int(it["judge_ooc"]), it["judge_reason"], it["scheme"], it["dialogue_id"], it["turn_id"]])
    with open(out_dir / "labels_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "human_ooc", "note"])
        for n in range(1, len(chosen) + 1):
            w.writerow([n, "", ""])
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# OOC judge 人工审核清单（盲评）", "",
             "逐条判断下面每个「角色回复」是否出戏，在 labels_template.csv 里对应 item_id 一行的 "
             "human_ooc 列填 1（出戏）或 0（没出戏）。**请先全部判完再看 key.csv。**", "",
             RUBRIC,
             "注：judge 只看到「角色设定 + 角色回复」。「用户上一句」仅供你参考理解语境，judge 没看到。", "", "---", ""]
    for n, it in enumerate(chosen, 1):
        card = dialogues[it["dialogue_id"]]["character_card"]
        name = card.get("display_name") or card.get("name", "")
        desc = (card.get("description") or "").strip().replace("\n", " ")
        lines += [f"## item {n}", f"**角色**：{name}", f"**角色设定**：{desc[:700]}{'…' if len(desc) > 700 else ''}",
                  f"**用户上一句（仅参考）**：{it['prev_user'] or '（无）'}", "",
                  f"**角色回复**：{it['text']}", "", "---", ""]
    (out_dir / "sheet.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--judge-port", default="8002")
    ap.add_argument("--base-url-template", default="http://localhost:{port}/v1")
    ap.add_argument("--model", default="qwen2.5-7b", help="审核哪个 agent 模型的回复")
    ap.add_argument("--raw-dirs", nargs="+", default=["results/raw_outputs"])
    ap.add_argument("--n-sample", type=int, default=600, help="先随机抽这么多轮重判，再从中取标记/未标记各 N 条")
    ap.add_argument("--n-flagged", type=int, default=20)
    ap.add_argument("--n-unflagged", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="results/ooc_audit")
    args = ap.parse_args()

    import yaml
    from src.eval.run_eval import load_dialogues

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dialogues = {d["dialogue_id"]: d for d in load_dialogues(config["data"]["dialogues_path"])}
    items = collect_turns(args.raw_dirs, args.model, dialogues)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n_sample]
    print(f"{args.model} 共抽 {len(items)} 个角色轮次重判")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url_template.format(port=args.judge_port), api_key="EMPTY")
    judge_model = config["judge_model"]["served_model_name"]
    with ThreadPoolExecutor(args.workers) as ex:
        judged = list(ex.map(lambda it: judge_one(client, judge_model, it, dialogues), items))

    valid = [j for j in judged if j["judge_ooc"] is not None]
    flagged = [j for j in valid if j["judge_ooc"]]
    unflagged = [j for j in valid if not j["judge_ooc"]]
    print(f"有效判定 {len(valid)}，其中被判出戏 {len(flagged)}（{len(flagged) / max(len(valid), 1):.1%}），"
          f"判定失败 {len(judged) - len(valid)}")
    if len(flagged) < args.n_flagged:
        print(f"警告：被标出戏的只有 {len(flagged)} 条，少于要求的 {args.n_flagged}；"
              f"可以调大 --n-sample 重跑。")
    chosen = flagged[: args.n_flagged] + unflagged[: args.n_unflagged]
    rng.shuffle(chosen)

    meta = {"model": args.model, "n_judged": len(valid), "n_flagged_total": len(flagged),
            "n_flagged_chosen": min(len(flagged), args.n_flagged),
            "n_unflagged_chosen": min(len(unflagged), args.n_unflagged), "seed": args.seed}
    write_outputs(Path(args.out_dir), chosen, dialogues, meta)
    print(f"已写到 {args.out_dir}/：sheet.md（给你读）、labels_template.csv（你填）、"
          f"key.csv（答案，判完再看）、meta.json")


if __name__ == "__main__":
    main()
