#!/usr/bin/env python3
"""
make_probes.py —— 给每条对话生成「记忆探针」，写回 dialogues.jsonl 的
`memory_probes` 字段。

## 为什么需要探针

v1 的两个指标（ooc_rate、fact_consistency）都只测"像不像这个角色"，不测
"记不记得发生过什么"。一个完全没有记忆的模型，只要 system prompt 里的角色卡
够详细，这两个指标都能拿高分——v1 实验里 Llama 的 fact_consistency 就是
1.000。换句话说，v1 没有任何指标在直接测记忆，而这个项目的题目恰恰是记忆。

探针直接测：用户在对话早期说过一个具体信息（名字、数字、地点、承诺），对话
后段再问一次"我刚才跟你说的那个 X 是什么"，看角色答不答得出来。

## 关键设计：题目必须来自用户发言，不能来自角色回复

`scripted_user_turns` 在所有记忆方案之间是完全相同的，所以从用户发言里出的题
对所有方案是同一套题，可比。如果从角色回复里出题，每个方案的回复不一样，题目
就不一样，那就没法比了（而且会变成"考它自己刚编的东西"）。

代价：用户发言里不一定有足够多的可考信息，所以每条对话的探针数量不固定，
生成不出来就是空列表，run_eval 对这条对话跳过探针指标（不是算 0 分）。

## 用法（需要 judge 的 vLLM 服务已起好，见 slurm/make_probes.slurm）

    python src/generate/make_probes.py --config configs/experiment.yaml \\
        --dialogues data/processed/dialogues.jsonl --judge-port 8002 \\
        --n-probes 3 --output data/processed/dialogues.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROBE_GEN_PROMPT = """Below are the first turns a user said to a role-play character.

Your job: find up to {n} pieces of CONCRETE information that the USER stated about
themselves or their situation, and turn each into a short question that tests whether
the character remembers it later.

Rules:
- Only use information the user explicitly stated. Never invent or infer.
- Pick concrete, checkable things: names, numbers, places, relationships, stated plans
  or preferences. Skip anything vague, emotional, or hypothetical.
- The question must be answerable from the user's own words alone.
- Write each question the way the same user would ask it later, in first person,
  addressing the character directly.
- The expected answer must be short (a few words), and must be exactly what the user said.
- CRITICAL: the question must NOT contain the answer. Bad: "Are you going back to Liyue?"
  with answer "going back to Liyue" — the answer is already in the question, so it tests
  nothing. Good: "Where did I say I was heading?" with answer "Liyue".
- Do NOT write yes/no questions (anything starting with Are/Is/Do/Did/Will/Can/Have).
  Ask what, where, who, which, how many, or whose.
- If the user said nothing concrete enough, return an empty list. An empty list is a
  perfectly acceptable answer; do not force it.

USER TURNS (numbered):
{turns}

Return a JSON OBJECT only (not a bare array), in exactly this form:
{{"probes": [{{"source_turn_index": <the number of the user turn it came from>, "question": "<question in the user's voice>", "expected_answer": "<a few words>"}}]}}
"""


def build_prompt(user_turns: list[str], n: int, from_first: int) -> str:
    numbered = "\n".join(f"{i}. {t[:400]}" for i, t in enumerate(user_turns[:from_first], 1))
    return PROBE_GEN_PROMPT.format(n=n, turns=numbered)


# 以这些词开头的是是非题，猜也有一半概率对，不能用来测记忆。
_YES_NO_STARTERS = {"are", "is", "was", "were", "do", "does", "did", "will", "would",
                     "can", "could", "have", "has", "had", "should", "am", "shall"}


def answer_leaks_into_question(question: str, expected: str) -> bool:
    """答案的实词是不是大部分已经出现在问题里了。

    实测第一次生成的 65 个探针里，多数是这种退化形态：
      Q: "Are you going back to Liyue?"  A: "going back to Liyue"
      Q: "Do we need to ask someone to help us?"  A: "someone to help us"
    模型把用户原话截一段改成疑问句，"标准答案"就是那一段本身。这种题任何
    模型都能答对（照抄问题即可），完全不测记忆，留在数据里会把所有方案的
    probe_accuracy 一起抬高、抹平方案之间的差异。
    """
    from src.memory.structured_memory import content_tokens
    a = content_tokens(expected)
    if not a:
        return True
    q = content_tokens(question)
    return len(a & q) / len(a) > 0.5


def validate_probe(probe: dict, user_turns: list[str], from_first: int) -> bool:
    """只保留"答案确实出现在它声称的那条用户发言里"的探针。

    这一步不能省。模型很容易生成一个看起来合理、但用户其实没说过的答案
    （比如把"我妹妹"写成"我妹妹 Anna"），那种题目无论角色怎么答都是错的，
    会系统性压低所有方案的探针得分、还可能压低得不均匀。这里用的是朴素的
    子串包含检查（忽略大小写和空白），宁可漏掉一些合法探针，也不要放进来
    一道错题。
    """
    if not isinstance(probe, dict):
        return False
    q = str(probe.get("question", "")).strip()
    a = str(probe.get("expected_answer", "")).strip()
    if not q or not a or len(a) > 120:
        return False
    # 答案不能已经写在问题里（见 answer_leaks_into_question 的说明）
    if answer_leaks_into_question(q, a):
        return False
    # 是非题猜对的概率太高，不要
    first_word = q.strip().strip("\"'").split()[0].lower().rstrip(",") if q.strip() else ""
    if first_word in _YES_NO_STARTERS:
        return False
    try:
        idx = int(probe.get("source_turn_index"))
    except (TypeError, ValueError):
        return False
    if not 1 <= idx <= min(from_first, len(user_turns)):
        return False
    source = user_turns[idx - 1].lower()
    a_norm = " ".join(a.lower().split())
    # 整体包含，或者（答案是多词短语时）每个词都出现在这条发言里
    if a_norm in " ".join(source.split()):
        return True
    words = [w for w in a_norm.split() if len(w) > 2]
    return bool(words) and all(w in source for w in words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge-port", default="8002")
    ap.add_argument("--base-url-template", default="http://localhost:{port}/v1")
    ap.add_argument("--n-probes", type=int, default=3,
                     help="每条对话最多保留几个探针。校验很严（答案不能写在问题里、"
                          "不能是是非题），所以让模型多生成一些再筛，见 --n-candidates")
    ap.add_argument("--n-candidates", type=int, default=8,
                     help="让模型一次生成多少候选，再用 validate_probe 筛掉不合格的。"
                          "候选数大于 n-probes，是因为实测合格率只有三成左右。")
    ap.add_argument("--from-first", type=int, default=10,
                     help="只从前这么多轮用户发言里出题，保证探针考的是'早期信息'，"
                          "而不是刚刚才说过的话（那样 raw_context 的窗口里还在，考不出记忆）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import yaml
    from openai import OpenAI

    from src.eval.metrics import as_list, call_judge

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = config["judge_model"]["served_model_name"]
    client = OpenAI(base_url=args.base_url_template.format(port=args.judge_port), api_key="EMPTY")

    rows = [json.loads(l) for l in Path(args.dialogues).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"读到 {len(rows)} 条对话，为每条生成最多 {args.n_probes} 个探针"
          f"（只从前 {args.from_first} 轮用户发言出题）")

    n_failed = 0

    def one(row: dict) -> dict:
        """单条对话生成探针。**任何异常都在这里吃掉**：一条坏响应不应该让
        整个作业退出。第一次跑这个脚本时就是因为没有这层保护，一条返回了
        裸数组的响应让 ThreadPoolExecutor.map 抛异常，47 条全部白跑。"""
        nonlocal n_failed
        try:
            return _one_inner(row)
        except Exception as e:
            n_failed += 1
            print(f"  [跳过 {row.get('dialogue_id', '?')}] 探针生成失败：{type(e).__name__}: {e}")
            row["memory_probes"] = []
            return row

    def _one_inner(row: dict) -> dict:
        turns = row.get("scripted_user_turns", [])
        if len(turns) < args.from_first:
            # 对话比探针取材范围还短：出题范围就缩到全部用户发言。
            from_first = len(turns)
        else:
            from_first = args.from_first
        if from_first < 2:
            row["memory_probes"] = []
            return row
        # 温度 0：探针是数据集的一部分，必须可复现。
        result = call_judge(client, model, build_prompt(turns, args.n_candidates, from_first),
                            temperature=0.0, max_tokens=768)
        # as_list 容忍模型直接返回裸数组 [...] 而不是 {"probes": [...]}
        probes = as_list(result, "probes")
        # 先全部校验再截断，不是先截断再校验——否则前几个不合格就白白浪费了候选
        kept = [p for p in probes if validate_probe(p, turns, from_first)][: args.n_probes]
        row["memory_probes"] = kept
        return row

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, rows))

    if n_failed:
        print(f"⚠ {n_failed}/{len(rows)} 条对话的探针生成调用失败，已按"
              f"「无探针」处理（不是算 0 分）。失败比例高的话看上面的报错原因。")
    n_probes = sum(len(r["memory_probes"]) for r in rows)
    n_with = sum(1 for r in rows if r["memory_probes"])
    Path(args.output).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"写出 {args.output}：{n_probes} 个探针，覆盖 {n_with}/{len(rows)} 条对话"
          f"（平均 {n_probes / max(len(rows), 1):.1f} 个/条）")
    if n_with < len(rows) * 0.5:
        print("⚠ 超过一半的对话没能生成通过校验的探针。先抽几条看看 scripted_user_turns "
              "里到底有没有可考的具体信息；如果确实没有，就在报告里说明探针只覆盖了部分对话，"
              "不要为了凑数放宽 validate_probe 的校验。")
    print("注意：探针覆盖率低不是 bug，是 PIPPA 用户发言本身信息密度低。报告里按"
          "'有探针的对话'这个子集报探针指标，并写清子集大小。")


if __name__ == "__main__":
    main()
