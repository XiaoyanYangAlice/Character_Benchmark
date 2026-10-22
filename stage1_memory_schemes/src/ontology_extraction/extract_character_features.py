#!/usr/bin/env python3
"""
extract_character_features.py

对 prepare_stories.py 产出的 stories.jsonl 批量跑 biography / emotion /
personality / physical 四类角色特征抽取，调用本地 vLLM 起的 OpenAI 兼容服务。

每个故事 × 每个类别 × 一种语言版本 = 一次调用。四类特征加起来就是 GOLEM
本体里 G17_Character_Feature 的四个子类；每条抽取额外记录 evidence（story_id +
sentence_id + 原文 span），对应本体里 E13_Attribute_Assignment 的证据溯源思路——
这一点直接决定了后面"结构化事件记忆"方案的"证据可追溯率"指标怎么算。

用法（先冒烟测试，--limit 2 只跑前两个故事）：
    python extract_character_features.py \
        --stories stories.jsonl \
        --prompts-dir ../prompts \
        --lang zh \
        --api-base http://localhost:8000/v1 \
        --model agent-model \
        --output character_features.jsonl \
        --limit 2

跑通后去掉 --limit 放量。
"""
import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

CATEGORIES = ["biography", "emotion", "personality", "physical"]


def load_prompt(prompts_dir: Path, category: str, lang: str) -> str:
    version = "v2" if lang == "zh" else "v4"
    path = prompts_dir / f"{category}_{version}_{lang}.txt"
    return path.read_text(encoding="utf-8")


def render_story(sentences: list[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


def extract_json(raw: str) -> dict | None:
    """模型偶尔还是会包一层 markdown 代码块，这里做个宽松兜底解析。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 尝试截取第一个 { 到最后一个 } 之间的内容
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def call_model(client: OpenAI, model: str, prompt_template: str, story_text: str,
                max_retries: int = 3, temperature: float = 0.0) -> dict | None:
    prompt = prompt_template.replace("{story_sentences}", story_text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content
            parsed = extract_json(content)
            if parsed is not None and "extractions" in parsed:
                return parsed
            print(f"  [retry {attempt+1}] bad JSON, retrying...")
        except Exception as e:
            print(f"  [retry {attempt+1}] API error: {e}")
        time.sleep(2)
    return None


def verify_span(extraction: dict, sentences_by_id: dict) -> bool:
    """核对 prompt 里要求的自检：span 必须逐字出现在对应 sentence_id 的句子里。"""
    sid = extraction.get("sentence_id")
    span = extraction.get("span", "")
    sent = sentences_by_id.get(sid, "")
    return bool(span) and span in sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True, type=Path)
    ap.add_argument("--prompts-dir", required=True, type=Path)
    ap.add_argument("--lang", choices=["zh", "en"], default="zh")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="agent-model")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个故事，冒烟测试用")
    args = ap.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    prompts = {cat: load_prompt(args.prompts_dir, cat, args.lang) for cat in CATEGORIES}

    stories = [json.loads(l) for l in args.stories.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        stories = stories[: args.limit]

    n_ok, n_fail, n_span_fail = 0, 0, 0

    with open(args.output, "w", encoding="utf-8") as out:
        for story in stories:
            story_id = story["story_id"]
            sentences_by_id = {s["id"]: s["text"] for s in story["sentences"]}
            story_text = render_story(story["sentences"])
            print(f"[story] {story_id} ({len(story['sentences'])} sentences)")

            for category in CATEGORIES:
                result = call_model(client, args.model, prompts[category], story_text)
                if result is None:
                    print(f"  [fail] {category}: no valid JSON after retries")
                    n_fail += 1
                    continue
                for ext in result.get("extractions", []):
                    ok = verify_span(ext, sentences_by_id)
                    if not ok:
                        n_span_fail += 1
                        continue
                    record = {
                        "story_id": story_id,
                        "feature_type": category,  # biography / emotion / personality / physical
                        "character": ext.get("character"),
                        "span": ext.get("span"),
                        "sentence_id": ext.get("sentence_id"),
                        "modality": ext.get("modality"),
                        "lang": args.lang,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_ok += 1

    print(f"\nDone. extractions written={n_ok}, span-check failed(dropped)={n_span_fail}, "
          f"category calls failed={n_fail}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
