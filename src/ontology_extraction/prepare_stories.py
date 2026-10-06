#!/usr/bin/env python3
"""
prepare_stories.py

把原始故事/长对话文本切成"编号句子"格式，喂给 biography/emotion/personality/
physical 四个抽取 prompt（这些 prompt 都要求输入形如 [1] xxx。[2] xxx。）。

用法：
    python prepare_stories.py --input-dir raw_stories/ --output stories.jsonl

输入目录下每个 .txt 文件是一段完整故事或一段角色扮演长对话（建议把多轮对话按
"说话人：内容" 的格式拼成连续叙述文本，或者直接保留对话形式——句子切分对两种
输入都适用，只是抽取出来的 character 需要对应到你的角色卡）。

输出 JSONL，每行一个故事：
{"story_id": "xxx", "source_file": "xxx.txt", "sentences": [{"id": 1, "text": "..."}, ...]}
"""
import argparse
import json
import re
from pathlib import Path

# 简单的中英文句子切分：按中文句号/问号/感叹号/分号，或英文 . ! ? 后跟空格/换行切分。
# 对话场景里如果有明显的换行分隔（每行一句/一轮），优先按换行切，避免把对话切碎。
SENT_SPLIT_ZH = re.compile(r"(?<=[。！？；])\s*")
SENT_SPLIT_EN = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sentences = []
    for line in lines:
        # 粗略判断中英文占比，选择对应的切分规则；也可以两种规则都跑一遍取更细的切分
        zh_chars = len(re.findall(r"[一-鿿]", line))
        if zh_chars > len(line) * 0.3:
            parts = [p for p in SENT_SPLIT_ZH.split(line) if p.strip()]
        else:
            parts = [p for p in SENT_SPLIT_EN.split(line) if p.strip()]
        sentences.extend(parts if parts else [line])
    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-sentences", type=int, default=5,
                     help="少于这个句子数的故事直接跳过（太短没有抽取价值）")
    args = ap.parse_args()

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for f in sorted(args.input_dir.glob("*.txt")):
            text = f.read_text(encoding="utf-8")
            sents = split_sentences(text)
            if len(sents) < args.min_sentences:
                print(f"[skip] {f.name}: only {len(sents)} sentences")
                continue
            record = {
                "story_id": f.stem,
                "source_file": f.name,
                "sentences": [{"id": i + 1, "text": s} for i, s in enumerate(sents)],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} stories to {args.output}")


if __name__ == "__main__":
    main()
