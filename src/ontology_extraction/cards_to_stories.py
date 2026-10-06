#!/usr/bin/env python3
"""
cards_to_stories.py

把 SillyTavern character card v2 格式（JSON，从 Chub.ai 等站点导出）转成
prepare_stories.py 能吃的 .txt。

获取角色卡的方式（人工步骤，脚本本身不联网下载，避免踩到各站点的反爬限制）：
1. 打开 https://chub.ai ，找想要的角色卡；
2. 角色页面一般有 "Download" / 导出 JSON 的按钮，存到本地一个文件夹，比如
   `data/raw_cards/`；
3. 逐个确认角色卡的使用许可（有的作者标了仅供个人使用，不允许用于训练/研究，
   跳过那些）；
4. 跑这个脚本转换格式。

character card v2 的核心字段一般是：
    {"data": {"name": ..., "description": ..., "personality": ...,
              "scenario": ..., "first_mes": ..., "mes_example": ...}}
字段名以你实际下载到的卡片为准，脚本里做了宽松兜底（v1/v2 两种顶层结构都尝试）。

用法：
    python cards_to_stories.py --input-dir ../../data/raw_cards \
        --output-dir ../../data/raw_stories
"""
import argparse
import json
from pathlib import Path


def extract_fields(card: dict) -> dict:
    """兼容 v1（字段在顶层）和 v2（字段在 data 里）两种格式。"""
    data = card.get("data", card)
    return {
        "name": data.get("name", "unknown"),
        "description": data.get("description", ""),
        "personality": data.get("personality", ""),
        "scenario": data.get("scenario", ""),
        "first_mes": data.get("first_mes", ""),
    }


def card_to_narrative(fields: dict) -> str:
    """把结构化字段拼成接近叙事文本的形式，方便抽取器按句子处理。
    这不是完美的叙事改写，只是把字段内容顺序拼接——抽取质量在冒烟测试阶段要
    重点检查这批数据（结构化字段本身信息密度高，但不一定是完整句子）。"""
    parts = []
    name = fields["name"]
    if fields["description"]:
        parts.append(fields["description"])
    if fields["personality"]:
        parts.append(f"{name}的性格：{fields['personality']}")
    if fields["scenario"]:
        parts.append(f"故事背景：{fields['scenario']}")
    if fields["first_mes"]:
        parts.append(fields["first_mes"])
    return "\n".join(p.strip() for p in parts if p.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path,
                     help="存放角色卡 .json 文件的目录")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for f in sorted(args.input_dir.glob("*.json")):
        try:
            card = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[skip] {f.name}: 不是合法 JSON")
            continue

        fields = extract_fields(card)
        text = card_to_narrative(fields)
        if not text.strip():
            print(f"[skip] {f.name}: 没提取到任何文本，检查字段名是否对得上")
            continue

        out_path = args.output_dir / f"card_{f.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        n_written += 1

    print(f"写出 {n_written} 个角色卡文本到 {args.output_dir}")


if __name__ == "__main__":
    main()
