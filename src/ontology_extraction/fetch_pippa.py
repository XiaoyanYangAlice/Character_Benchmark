#!/usr/bin/env python3
"""
fetch_pippa.py

从 HuggingFace 拉 PIPPA 数据集（PygmalionAI/pippa），按轮数筛选，把每个角色的
persona 描述（bot_description 字段）写成一个 .txt，供 prepare_stories.py 和
角色本体抽取器使用。

PIPPA 是 gated dataset，第一次用需要：
1. 浏览器打开 https://huggingface.co/datasets/PygmalionAI/pippa，登录后点击
   接受使用条款；
2. 本地/登录节点跑 `hf auth login`，粘贴你的 HF token（在
   https://huggingface.co/settings/tokens 生成，Read 权限即可）。

字段名以实际下载到的数据为准——这份脚本第一次跑请先用 --inspect 看一眼真实字段，
不要直接相信下面写死的字段名（HF 上数据集字段有改名的先例）。

用法：
    # 第一次：先看字段长什么样
    python fetch_pippa.py --inspect

    # 确认字段名后正式跑
    python fetch_pippa.py --output-dir ../../data/raw_stories --min-turns 15 --limit 70
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def inspect(ds):
    print("Dataset features:")
    print(ds.features)
    print("\n第一条样本（截断显示）：")
    row = ds[0]
    for k, v in row.items():
        preview = str(v)
        if len(preview) > 300:
            preview = preview[:300] + "...(截断)"
        print(f"  {k}: {preview}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("../../data/raw_stories"))
    ap.add_argument("--min-turns", type=int, default=15)
    ap.add_argument("--limit", type=int, default=70, help="最多导出多少个角色")
    ap.add_argument("--inspect", action="store_true", help="只打印字段结构，不写文件")
    ap.add_argument("--description-field", default="bot_description",
                     help="persona 描述字段名，--inspect 看到的实际字段名和这不一致时用这个参数改")
    ap.add_argument("--conversation-field", default="conversation",
                     help="对话轮次字段名，用于按轮数筛选")
    args = ap.parse_args()

    print("下载/加载 PIPPA（第一次会比较慢，之后走本地缓存）...")
    ds = load_dataset("PygmalionAI/pippa", data_files="pippa.jsonl", split="train")

    if args.inspect:
        inspect(ds)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen_bots = set()
    n_written = 0

    for row in ds:
        if n_written >= args.limit:
            break
        try:
            conv = row[args.conversation_field]
            desc = row[args.description_field]
        except KeyError as e:
            raise SystemExit(
                f"字段名不对：{e}。先跑 `python fetch_pippa.py --inspect` 看真实字段名，"
                f"再用 --description-field / --conversation-field 指定。"
            )

        n_turns = len(conv) if hasattr(conv, "__len__") else 0
        if n_turns < args.min_turns:
            continue

        bot_id = row.get("bot_id", f"bot_{n_written}")
        if bot_id in seen_bots or not desc or not desc.strip():
            continue
        seen_bots.add(bot_id)

        out_path = args.output_dir / f"pippa_{bot_id}.txt"
        out_path.write_text(desc.strip(), encoding="utf-8")
        n_written += 1

    print(f"写出 {n_written} 个角色描述到 {args.output_dir}")
    print("下一步：python prepare_stories.py --input-dir ../../data/raw_stories "
          "--output ../../data/processed/stories.jsonl")


if __name__ == "__main__":
    main()
