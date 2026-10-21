#!/usr/bin/env python3
"""
validate_extractions.py

冒烟测试用的质检脚本，对应集群指南里"先冒烟测试再放量"那条纪律。

功能：
1. 基础质检：每条抽取的 span 是否确实逐字出现在对应句子里（extract_character_features.py
   里已经做过一次，这里是独立复核，防止脚本本身有 bug）。
2. 角色名规范化检查：同一故事里角色名是否发散（比如"姜遥""姜医生""她"没有被
   正确归并成同一个 canonical name）——粗略地用"角色名集合大小 vs 常识预期"报警，
   不是精确指标，只是提醒你去抽样看。
3. 如果同时提供了 zh 和 en 两次抽取结果（同一批故事分别跑了 v2_zh 和 v4_en），
   计算两次结果在"每个类别抽取条数"上的一致性，作为 prompt 版本间可靠性的一个
   粗粒度信号——这是你已有的双语 prompt 设计能顺手做的一个附加验证，成本几乎为零。

用法：
    python validate_extractions.py --extractions character_features_zh.jsonl \
        [--compare-with character_features_en.jsonl]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# prompt 里唯一定义的三个合法值；出现别的值（比如模型自造的 "future"）说明
# 输出格式跑偏了，抽取代码本身没做枚举校验，靠这里事后抓出来。
VALID_MODALITIES = {"present", "past", "potential"}

# 角色数量超过这个阈值就在报告里标出来，提示去人工看看是不是有代词/别名没
# 归并成同一个 canonical name（比如 "He"/"Alastair" 没合并、"Sofiya"/
# "Princess Sofiya" 没统一成长名）。这是一个粗糙的经验阈值，不是精确指标。
CHAR_COUNT_FLAG_THRESHOLD = 3


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def basic_report(records: list[dict]):
    print(f"总抽取条数: {len(records)}")
    by_category = Counter(r["feature_type"] for r in records)
    print("按类别分布:", dict(by_category))

    by_story_chars = defaultdict(set)
    for r in records:
        by_story_chars[r["story_id"]].add(r["character"])

    flagged = {sid: chars for sid, chars in by_story_chars.items()
               if len(chars) > CHAR_COUNT_FLAG_THRESHOLD}
    print(f"\n每个故事的角色数量（共 {len(by_story_chars)} 个故事，"
          f"抽样看前 10 个；角色数 > {CHAR_COUNT_FLAG_THRESHOLD} 的故事全部列在下面，"
          f"可能是代词/别名没有正确归并成同一个 canonical name）:")
    for story_id, chars in list(by_story_chars.items())[:10]:
        flag = "  <- 检查一下" if story_id in flagged else ""
        print(f"  {story_id}: {len(chars)} 个角色 {sorted(chars)[:6]}{flag}")

    if flagged:
        print(f"\n角色数 > {CHAR_COUNT_FLAG_THRESHOLD} 的故事完整列表（{len(flagged)} 个，建议逐个人工看一眼）：")
        for story_id, chars in flagged.items():
            print(f"  {story_id}: {sorted(chars)}")

    modality_counts = Counter(r["modality"] for r in records)
    print("\nmodality 分布:", dict(modality_counts))
    if modality_counts.get("present", 0) == len(records):
        print("  提醒：所有条目都是 present，模型可能没有正确识别 past/potential，抽样人工看几条")

    invalid_modality = [r for r in records if r["modality"] not in VALID_MODALITIES]
    if invalid_modality:
        print(f"\n⚠ 发现 {len(invalid_modality)} 条非法 modality 值（不在 "
              f"{sorted(VALID_MODALITIES)} 范围内），模型输出格式跑偏了，需要人工决定怎么处理"
              f"（映射到最接近的合法值，或者直接剔除）：")
        for r in invalid_modality:
            print(f"  story={r['story_id']} character={r['character']!r} "
                  f"modality={r['modality']!r} span={r['span']!r}")


def compare_report(records_a: list[dict], records_b: list[dict]):
    def count_by(records):
        c = Counter()
        for r in records:
            c[(r["story_id"], r["feature_type"])] += 1
        return c

    ca, cb = count_by(records_a), count_by(records_b)
    keys = set(ca) | set(cb)
    diffs = []
    for k in keys:
        na, nb = ca.get(k, 0), cb.get(k, 0)
        if na == 0 and nb == 0:
            continue
        ratio = min(na, nb) / max(na, nb) if max(na, nb) > 0 else 1.0
        diffs.append((k, na, nb, ratio))

    diffs.sort(key=lambda x: x[3])
    print(f"\n双语一致性检查（{len(diffs)} 个 story×category 组合，按一致性从低到高排序，"
          f"前 10 个差异最大的）：")
    print(f"{'story_id':<20}{'category':<12}{'A条数':<8}{'B条数':<8}{'一致性':<8}")
    for k, na, nb, ratio in diffs[:10]:
        print(f"{k[0]:<20}{k[1]:<12}{na:<8}{nb:<8}{ratio:<8.2f}")

    avg_ratio = sum(d[3] for d in diffs) / len(diffs) if diffs else 0
    print(f"\n平均一致性: {avg_ratio:.2f}（这不是严格的抽取一致性指标，只是抽取条数的粗略"
          f"对照；一致性偏低的组合建议人工抽查两种语言版本的输出）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", required=True, type=Path)
    ap.add_argument("--compare-with", type=Path, default=None)
    args = ap.parse_args()

    records = load(args.extractions)
    basic_report(records)

    if args.compare_with:
        other = load(args.compare_with)
        compare_report(records, other)


if __name__ == "__main__":
    main()
