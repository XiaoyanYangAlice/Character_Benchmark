"""OOC judge prompt 的格式回归：V1/V2 都必须能 .format(character_card, turn_text)，
角色卡里带花括号（比如 PIPPA 的 {{user}}）不能让 format 炸掉，V2 的输出格式里 reason 在 ooc 前。"""
import tempfile
from pathlib import Path

from src.eval.judge_prompts import OOC_JUDGE_PROMPT, OOC_JUDGE_PROMPT_V2
from src.eval.ooc_judge_regress import confusion, parse_sheet


def test_both_ooc_prompts_format_with_braces_in_card():
    for tpl in (OOC_JUDGE_PROMPT, OOC_JUDGE_PROMPT_V2):
        out = tpl.format(character_card="Narrator ... {{user}} ... {x}", turn_text="回复 {y}")
        assert "{{user}}" in out and "回复 {y}" in out


def test_v2_prompt_json_schema_puts_reason_before_verdict():
    out = OOC_JUDGE_PROMPT_V2.format(character_card="c", turn_text="t")
    schema = out.strip().splitlines()[-1]
    assert schema.index('"reason"') < schema.index('"ooc"')


def test_confusion_counts_expected_values():
    human = {1: 1, 2: 1, 3: 0, 4: 0}
    c = confusion({1: 1.0, 2: 0.0, 3: 0.5, 4: 0.0}, human, [1, 2, 3, 4])
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1.0, 1.0, 0.5, 1.5)
    assert abs(c["precision"] - 1 / 1.5) < 1e-9 and c["recall"] == 0.5


def test_parse_sheet_extracts_multiline_reply():
    p = Path(tempfile.mkdtemp()) / "sheet.md"
    p.write_text("# t\n\n---\n\n## item 1\n**角色**：A\n**角色设定**：卡片\n**用户上一句（仅参考）**：hi\n\n"
                 "**角色回复**：第一行\n第二行\n\n---\n\n## item 2\n**角色**：B\n**角色设定**：卡2\n"
                 "**用户上一句（仅参考）**：x\n\n**角色回复**：单行\n\n---\n", encoding="utf-8")
    s = parse_sheet(p)
    assert s[1]["reply"] == "第一行\n第二行" and s[1]["card_truncated"] == "卡片"
    assert s[2]["reply"] == "单行"
