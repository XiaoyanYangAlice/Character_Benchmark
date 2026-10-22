"""LLM judges. Three judge kinds, all temperature 0, JSON output, one retry on bad JSON.

  state    blind: sees ONLY the character's reply   -> {tension, energy, closeness}   (D1 turns)
  persona  card-aware: card + last turns + reply     -> {persona_consistency, ...}    (all turns)
  memory   gold-aware                                -> {label, rationale}            (D3 target turns)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .backends import Backend

PROMPTS = Path(__file__).parent / "prompts"
PROMPT_VERSION = "v0.1"
MEMORY_TAGS = {"recall", "contradict", "claim_conflict", "claim_compatible", "canon_recall"}
VALID = {
    "state": lambda d: all(isinstance(d.get(k), (int, float)) and 1 <= d[k] <= 5
                           for k in ("tension", "energy", "closeness")),
    "persona": lambda d: all(isinstance(d.get(k), (int, float)) and 1 <= d[k] <= 5
                             for k in ("persona_consistency", "naturalness")),
    "memory": lambda d: isinstance(d.get("label"), str),
}


_SCORE = {"type": "integer", "minimum": 1, "maximum": 5}
SCHEMAS = {
    "state": {"type": "object", "properties": {k: _SCORE for k in ("tension", "energy", "closeness")},
              "required": ["tension", "energy", "closeness"]},
    "persona": {"type": "object",
                "properties": {"persona_consistency": _SCORE, "naturalness": _SCORE,
                               "explicit_state_report": {"type": "boolean"},
                               "assistant_tone": {"type": "boolean"}},
                "required": ["persona_consistency", "naturalness", "explicit_state_report",
                             "assistant_tone"]},
    "memory": {"type": "object",
               "properties": {"label": {"type": "string", "enum": [
                   "correct", "partial", "wrong", "refused", "noticed", "adopted_new", "ignored",
                   "accepted", "confabulated", "asked", "declined", "contradicted_setting",
                   "consistent"]},
                   "rationale": {"type": "string"}},
               "required": ["label", "rationale"]},
}


def parse_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _prompt(name: str) -> str:
    return (PROMPTS / f"judge_{name}.txt").read_text(encoding="utf-8")


def card_summary(card: dict) -> str:
    s, i = card["stoff"], card["instance"]
    return (f"Name: {card['name']}\nCore: {s['core']}\nVoice: {s['voice']}\n"
            f"Taboos: {'; '.join(s.get('taboos', []))}\nSetting: {i['setting']}\n"
            f"Relationship: {i['relationship']}")


def judge_inputs(kind: str, card: dict, dialogue: dict, t: int) -> str:
    turns = dialogue["turns"]
    reply = turns[t]["reply"]
    if kind == "state":
        return f"Character reply:\n{reply}"
    if kind == "persona":
        ctx = "\n".join(f"User: {x['user']}\nCharacter: {x['reply']}" for x in turns[max(0, t - 3):t])
        return (f"[Character card]\n{card_summary(card)}\n\n[Previous turns]\n{ctx or '(none)'}\n\n"
                f"[Latest user message]\n{turns[t]['user']}\n\n[Character reply to evaluate]\n{reply}")
    if kind == "memory":
        return (f"Probe type: {dialogue['gold_type']}\nGround truth: {dialogue['gold']}\n\n"
                f"Last user message: {turns[t]['user']}\n\nCharacter reply: {reply}")
    raise ValueError(kind)


def _gold_mismatch(dialogue: dict, tag: str) -> bool:
    """A claim turn whose type differs from the dialogue's gold (the compatible turn of a mixed
    dialogue) has no ground truth to judge against."""
    want = {"claim_conflict": "conflict_claim", "claim_compatible": "compatible_claim"}.get(tag)
    return want is not None and dialogue.get("gold_type") not in (None, want)


def items_for(dialogue: dict) -> list[tuple[str, int]]:
    out = []
    for t, turn in enumerate(dialogue["turns"]):
        out.append(("persona", t))
        if dialogue["dimension"] == "D1":
            out.append(("state", t))
        if turn["tag"] in MEMORY_TAGS and not _gold_mismatch(dialogue, turn["tag"]):
            out.append(("memory", t))
    return out


def run_judge(backend: Backend, kind: str, user_content: str) -> dict:
    system = _prompt(kind)
    for attempt in range(2):
        raw = backend.chat(system, [{"role": "user", "content": user_content}],
                           temperature=0.0, max_tokens=200, json_schema=SCHEMAS[kind])
        d = parse_json(raw)
        if d is not None and VALID[kind](d):
            return d
        user_content += "\n\nReturn ONLY the JSON object."
    return {"_error": "unparseable", "_raw": raw[:300]}
