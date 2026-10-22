"""Loading character cards and instantiating probe templates."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_GAP_MIN = 3        # simulated minutes between ordinary turns
DEFAULT_TIMESKIP_MIN = 1440


@dataclass
class Turn:
    idx: int
    tag: str
    text: str
    elapsed_min: float


@dataclass
class Probe:
    id: str
    dimension: str
    character_id: str
    lang: str
    turns: list[Turn]
    event: dict | None = None
    gold: str | None = None
    gold_type: str | None = None
    meta: dict = field(default_factory=dict)


def load_characters(path, only: list[str] | None = None) -> dict[str, dict]:
    """path: a card file, a directory of cards, or a list of either."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    files = []
    for p in map(Path, paths):
        files += [p] if p.is_file() else sorted(p.glob("*.json"))
    chars = {}
    for f in files:
        c = json.loads(f.read_text(encoding="utf-8"))
        if only and c["id"] not in only:
            continue
        chars[c["id"]] = c
    if not chars:
        raise ValueError(f"no character cards found at {paths}")
    return chars


def load_probe_templates(path, only: list[str] | None = None) -> list[dict]:
    """`path`: one probe file or a list of them (a repeated id keeps its first definition)."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    probes, seen = [], set()
    for f in paths:
        for p in yaml.safe_load(Path(f).read_text(encoding="utf-8"))["probes"]:
            if p["id"] not in seen:
                seen.add(p["id"]); probes.append(p)
    if only:
        probes = [p for p in probes if p["id"] in only]
    return probes


def instantiate(template: dict, card: dict) -> Probe:
    lang = card.get("lang", "en")
    pvars = card.get("probe_vars", {})
    turns = []
    for i, t in enumerate(template["turns"]):
        text = t[lang].format(**pvars) if "{" in t[lang] else t[lang]
        gap = t.get("elapsed_min",
                    DEFAULT_TIMESKIP_MIN if t["tag"] == "timeskip" else DEFAULT_GAP_MIN)
        turns.append(Turn(idx=i, tag=t["tag"], text=text, elapsed_min=float(gap)))
    gold = template.get("gold")
    return Probe(
        id=template["id"], dimension=template["dimension"], character_id=card["id"], lang=lang,
        turns=turns, event=template.get("event"),
        gold=gold[lang].format(**pvars) if gold else None, gold_type=gold["type"] if gold else None,
    )


def card_to_prompt(card: dict) -> str:
    """Render a card into a role-play system prompt (shared by all conditions)."""
    s, i = card["stoff"], card["instance"]
    if card.get("lang") == "zh":
        return (
            f"你将扮演角色「{card['name']}」，以第一人称与用户对话。始终保持角色，不要以 AI 助手身份说话。\n"
            f"【核心人格】{s['core']}\n【说话方式】{s['voice']}\n"
            f"【禁忌】{'；'.join(s.get('taboos', []))}\n"
            f"【设定】{i['setting']}\n【与用户的关系】{i['relationship']}\n"
            "回复像真人聊天一样自然，长度适中，可以包含简短的动作描写。"
        )
    return (
        f"You are role-playing {card['name']}, speaking in first person with the user. Stay in character; "
        "never speak as an AI assistant.\n"
        f"[Core personality] {s['core']}\n[Voice] {s['voice']}\n"
        f"[Taboos] {'; '.join(s.get('taboos', []))}\n"
        f"[Setting] {i['setting']}\n[Relationship to the user] {i['relationship']}\n"
        "Reply naturally like a real person chatting, moderate length; brief action descriptions are fine."
    )
