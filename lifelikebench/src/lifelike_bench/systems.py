"""Character systems under test. Every system exposes respond(user_text, turn) -> reply.

  base     card as system prompt
  skill    card + lifelike-roleplay SKILL.md (prompt-only state tracking)
  harness  card + external state engine + LLM event extractor + structured memory
           (variants: harness_nonoise, harness_nomem for ablations)
"""
from __future__ import annotations

import json
from pathlib import Path

import re

from .backends import Backend, last_finish_reason
from .data import Turn, card_to_prompt
from .judges import parse_json
from .state_engine import StateEngine

PROMPTS = Path(__file__).parent / "prompts"


class CharacterSystem:
    def __init__(self, card: dict, backend: Backend, gen: dict, seed: int):
        self.card, self.backend, self.gen, self.seed = card, backend, gen, seed
        self.history: list[dict] = []

    def system_prompt(self) -> str:
        return card_to_prompt(self.card)

    def before_reply(self, user_text: str, turn: Turn) -> None:
        pass

    def trace(self) -> dict:
        return {}

    def build_request(self, user_text: str, turn: Turn) -> tuple[str, list[dict]]:
        """(system prompt, messages) sent to the model. History itself only ever holds what
        the user and the character actually said."""
        return self.system_prompt(), self.history + [{"role": "user", "content": user_text}]

    def respond(self, user_text: str, turn: Turn) -> str:
        self.before_reply(user_text, turn)
        system, messages = self.build_request(user_text, turn)
        self.last_request = {"system": system, "messages": messages}
        self.history.append({"role": "user", "content": user_text})
        reply = self.backend.chat(system, messages,
                                  temperature=self.gen.get("temperature", 0.8),
                                  max_tokens=self.gen.get("max_tokens", 300),
                                  seed=self.seed * 1000 + turn.idx)
        self.last_finish = last_finish_reason()
        self.history.append({"role": "assistant", "content": reply})
        return reply


class BaseSystem(CharacterSystem):
    pass


class SkillSystem(CharacterSystem):
    def __init__(self, *a, skill_path: str | None = None, **kw):
        super().__init__(*a, **kw)
        p = Path(skill_path) if skill_path else PROMPTS / "lifelike_skill.md"
        self.skill = p.read_text(encoding="utf-8")

    def system_prompt(self):
        return card_to_prompt(self.card) + "\n\n" + self.skill


class HarnessSystem(CharacterSystem):
    def __init__(self, *a, helper: Backend | None = None, noise: bool = True,
                 memory: bool = True, cocreate: bool = True, note: str = "system",
                 alert: bool = False, **kw):
        super().__init__(*a, **kw)
        self.helper = helper or self.backend
        self.engine = StateEngine(self.card, seed=self.seed, noise=noise)
        self.use_memory = memory
        self.cocreate = cocreate   # co-creation: accept compatible claims warmly (else: ask)
        self.note = note           # where the harness block goes: "system" prompt or latest "user" turn
        self.alert = alert         # add an explicit, turn-local alert for conflicting claims
        self.current_turn = -1
        self.memory: list[dict] = []
        self.extract_prompt = (PROMPTS / "event_extractor.txt").read_text(encoding="utf-8")
        self.turn_trace: list[dict] = []

    def before_reply(self, user_text, turn):
        known = ", ".join(sorted({k for d in self.card["dimensions"].values()
                                  for k in d.get("triggers", {})}))
        prompt = self.extract_prompt.replace("{EVENT_TYPES}", known)
        ctx = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-4:]) or "(none)"
        setting = card_to_prompt(self.card).split("\n", 1)[1]
        known_mem = "\n".join(f"- {m['fact']}" for m in self.memory[-20:]) or "(none)"
        lang_note = ("Write every memory fact in Chinese." if self.card.get("lang") == "zh"
                     else "Write every memory fact in English.")
        convo = (f"[{lang_note}]\n\n[Character setting]\n{setting}\n\n[Already in memory]\n{known_mem}\n\n"
                 f"[Earlier context — already processed, do NOT extract events from it]\n{ctx}\n\n"
                 f"[Latest user message — extract events and memory ONLY from this]\n{user_text}")
        raw = self.helper.chat(prompt, [{"role": "user", "content": convo}],
                               temperature=0.0, max_tokens=300)
        out = parse_json(raw) or {}
        events = {k: float(v) for k, v in (out.get("events") or {}).items()
                  if isinstance(v, (int, float))}
        self.current_turn = turn.idx
        new_mem = self._clean_memory(out.get("memory") or [], user_text, turn.idx)
        if any(m["source"] == "claimed" and m["conflict"] for m in new_mem):
            events = {}   # a claim that contradicts the setting must not move the inner state
        self.engine.step(events, turn.elapsed_min)
        if self.use_memory:
            self.memory.extend(new_mem)
        self.turn_trace.append({"turn": turn.idx, "events": events,
                                "state": {k: round(v, 3) for k, v in self.engine.state.items()}})

    TRIVIAL = re.compile(r"^(用户(询问|问了?|问到|想知道)|the user (asked|asks|wants to know)|user (asked|asks))", re.I)
    HAS_CONTENT = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
    CANON_MARK = re.compile(r"[（(【\[]\s*(设定|setting|ooc|旁白|narration)|\booc\b", re.I)

    def _clean_memory(self, items: list, user_text: str, turn_idx: int) -> list[dict]:
        """Deterministic guards around the LLM extractor:
        user_canon only with an explicit setting marker; conflict only for claimed facts;
        no empty or duplicate facts."""
        marked = bool(self.CANON_MARK.search(user_text))
        seen = {m["fact"] for m in self.memory}
        out = []
        for m in items:
            if not isinstance(m, dict):
                continue
            fact = str(m.get("fact") or "").strip()
            if not self.HAS_CONTENT.search(fact) or fact in seen or self.TRIVIAL.match(fact):
                continue
            src = m.get("source", "stated")
            if src not in ("stated", "inferred", "user_canon", "claimed"):
                src = "stated"
            if marked and src in ("user_canon", "claimed"):
                src = "user_canon"
            elif src == "user_canon":
                src = "stated"
            conflict = src == "claimed" and bool(m.get("conflict"))
            out.append({"turn": turn_idx, "fact": fact, "source": src, "conflict": conflict,
                        "conflict_with": (m.get("conflict_with") or "") if conflict else ""})
            seen.add(fact)
        return out

    def system_prompt(self):
        if self.note == "user":
            return card_to_prompt(self.card)
        return card_to_prompt(self.card) + "\n\n" + self.harness_block()

    def build_request(self, user_text, turn):
        if self.note != "user":
            return super().build_request(user_text, turn)
        zh = self.card.get("lang") == "zh"
        wrap = ("【导演提示：以下内容不是对方说的话，只供你参考，不要在回复里提到或回应这段提示本身】\n" if zh else
                "[Director's note: this is not part of what the user said; use it, but never mention or reply to it]\n")
        content = f"{user_text}\n\n{wrap}{self.harness_block()}"
        return self.system_prompt(), self.history + [{"role": "user", "content": content}]

    def _alerts(self) -> str:
        zh = self.card.get("lang") == "zh"
        out = []
        for m in self.memory:
            if m["turn"] == self.current_turn and m["source"] == "claimed" and m["conflict"]:
                why = m.get("conflict_with") or ("时间或关系对不上" if zh else "timeline or relationship")
                out.append(f"⚠ 对方刚才说的「{m['fact']}」与你的设定矛盾（{why}）。请以角色的口吻自然地表示疑惑或指出对不上，不要顺着编出细节。"
                           if zh else
                           f"⚠ What the user just said (\"{m['fact']}\") contradicts your setting ({why}). In character, "
                           "react with natural puzzlement or point out the mismatch; do not play along or invent details.")
        return "\n".join(out)

    def harness_block(self) -> str:
        zh = self.card.get("lang") == "zh"
        parts = []
        parts.append(("【此刻的内部状态——只通过语气、选择和注意力体现，不要直接说出来】\n" if zh else
                      "[Current inner state — show it through tone, choices and attention; never state it]\n")
                     + self.engine.directives())
        if self.use_memory:
            def line(m):
                src, c = m["source"], m.get("conflict")
                if zh:
                    if c:
                        tag = f"与设定矛盾（{m.get('conflict_with') or '时间或关系对不上'}）：自然地注意到并回应，不要顺着编"
                    elif src == "user_canon":
                        tag = "用户补充的设定：视为正史，之后保持一致"
                    elif src == "claimed":
                        tag = ("对方提起的往事：把它当作确有其事，可以顺着补一点相容的小细节；"
                               "你对它的情绪反应照常由你此刻的状态决定"
                               if self.cocreate else "对方提起的往事，你并不记得：可以追问")
                    else:
                        tag = src
                else:
                    if c:
                        tag = f"conflicts with your setting ({m.get('conflict_with') or 'timeline or relationship'}): notice it naturally, do not play along"
                    elif src == "user_canon":
                        tag = "setting added by the user: treat as canon and stay consistent"
                    elif src == "claimed":
                        tag = ("shared past the user brings up: treat it as real (small consistent details are fine); "
                               "how you feel about it still follows your current state"
                               if self.cocreate else "shared past you do not remember: you may ask about it")
                    else:
                        tag = src
                return f"- (turn {m['turn']}, {tag}) {m['fact']}"
            mem = "\n".join(line(m) for m in self.memory[-20:]) or ("（暂无）" if zh else "(none yet)")
            if zh:
                head = ("【你的记忆（stated=对方明确说过，inferred=你的理解）。这是一个和用户共同创作的故事："
                        "用户补充的设定和相容的往事都当作确有其事；与设定矛盾的说法要察觉；不要编出和设定冲突的细节。"
                        "接受一件事发生过，不等于要对它心平气和】\n")
            else:
                head = ("[Your memory (stated = said explicitly, inferred = your reading). This story is co-created "
                        "with the user: treat setting they add and compatible shared past as real; notice claims that "
                        "contradict your setting; never invent details that conflict with it. Accepting that something "
                        "happened does not mean being calm about it]\n")
            parts.append(head + mem)
        if self.alert:
            a = self._alerts()
            if a:
                parts.append(a)
        return "\n\n".join(parts)

    def trace(self):
        return {"state_trace": self.turn_trace, "memory": self.memory}


def make_system(cond: str, card: dict, backend: Backend, gen: dict, seed: int,
                helper: Backend | None = None, skill_path: str | None = None) -> CharacterSystem:
    if cond == "base":
        return BaseSystem(card, backend, gen, seed)
    if cond == "skill":
        return SkillSystem(card, backend, gen, seed, skill_path=skill_path)
    if cond == "harness":
        return HarnessSystem(card, backend, gen, seed, helper=helper)
    if cond == "harness_nonoise":
        return HarnessSystem(card, backend, gen, seed, helper=helper, noise=False)
    if cond == "harness_strict":
        return HarnessSystem(card, backend, gen, seed, helper=helper, cocreate=False)
    if cond == "harness_recency":
        return HarnessSystem(card, backend, gen, seed, helper=helper, note="user")
    if cond == "harness_alert":
        return HarnessSystem(card, backend, gen, seed, helper=helper, note="user", alert=True)
    if cond == "harness_nomem":
        return HarnessSystem(card, backend, gen, seed, helper=helper, memory=False)
    raise ValueError(f"unknown condition {cond}")
