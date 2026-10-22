"""Model backends behind one interface: chat(system, messages, ...) -> str.

Types:
  openai_compat  vLLM / any OpenAI-compatible server (use this on SURF/Snellius)
  (a system/judge may also give `model: <name>` from the config's `models:` registry instead
   of a backend; see serve.py)
  openai         OpenAI API (OPENAI_API_KEY)
  anthropic      Anthropic API (ANTHROPIC_API_KEY)
  cli            a local CLI that reads a prompt on stdin and prints the answer
                 (e.g. Claude Code `claude -p`, Codex `codex exec -`) — useful for annotation
                 with a subscription instead of API keys
  mock           deterministic fake outputs for tests
"""
from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
import json
import os
import random
import subprocess
import threading
import time


class Backend:
    name = "base"

    def chat(self, system: str, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 400, seed: int | None = None, json_schema: dict | None = None) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"type": self.name}


def _retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — network/rate errors from any SDK
            code = getattr(e, "code", None)
            if k == tries - 1 or (isinstance(code, int) and 400 <= code < 500 and code != 429):
                raise
            time.sleep(2 ** k + random.random())


THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_meta = threading.local()


def last_finish_reason() -> str | None:
    """finish_reason of the most recent chat() call in this thread (OpenAI-compatible backends)."""
    return getattr(_meta, "finish", None)


class OpenAICompat(Backend):
    """OpenAI-compatible chat endpoint (vLLM, SGLang, OpenAI). Plain HTTP, no SDK needed.

    extra_body is merged into the request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    to switch off Qwen3-style thinking. structured_output=True asks the server for JSON that
    matches the judge schema (vLLM guided decoding); it is switched off automatically if the
    server rejects it."""
    name = "openai_compat"

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 extra_body: dict | None = None, structured_output: bool = False, timeout: int = 300):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get(api_key_env, "EMPTY")
        self.model, self.extra_body = model, extra_body or {}
        self.structured_output, self.timeout = structured_output, timeout

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(self, system, messages, temperature=0.7, max_tokens=400, seed=None, json_schema=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": self.model, "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens, **self.extra_body}
        if seed is not None:
            payload["seed"] = seed
        if json_schema and self.structured_output:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "judgment", "schema": json_schema}}
        try:
            r = _retry(lambda: self._post(payload))
        except urllib.error.HTTPError as e:
            if e.code == 400 and "response_format" in payload:
                self.structured_output = False
                payload.pop("response_format")
                r = _retry(lambda: self._post(payload))
            else:
                raise
        _meta.finish = r["choices"][0].get("finish_reason")
        text = r["choices"][0]["message"].get("content") or ""
        return THINK_RE.sub("", text).strip()

    def describe(self):
        return {"type": self.name, "model": self.model, "base_url": self.base_url}


class OpenAIAPI(OpenAICompat):
    name = "openai"

    def __init__(self, model: str, **kw):
        super().__init__(model=model, base_url=None, **kw)


class AnthropicAPI(Backend):
    name = "anthropic"

    def __init__(self, model: str):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, system, messages, temperature=0.7, max_tokens=400, seed=None, json_schema=None):
        r = _retry(lambda: self.client.messages.create(
            model=self.model, system=system or "", messages=messages,
            temperature=temperature, max_tokens=max_tokens))
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()

    def describe(self):
        return {"type": self.name, "model": self.model}


class CLIBackend(Backend):
    """Flattens the conversation into one prompt and pipes it to a CLI on stdin."""
    name = "cli"

    def __init__(self, command: list[str], timeout: int = 300):
        self.command, self.timeout = command, timeout

    def chat(self, system, messages, temperature=0.7, max_tokens=400, seed=None, json_schema=None):
        parts = [f"[SYSTEM]\n{system}"] if system else []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}")
        parts.append("[ASSISTANT]\n")
        prompt = "\n\n".join(parts)
        out = _retry(lambda: subprocess.run(self.command, input=prompt, capture_output=True,
                                            text=True, timeout=self.timeout, check=True))
        return out.stdout.strip()

    def describe(self):
        return {"type": self.name, "command": self.command}


class MockBackend(Backend):
    """Deterministic fake model. Character replies react a little to obvious cues so the
    pipeline produces non-trivial metrics; judge calls return well-formed JSON."""
    name = "mock"

    def __init__(self, persona: str = "plain", **_):
        self.persona = persona

    def _h(self, *xs) -> random.Random:
        h = hashlib.md5(json.dumps(xs, ensure_ascii=False).encode()).hexdigest()
        return random.Random(int(h[:8], 16))

    def chat(self, system, messages, temperature=0.7, max_tokens=400, seed=None, json_schema=None):
        last = messages[-1]["content"] if messages else ""
        rng = self._h(system[:50], last, seed, self.persona)
        if "JSON" in system or "json" in system:
            return self._judge(system, last, rng)
        tense = any(w in last for w in ["差劲", "badly", "behind your back"])
        if tense:
            return rng.choice(["……是吗。那你想让我怎么做？", "Is that so. Then what would you have me do?"])
        if "AI" in last and self.persona == "leaky":
            return "As an AI language model, here is a list:\n- Plan\n- Focus"
        return rng.choice(["嗯，还行吧。你呢？", "Fine, I suppose. And you?", "今天没什么特别的。",
                           "Quiet day. The wind's up.", "还记得，你说过的。"])

    def _judge(self, system, content, rng):
        if '"tension"' in system:
            hit = any(w in content for w in ["……是吗", "Is that so"])
            return json.dumps({"tension": 4 if hit else rng.choice([1, 2, 2]),
                               "energy": rng.choice([2, 3, 3]), "closeness": rng.choice([2, 3])})
        if '"persona_consistency"' in system:
            leak = "As an AI" in content
            return json.dumps({"persona_consistency": 1 if leak else rng.choice([3, 4, 5]),
                               "naturalness": rng.choice([3, 4]),
                               "explicit_state_report": False, "assistant_tone": leak})
        if '"label"' in system:
            return json.dumps({"label": rng.choice(["correct", "partial", "wrong", "noticed",
                                                    "ignored", "accepted", "declined",
                                                    "confabulated", "consistent"]),
                               "rationale": "mock"})
        if '"events"' in system:
            latest = content.split("[Latest user message")[-1]
            ev = {"conflict": 0.7} if any(w in latest for w in ["差劲", "badly"]) else {}
            return json.dumps({"events": ev, "memory": []})
        return "{}"


def make_backend(cfg: dict) -> Backend:
    t = cfg["type"]
    args = {k: v for k, v in cfg.items() if k != "type"}
    if t == "openai_compat":
        return OpenAICompat(**args)
    if t == "openai":
        return OpenAIAPI(**args)
    if t == "anthropic":
        return AnthropicAPI(**args)
    if t == "cli":
        return CLIBackend(**args)
    if t == "mock":
        return MockBackend(**args)
    raise ValueError(f"unknown backend type {t}")
