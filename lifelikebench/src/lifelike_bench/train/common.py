"""Shared training utilities. Torch/transformers/peft are imported lazily so that the data
logic here can be tested without them."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------- config
def load_train_config(path: str, overrides: dict | None = None) -> dict:
    from ..cli import _expand
    cfg = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if overrides:
        cfg.update(overrides)
    return cfg


def smoke_overrides(kind: str) -> dict:
    """Tiny settings for a first check on the cluster (a few minutes)."""
    if kind == "sft":
        return {"max_steps": 4, "grad_accum": 2, "max_val_examples": 8, "save_every_epoch": False}
    if kind == "dpo":
        return {"max_steps": 3, "grad_accum": 2, "n_val": 4}
    if kind == "dpo_pairs":
        return {"limit": 16}
    return {"steps": 3, "batch_size": 2, "max_new_tokens": 48, "eval_every": 2, "n_val": 4, "save_every": 0}


# ---------------------------------------------------------------- chat rendering
def render(tok, system: str, messages: list[dict], add_generation_prompt: bool) -> str:
    """Chat-template text, with Qwen3 thinking switched off (other templates ignore the flag)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)


def prompt_and_target(tok, system: str, history: list[dict], reply: str) -> tuple[str, str]:
    """Split one assistant turn into (prompt, target) text.

    The prompt is exactly the generation prompt the model sees at inference time; the target is
    the reply plus whatever end-of-turn tokens the template appends after it."""
    prompt = render(tok, system, history, add_generation_prompt=True)
    full = render(tok, system, history + [{"role": "assistant", "content": reply}], add_generation_prompt=False)
    if full.startswith(prompt):
        return prompt, full[len(prompt):]
    j = full.rfind(reply)
    if j >= 0:   # template renders the final turn differently from the generation prompt
        return prompt, full[j:]
    return prompt, reply + (tok.eos_token or "")


def encode(tok, prompt: str, target: str, max_len: int) -> dict:
    """input_ids / labels with the prompt masked (-100). Long prompts are cut from the left."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    t = t[:max_len // 2] if len(t) > max_len // 2 else t
    p = p[-(max_len - len(t)):] if len(p) + len(t) > max_len else p
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "n_target": len(t)}


def sft_examples(rows: list[dict]) -> list[tuple[str, list[dict], str]]:
    """Every assistant turn of every dialogue -> (system, history, reply)."""
    out = []
    for r in rows:
        msgs = r["messages"]
        system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        body = msgs[1:] if system else msgs
        for i, m in enumerate(body):
            if m["role"] == "assistant":
                out.append((system, body[:i], m["content"]))
    return out


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def weighted_sampler(rows: list[dict], weights: dict, seed: int):
    """Yield rows forever, oversampling tags listed in `weights` (default weight 1)."""
    rng = random.Random(seed)
    w = [float(weights.get(r.get("tag"), 1.0)) for r in rows]
    while True:
        yield rng.choices(rows, weights=w, k=1)[0]


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


class Logger:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.f = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, **kw):
        kw["elapsed_s"] = round(time.time() - self.t0, 1)
        line = json.dumps(kw, ensure_ascii=False)
        self.f.write(line + "\n")
        self.f.flush()
        print(line, flush=True)


# ---------------------------------------------------------------- model loading (needs torch)
def load_model(cfg: dict, adapter: str | None = None):
    """Base model in bf16 + a trainable LoRA adapter (new, or loaded from `adapter`)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"attn_implementation": cfg.get("attn_implementation", "sdpa")}
    try:   # transformers >= 4.56 / 5.x
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, **kw)
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        lc = cfg.get("lora", {})
        model = get_peft_model(model, LoraConfig(
            r=lc.get("r", 64), lora_alpha=lc.get("alpha", 128), lora_dropout=lc.get("dropout", 0.05),
            target_modules=lc.get("target_modules",
                                  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    return model, tok


def save_adapter(model, tok, path: Path, meta: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "lifelike_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved adapter to {path}", flush=True)


def env_report() -> dict:
    import torch
    import transformers
    import peft
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hf_offline": os.environ.get("HF_HUB_OFFLINE")}
