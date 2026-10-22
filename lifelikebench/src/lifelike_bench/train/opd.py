"""On-policy distillation with privileged information (self-distillation of the harness).

Student = base model + LoRA, prompted with the character card and the dialogue only.
Teacher = the SAME base model with the adapter switched off, prompted with exactly what the
harness_alert teacher saw (card + director's note with state, provenance-tagged memory and the
turn-local conflict alert).

Each step:
  1. sample B prompts (weighted by scenario tag) from opd_train.jsonl
  2. the student generates a reply (adapter on, no grad)
  3. both models score the student's tokens; loss = full-vocabulary reverse KL
        sum_v p_s(v) [log p_s(v) - log p_t(v)]   averaged over response tokens (incl. end token)
     The gradient flows only through the student.

The dialogue history is the teacher's (from the recorded transcripts); only the current turn is
sampled on-policy. This is the usual turn-level approximation for multi-turn distillation.

  python -m lifelike_bench.train.opd configs/train_opd.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import (Logger, env_report, load_model, load_train_config, lr_at, read_jsonl, render,
                     save_adapter, smoke_overrides, weighted_sampler)


def eos_ids(model, tok) -> list[int]:
    e = getattr(model.generation_config, "eos_token_id", None) or tok.eos_token_id
    return list(e) if isinstance(e, (list, tuple)) else [e]


def cut_at_eos(ids: list[int], eos: set[int], pad: int) -> list[int]:
    out = []
    for i in ids:
        if i in eos:
            out.append(i)
            return out
        if i == pad and pad not in eos:
            break
        out.append(i)
    return out


def prompt_ids(tok, side: dict, max_prompt: int) -> list[int]:
    ids = tok(render(tok, side["system"], side["messages"], add_generation_prompt=True),
              add_special_tokens=False)["input_ids"]
    return ids[-max_prompt:]


def generate(model, tok, prompts: list[list[int]], cfg: dict, greedy=False) -> list[list[int]]:
    import torch
    model.eval()
    L = max(len(p) for p in prompts)
    pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (L - len(p)) + p for p in prompts], device="cuda")
    att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    eos = eos_ids(model, tok)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=cfg.get("max_new_tokens", 256),
                             do_sample=not greedy, temperature=cfg.get("temperature", 1.0),
                             top_p=cfg.get("top_p", 1.0), eos_token_id=eos, pad_token_id=pad,
                             use_cache=True)
    model.train()
    return [cut_at_eos(row[L:].tolist(), set(eos), pad) for row in out]


def response_logits(model, prompt: list[int], resp: list[int]):
    """Logits that predict each response token: shape [len(resp), vocab]."""
    import torch
    ids = torch.tensor([prompt + resp], device="cuda")
    logits = model(input_ids=ids, use_cache=False).logits[0]
    return logits[len(prompt) - 1: len(prompt) - 1 + len(resp)].float()


def kl_reverse(s_logits, t_logits):
    import torch.nn.functional as F
    ls = F.log_softmax(s_logits, -1)
    lt = F.log_softmax(t_logits, -1)
    return (ls.exp() * (ls - lt)).sum(-1)          # per token


def step_loss(model, tok, rows, cfg, train=True):
    """Generate for `rows`, return (mean KL, stats, samples). Backprops when train=True."""
    import torch
    mp = cfg.get("max_prompt_tokens", 6000)
    s_prompts = [prompt_ids(tok, r["student"], mp) for r in rows]
    t_prompts = [prompt_ids(tok, r["teacher"], mp) for r in rows]
    resps = generate(model, tok, s_prompts, cfg, greedy=not train)
    n_tok = sum(len(x) for x in resps)
    tot, per = 0.0, []
    for sp, tp, rp in zip(s_prompts, t_prompts, resps):
        if not rp:
            per.append(0.0)
            continue
        with torch.no_grad(), model.disable_adapter():
            t_log = response_logits(model, tp, rp)
        if train:
            s_log = response_logits(model, sp, rp)
            kl = kl_reverse(s_log, t_log)
            (kl.sum() / max(1, n_tok)).backward()      # token-level mean over the batch
        else:
            with torch.no_grad():
                kl = kl_reverse(response_logits(model, sp, rp), t_log)
        tot += kl.sum().item()
        per.append(kl.mean().item())
        del t_log, kl
    samples = [{"id": r["id"], "tag": r["tag"], "user": r["student"]["messages"][-1]["content"],
                "student": tok.decode(rp, skip_special_tokens=True), "teacher_ref": r["teacher_reply"]}
               for r, rp in zip(rows, resps)]
    return tot / max(1, n_tok), {"resp_tokens": n_tok / max(1, len(rows)),
                                  "truncated": sum(1 for rp in resps if rp and rp[-1] not in eos_ids(model, tok))}, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_train_config(a.config, smoke_overrides("opd") if a.smoke else None)
    import torch

    out = Path(cfg["output_dir"] + ("_smoke" if a.smoke else ""))
    log = Logger(out)
    log(event="start", config=cfg, env=env_report())
    random.seed(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))
    model, tok = load_model(cfg, adapter=cfg.get("init_adapter") or None)
    tok.padding_side = "left"
    # sample from the full distribution: drop the model's default top-k/top-p (Qwen3 ships top_k=20)
    gc = model.generation_config
    gc.top_k, gc.top_p, gc.temperature = None, 1.0, 1.0

    train = read_jsonl(cfg["train_file"])
    val = read_jsonl(cfg["val_file"])
    rng = random.Random(cfg.get("seed", 0))
    val = rng.sample(val, min(len(val), cfg.get("n_val", 64)))
    sampler = weighted_sampler(train, cfg.get("tag_weights", {}), cfg.get("seed", 0))
    log(event="data", train_rows=len(train), val_rows=len(val))

    steps, B = cfg.get("steps", 200), cfg.get("batch_size", 16)
    accum = cfg.get("grad_accum", 1)
    base_lr = cfg.get("lr", 5e-5)
    warm = int(cfg.get("warmup_ratio", 0.05) * steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr,
                            weight_decay=cfg.get("weight_decay", 0.0))
    samples_f = (out / "samples.jsonl").open("a", encoding="utf-8")

    def evaluate(step):
        kls, lens = [], []
        mb = cfg.get("eval_batch_size", 16)
        for i in range(0, len(val), mb):
            kl, st, smp = step_loss(model, tok, val[i:i + mb], cfg, train=False)
            kls.append(kl); lens.append(st["resp_tokens"])
            if i == 0:
                for s in smp[:6]:
                    samples_f.write(json.dumps({"step": step, **s}, ensure_ascii=False) + "\n")
                samples_f.flush()
        log(event="val", step=step, val_kl=round(sum(kls) / len(kls), 4), val_resp_tokens=round(sum(lens) / len(lens), 1))

    evaluate(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1, steps, base_lr, warm)
        kls, sts = [], []
        for _ in range(accum):
            rows = [next(sampler) for _ in range(B)]
            kl, st, _ = step_loss(model, tok, rows, cfg, train=True)
            kls.append(kl); sts.append(st)
        if accum > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0)).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % cfg.get("log_every", 1) == 0:
            log(event="train", step=step, kl=round(sum(kls) / len(kls), 4), grad_norm=round(gn, 3),
                resp_tokens=round(sum(s["resp_tokens"] for s in sts) / len(sts), 1),
                truncated=sum(s["truncated"] for s in sts), lr=opt.param_groups[0]["lr"])
        if cfg.get("eval_every") and step % cfg["eval_every"] == 0:
            evaluate(step)
        if cfg.get("save_every") and step % cfg["save_every"] == 0 and step < steps:
            save_adapter(model, tok, out / f"step{step}", {"kind": "opd", "step": step})
    save_adapter(model, tok, out / "adapter", {"kind": "opd", "step": steps, "config": cfg})
    log(event="done", step=steps)


if __name__ == "__main__":
    main()
