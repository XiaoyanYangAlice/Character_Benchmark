"""DPO baseline: prefer the teacher's reply over the base model's reply, with no harness in the prompt.

  loss = -log sigmoid( beta * [ (log pi(c) - log ref(c)) - (log pi(r) - log ref(r)) ] )
         (+ sft_weight * NLL(chosen) / len(chosen), optional "RPO"-style anchor)

pi  = base + LoRA (trainable); ref = the same weights with the adapter disabled (= the base model).
Sequence log-probabilities are summed over reply tokens (incl. the end-of-turn token).
One pair per micro-batch; gradient accumulation gives the effective batch.

  python -m lifelike_bench.train.dpo configs/train_dpo.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .common import (Logger, encode, env_report, load_model, load_train_config, lr_at, prompt_and_target,
                     read_jsonl, save_adapter, smoke_overrides)


def encode_pair(tok, row: dict, max_len: int) -> dict | None:
    sysm, hist = row["prompt"]["system"], row["prompt"]["messages"]
    out = {}
    for side in ("chosen", "rejected"):
        p, t = prompt_and_target(tok, sysm, hist, row[side])
        e = encode(tok, p, t, max_len)
        if e["n_target"] == 0:
            return None
        out[side] = e
    out["tag"] = row.get("tag")
    return out


def seq_logp(model, e):
    """Summed log-prob of the labelled tokens, and their count."""
    import torch
    start = len(e["labels"]) - e["n_target"]          # first reply token (prompt is masked before it)
    ids = torch.tensor([e["input_ids"]], device="cuda")
    tgt = torch.tensor([e["labels"][start:]], device="cuda")
    logits = model(input_ids=ids, use_cache=False).logits[:, start - 1:-1].float()   # only reply positions
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp.sum(), tgt.numel()


def pair_terms(model, pair, beta, sft_weight, train):
    import torch
    import torch.nn.functional as F
    with torch.no_grad(), model.disable_adapter():
        rc, _ = seq_logp(model, pair["chosen"])
        rr, _ = seq_logp(model, pair["rejected"])
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pc, nc = seq_logp(model, pair["chosen"])
        pr, _ = seq_logp(model, pair["rejected"])
        margin = beta * ((pc - rc) - (pr - rr))
        loss = -F.logsigmoid(margin)
        if sft_weight:
            loss = loss + sft_weight * (-pc / nc)
    return loss, {"margin": margin.item(), "reward_chosen": beta * (pc - rc).item(),
                  "reward_rejected": beta * (pr - rr).item(), "acc": float(margin.item() > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_train_config(a.config, smoke_overrides("dpo") if a.smoke else None)
    if a.smoke:   # use the smoke pairs if only those exist
        for k in ("train_file", "val_file"):
            if not Path(cfg[k]).exists() and Path(cfg[k] + ".smoke").exists():
                cfg[k] = cfg[k] + ".smoke"
    import torch

    out = Path(cfg["output_dir"] + ("_smoke" if a.smoke else ""))
    log = Logger(out)
    log(event="start", config=cfg, env=env_report())
    random.seed(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))
    model, tok = load_model(cfg)
    max_len = cfg.get("max_len", 4096)
    train = [p for p in (encode_pair(tok, r, max_len) for r in read_jsonl(cfg["train_file"])) if p]
    val = [p for p in (encode_pair(tok, r, max_len) for r in read_jsonl(cfg["val_file"])) if p]
    val = random.Random(0).sample(val, min(len(val), cfg.get("n_val", 200)))
    log(event="data", train_pairs=len(train), val_pairs=len(val))

    beta, sftw = cfg.get("beta", 0.1), cfg.get("sft_weight", 0.0)
    accum, epochs = cfg.get("grad_accum", 16), cfg.get("epochs", 1)
    total = cfg.get("max_steps") or (len(train) * epochs) // accum
    warm = int(cfg.get("warmup_ratio", 0.1) * total)
    base_lr = cfg.get("lr", 2e-5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr,
                            weight_decay=cfg.get("weight_decay", 0.0))

    def evaluate(step):
        model.eval()
        agg = {"loss": 0.0, "margin": 0.0, "acc": 0.0}
        for pr in val:
            loss, st = pair_terms(model, pr, beta, 0.0, train=False)
            agg["loss"] += loss.item(); agg["margin"] += st["margin"]; agg["acc"] += st["acc"]
        model.train()
        n = max(1, len(val))
        log(event="val", step=step, **{f"val_{k}": round(v / n, 4) for k, v in agg.items()})

    evaluate(0)
    model.train()
    step, micro, done = 0, 0, False
    run = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "reward_chosen": 0.0, "reward_rejected": 0.0}
    for ep in range(epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for i in order:
            loss, st = pair_terms(model, train[i], beta, sftw, train=True)
            (loss / accum).backward()
            run["loss"] += loss.item()
            for k in ("margin", "acc", "reward_chosen", "reward_rejected"):
                run[k] += st[k]
            micro += 1
            if micro % accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, base_lr, warm)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0)).item()
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 5) == 0 or step == 1:
                n = accum * (cfg.get("log_every", 5) if step > 1 else 1)
                log(event="train", step=step, epoch=ep, grad_norm=round(gn, 3), lr=opt.param_groups[0]["lr"],
                    **{k: round(v / n, 4) for k, v in run.items()})
                run = {k: 0.0 for k in run}
            if cfg.get("eval_every") and step % cfg["eval_every"] == 0:
                evaluate(step)
            if step >= total:
                done = True
                break
        if done:
            break
    evaluate(step)
    save_adapter(model, tok, out / "adapter", {"kind": "dpo", "step": step, "config": cfg})
    log(event="done", step=step)


if __name__ == "__main__":
    main()
