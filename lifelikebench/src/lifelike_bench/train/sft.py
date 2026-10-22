"""Off-policy context distillation (SFT baseline): LoRA-tune the student on the teacher's replies,
with the harness removed from the prompt. Loss only on assistant tokens.

  python -m lifelike_bench.train.sft configs/train_sft.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import (Logger, encode, env_report, load_model, load_train_config, lr_at, prompt_and_target,
                     read_jsonl, save_adapter, sft_examples, smoke_overrides)


def build(tok, rows, max_len):
    ex = []
    for system, history, reply in sft_examples(rows):
        p, t = prompt_and_target(tok, system, history, reply)
        e = encode(tok, p, t, max_len)
        if e["n_target"] > 0:
            ex.append(e)
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_train_config(a.config, smoke_overrides("sft") if a.smoke else None)
    import torch

    out = Path(cfg["output_dir"] + ("_smoke" if a.smoke else ""))
    log = Logger(out)
    log(event="start", config=cfg, env=env_report())
    random.seed(cfg.get("seed", 0)); torch.manual_seed(cfg.get("seed", 0))

    model, tok = load_model(cfg)
    train = build(tok, read_jsonl(cfg["train_file"]), cfg.get("max_len", 3072))
    val = build(tok, read_jsonl(cfg["val_file"]), cfg.get("max_len", 3072))[: cfg.get("max_val_examples", 400)]
    log(event="data", train_examples=len(train), val_examples=len(val),
        train_target_tokens=sum(e["n_target"] for e in train))

    accum = cfg.get("grad_accum", 16)
    epochs = cfg.get("epochs", 2)
    total = cfg.get("max_steps") or (len(train) * epochs) // accum
    warmup = int(cfg.get("warmup_ratio", 0.05) * total)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 0.0))

    def run_val():
        model.eval()
        losses = []
        with torch.no_grad():
            for e in val:
                ids = torch.tensor([e["input_ids"]], device="cuda")
                lab = torch.tensor([e["labels"]], device="cuda")
                losses.append(model(input_ids=ids, labels=lab).loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    log(event="val", step=0, val_loss=run_val())
    model.train()
    step, micro, run_loss = 0, 0, 0.0
    order = list(range(len(train)))
    done = False
    for ep in range(epochs):
        random.shuffle(order)
        for i in order:
            e = train[i]
            ids = torch.tensor([e["input_ids"]], device="cuda")
            lab = torch.tensor([e["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=lab).loss / accum
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total, cfg.get("lr", 1e-4), warmup)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.get("log_every", 10) == 0 or step == 1:
                    log(event="train", step=step, epoch=ep, loss=round(run_loss / min(step, cfg.get("log_every", 10)) if step > 1 else run_loss, 4),
                        lr=opt.param_groups[0]["lr"])
                    run_loss = 0.0
                if step >= total:
                    done = True
                    break
        log(event="val", step=step, epoch=ep, val_loss=run_val())
        if cfg.get("save_every_epoch", True) and not done:
            save_adapter(model, tok, out / f"epoch{ep}", {"kind": "sft", "step": step, "epoch": ep})
        if done:
            break
    save_adapter(model, tok, out / "adapter", {"kind": "sft", "step": step, "config": cfg})
    log(event="done", step=step)


if __name__ == "__main__":
    main()
