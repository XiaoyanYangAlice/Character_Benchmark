"""Build preference pairs for the DPO baseline: chosen = the teacher's reply (8B + alert harness),
rejected = the plain base model's reply to the SAME student prompt (card + dialogue, no harness).

The rejected replies are sampled once, offline, with vLLM (fast), from the base model the student
starts from. Both sides of a pair therefore share the student's prompt; only the reply differs.

  python -m lifelike_bench.train.make_dpo_pairs configs/train_dpo.yaml [--smoke]
  (needs a GPU; run it through slurm/train.sbatch: sbatch slurm/train.sbatch make_dpo_pairs configs/train_dpo.yaml)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .common import load_train_config, read_jsonl, render, smoke_overrides


def keep_pair(chosen: str, rejected: str, finish: str | None) -> str | None:
    """Return a reason to drop the pair, or None to keep it."""
    if not rejected.strip():
        return "empty_rejected"
    if finish not in (None, "stop"):
        return "rejected_unfinished"
    if rejected.strip() == chosen.strip():
        return "identical"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_train_config(a.config, smoke_overrides("dpo_pairs") if a.smoke else None)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    llm = LLM(model=cfg["base_model"], dtype="bfloat16", seed=cfg.get("seed", 0),
              max_model_len=cfg.get("pairs_max_model_len", 8192),
              gpu_memory_utilization=cfg.get("pairs_gpu_memory_utilization", 0.85))
    sp = SamplingParams(temperature=cfg.get("pairs_temperature", 0.8), top_p=cfg.get("pairs_top_p", 0.8),
                        max_tokens=cfg.get("pairs_max_tokens", 300), seed=cfg.get("seed", 0))

    for split in ("train", "val"):
        rows = read_jsonl(cfg[f"opd_{split}_file"])
        if cfg.get("limit"):
            rows = random.Random(0).sample(rows, min(len(rows), cfg["limit"]))
        prompts = [render(tok, r["student"]["system"], r["student"]["messages"], add_generation_prompt=True)
                   for r in rows]
        outs = llm.generate(prompts, sp)
        stats = Counter()
        out_path = Path(cfg[f"{split}_file"] + (".smoke" if a.smoke else ""))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                rej = o.outputs[0].text
                why = keep_pair(r["teacher_reply"], rej, o.outputs[0].finish_reason)
                if why:
                    stats[why] += 1
                    continue
                f.write(json.dumps({"id": r["id"], "character": r["character"], "tag": r["tag"],
                                    "prompt": r["student"], "chosen": r["teacher_reply"], "rejected": rej},
                                   ensure_ascii=False) + "\n")
                stats["kept"] += 1
        print(split, out_path, dict(stats), flush=True)


if __name__ == "__main__":
    main()
