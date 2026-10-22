"""Turn teacher transcripts into training data for the student (no harness at all).

  PYTHONPATH=src python -m lifelike_bench.train.build_data configs/train_teacher.yaml --out DIR

Writes, split by CHARACTER (held-out characters go to *_val):
  sft_{train,val}.jsonl  {"id", "character", "messages": [system(card only), user, assistant, ...]}
                         -> off-policy context distillation: the student learns the teacher's
                            replies without seeing the harness.
  opd_{train,val}.jsonl  one row per assistant turn:
                         {"id", "character", "tag",
                          "student": {"system", "messages"},   # card + clean history + user turn
                          "teacher": {"system", "messages"},   # exactly what the teacher saw
                          "teacher_reply"}
                         -> on-policy distillation prompts (student samples, teacher scores).
  stats.json

Filtering (default on when judgments exist): drop dialogues whose memory turns the judge did not
label as the desired behaviour (conflict -> noticed, compatible -> accepted, canon -> consistent),
turns that did not finish normally, and replies that mention the director's note.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..cli import load_config, out_dir
from ..data import card_to_prompt, load_characters
from ..runner import dialogue_key, read_jsonl

# desired label per memory turn, keyed by the turn's tag; a label only counts when the dialogue's
# gold type matches the tag (in train_mixed the gold is the conflict, so the judge's label on the
# compatible turn was made against the wrong ground truth and is ignored)
TAG_GOOD = {"claim_conflict": ("conflict_claim", {"noticed"}),
            "claim_compatible": ("compatible_claim", {"accepted"}),
            "canon_recall": ("canon", {"consistent"})}
LEAK = ("导演提示", "Director's note", "director's note", "【你的记忆", "[Your memory", "此刻的内部状态",
        "Current inner state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = out_dir(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    chars = load_characters(cfg["characters"], cfg.get("only_characters"))
    T = read_jsonl(run / "transcripts.jsonl")
    J = read_jsonl(run / "judgments.jsonl") if (run / "judgments.jsonl").exists() else []
    mem = defaultdict(list)
    for r in J:
        if r["kind"] == "memory" and "_error" not in r["result"]:
            mem[(r["key"], r["turn"])].append(r["result"]["label"])
    filt = bool(J) and not a.no_filter

    ids = sorted(chars)
    rng = random.Random(a.seed)
    rng.shuffle(ids)
    n_val = max(1, round(a.val_frac * len(ids)))
    val_chars = set(ids[:n_val])

    stats = Counter()
    files = {k: (out / f"{k}.jsonl").open("w", encoding="utf-8")
             for k in ("sft_train", "sft_val", "opd_train", "opd_val")}
    for d in T:
        stats["dialogues"] += 1
        key = dialogue_key(d)
        if filt:
            failed = False
            for i, t in enumerate(d["turns"]):
                gt, good = TAG_GOOD.get(t["tag"], (None, None))
                if gt is None or gt != d.get("gold_type"):
                    continue
                labs = mem.get((key, i), [])
                if not labs or any(l not in good for l in labs):
                    failed = True
            if failed:
                stats["dropped_teacher_failed"] += 1
                stats[f"dropped_teacher_failed:{d['probe']}"] += 1
                continue
        if any(t.get("finish_reason") not in (None, "stop") for t in d["turns"]):
            stats["dropped_unfinished"] += 1
            continue
        if any(x in t["reply"] for t in d["turns"] for x in LEAK):
            stats["dropped_note_leak"] += 1
            continue
        if any("request" not in t for t in d["turns"]):
            raise SystemExit("transcripts have no recorded requests: run the teacher with record_requests: true")
        card = chars[d["character"]]
        system = card_to_prompt(card)
        split = "val" if d["character"] in val_chars else "train"
        history = []
        for t in d["turns"]:
            files[f"opd_{split}"].write(json.dumps({
                "id": f"{key}|{t['idx']}", "character": d["character"], "tag": t["tag"],
                "student": {"system": system, "messages": history + [{"role": "user", "content": t["user"]}]},
                "teacher": t["request"], "teacher_reply": t["reply"]}, ensure_ascii=False) + "\n")
            history = history + [{"role": "user", "content": t["user"]},
                                 {"role": "assistant", "content": t["reply"]}]
            stats[f"opd_{split}_rows"] += 1
        files[f"sft_{split}"].write(json.dumps({
            "id": key, "character": d["character"],
            "messages": [{"role": "system", "content": system}] + history}, ensure_ascii=False) + "\n")
        stats[f"sft_{split}_dialogues"] += 1
    for f in files.values():
        f.close()
    stats["val_characters"] = len(val_chars)
    (out / "stats.json").write_text(json.dumps({**stats, "filtered": filt,
                                                 "val_character_ids": sorted(val_chars)}, indent=1),
                                    encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
