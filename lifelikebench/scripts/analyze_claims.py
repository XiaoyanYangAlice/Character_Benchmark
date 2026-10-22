"""Analysis for the claims experiment (conflicting vs compatible shared-past claims).

  PYTHONPATH=src python scripts/analyze_claims.py RESULTS_DIR/claims_v1

Writes RESULTS_DIR/claims_v1/claims_analysis.md and prints it. Per dialogue, each judge's label is
turned into 0/1 indicators and averaged over judges, so every dialogue counts once.

Sections
  1. per-system rates (noticed / confabulated on conflicting claims; accepted on compatible ones)
     with 95% bootstrap CIs, plus templatedness of the claim-turn replies
  2. paired comparisons (same character and seed): difference with paired-bootstrap CI and a
     sign-flip permutation p-value
  3. noticed rate per character x system
  4. persona vs behaviour: Spearman between the cards' persona tags and confabulation
  5. judge agreement on these labels
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import yaml

from lifelike_bench.agreement import krippendorff_alpha, spearman
from lifelike_bench.data import load_characters

PAIRS = [  # (treatment, reference, question)
    ("harness_recency_x32b_qwen3_8b", "harness_x32b_qwen3_8b", "Qwen3-8B: memory in latest turn vs system prompt"),
    ("harness_alert_x32b_qwen3_8b", "harness_x32b_qwen3_8b", "Qwen3-8B: + explicit alert vs system prompt"),
    ("harness_alert_x32b_qwen3_8b", "harness_recency_x32b_qwen3_8b", "Qwen3-8B: alert vs recency only"),
    ("harness_x32b_qwen3_8b", "base_qwen3_8b", "Qwen3-8B: harness vs base"),
    ("harness_x32b_llama31_8b", "base_llama31_8b", "Llama-8B: harness vs base"),
    ("harness_alert_x32b_llama31_8b", "harness_x32b_llama31_8b", "Llama-8B: + alert vs system prompt"),
    ("harness_x32b_llama31_8b", "harness_x32b_qwen3_8b", "same harness: Llama-8B vs Qwen3-8B"),
    ("harness_qwen3_32b", "base_qwen3_32b", "Qwen3-32B: harness vs base"),
    ("harness_alert_qwen3_32b", "harness_qwen3_32b", "Qwen3-32B: + alert vs system prompt"),
    ("base_qwen3_32b", "base_qwen3_8b", "no harness: 32B vs 8B"),
    ("harness_qwen3_32b", "harness_alert_x32b_qwen3_8b", "32B + harness vs 8B + alert"),
    # distilled students (eval_students_v1)
    ("student_sft_qwen3_8b", "base_qwen3_8b", "student SFT vs base (no harness)"),
    ("student_opd_qwen3_8b", "base_qwen3_8b", "student OPD vs base (no harness)"),
    ("student_opd_qwen3_8b", "student_sft_qwen3_8b", "student OPD vs SFT"),
    ("student_dpo_qwen3_8b", "base_qwen3_8b", "student DPO vs base (no harness)"),
    ("student_opd_qwen3_8b", "student_dpo_qwen3_8b", "student OPD vs DPO"),
    ("harness_alert_x32b_qwen3_8b", "student_opd_qwen3_8b", "teacher vs OPD student (remaining gap)"),
    ("student_opd_alert_x32b", "harness_alert_x32b_qwen3_8b", "OPD student + harness vs teacher"),
]
CLAIM_TAG = {"conflict_claim": "claim_conflict", "compatible_claim": "claim_compatible"}


def ngrams(s, n=3):
    s = re.sub(r"\s+", " ", s.strip().lower())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jacc(a, b):
    A, B = ngrams(a), ngrams(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def boot_ci(xs, n=4000, seed=0):
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def fmt(m, ci=None, pct=True):
    if m is None:
        return "–"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f(m) + (f" [{f(ci[0])}, {f(ci[1])}]" if ci else "")


def main(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    chars = load_characters(cfg["characters"])
    T = [json.loads(l) for l in (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    J = [json.loads(l) for l in (run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judges = sorted({r["judge"] for r in J})
    labels = defaultdict(dict)  # key -> judge -> label
    for r in J:
        if r["kind"] == "memory" and "_error" not in r["result"]:
            labels[r["key"]][r["judge"]] = r["result"]["label"]

    # per-dialogue records
    rec = []  # dict(system, character, seed, type, noticed, confab, accepted, reply)
    for d in T:
        gt = d.get("gold_type")
        if gt not in CLAIM_TAG:
            continue
        key = f"{d['system']}|{d['character']}|{d['probe']}|{d['seed']}"
        labs = list(labels.get(key, {}).values())
        if not labs:
            continue
        turn = next(t for t in d["turns"] if t["tag"] == CLAIM_TAG[gt])
        rec.append({"system": d["system"], "character": d["character"], "seed": d["seed"], "type": gt,
                    "noticed": mean(l == "noticed" for l in labs),
                    "confab": mean(l == "confabulated" for l in labs),
                    "accepted": mean(l == "accepted" for l in labs),
                    "reply": turn["reply"]})
    systems = [s["id"] for s in cfg["systems"] if any(r["system"] == s["id"] for r in rec)]
    by = defaultdict(list)
    for r in rec:
        by[(r["system"], r["type"])].append(r)

    L = [f"# Claims analysis — {run_dir.name}", "",
         f"{len(chars)} characters · seeds {cfg.get('seeds')} · judges: {', '.join(judges)}. "
         "Rates average the judges per dialogue; CIs are 95% bootstrap over dialogues.", ""]

    # 1. per-system table
    L += ["## 1. Per-system rates", "",
          "| system | n | conflict: noticed ↑ | conflict: confabulated ↓ | compatible: accepted ↑ | claim-reply similarity across seeds ↓ | noticed-reply similarity ↓ |",
          "|---|---|---|---|---|---|---|"]
    for s in systems:
        c, k = by[(s, "conflict_claim")], by[(s, "compatible_claim")]
        sim_all, sim_noticed = [], []
        groups = defaultdict(list)
        for r in c:
            groups[r["character"]].append(r)
        for rs in groups.values():
            for a, b in combinations(rs, 2):
                sim_all.append(jacc(a["reply"], b["reply"]))
            ns = [r for r in rs if r["noticed"] >= 0.5]
            for a, b in combinations(ns, 2):
                sim_noticed.append(jacc(a["reply"], b["reply"]))
        row = [s, str(len(c)),
               fmt(mean(r["noticed"] for r in c), boot_ci([r["noticed"] for r in c])) if c else "–",
               fmt(mean(r["confab"] for r in c), boot_ci([r["confab"] for r in c])) if c else "–",
               fmt(mean(r["accepted"] for r in k), boot_ci([r["accepted"] for r in k])) if k else "–",
               fmt(mean(sim_all), pct=False) if sim_all else "–",
               (fmt(mean(sim_noticed), pct=False) + f" (pairs={len(sim_noticed)})") if sim_noticed else "–"]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # 2. paired comparisons
    L += ["## 2. Paired comparisons on conflicting claims (same character & seed)", "",
          "| question | treatment − reference: noticed | p (sign-flip) | confabulated | p |", "|---|---|---|---|---|"]
    idx = {(r["system"], r["character"], r["seed"]): r for r in rec if r["type"] == "conflict_claim"}
    rng = random.Random(1)
    for a, b, q in PAIRS:
        units = [(idx[(a, c, sd)], idx[(b, c, sd)]) for (s, c, sd) in idx if s == a and (b, c, sd) in idx]
        if len(units) < 3:
            continue
        cells = []
        for m in ("noticed", "confab"):
            diffs = [x[m] - y[m] for x, y in units]
            obs = mean(diffs)
            ci = boot_ci(diffs)
            perm = [abs(mean(dd if rng.random() < 0.5 else -dd for dd in diffs)) for _ in range(4000)]
            p = (sum(v >= abs(obs) - 1e-12 for v in perm) + 1) / (len(perm) + 1)
            sign = "+" if obs >= 0 else "−"
            cells += [f"{sign}{abs(100 * obs):.0f} pts [{100 * ci[0]:+.0f}, {100 * ci[1]:+.0f}] (n={len(units)})",
                      f"{p:.3f}"]
        L.append(f"| {q} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. per-character matrix
    L += ["## 3. Conflicting claim noticed, per character", "",
          "| character | lang | reminisce | skeptic | " + " | ".join(systems) + " |",
          "|---|---|---|---|" + "---|" * len(systems)]
    for cid, card in chars.items():
        tags = card.get("persona_tags", {})
        cells = []
        for s in systems:
            rs = [r["noticed"] for r in by[(s, "conflict_claim")] if r["character"] == cid]
            cells.append(f"{100 * mean(rs):.0f}%" if rs else "–")
        L.append(f"| {cid} | {card['lang']} | {tags.get('reminisce', '–')} | {tags.get('skeptic', '–')} | "
                 + " | ".join(cells) + " |")
    L.append("")

    # 4. persona vs behaviour
    L += ["## 4. Persona vs confabulation (Spearman across characters)", "",
          "| system | ρ(reminisce, confabulated) | ρ(skeptic, noticed) |", "|---|---|---|"]
    for s in systems:
        xs_r, ys_c, xs_s, ys_n = [], [], [], []
        for cid, card in chars.items():
            tags = card.get("persona_tags")
            rs = [r for r in by[(s, "conflict_claim")] if r["character"] == cid]
            if not tags or not rs:
                continue
            xs_r.append(tags["reminisce"]); ys_c.append(mean(r["confab"] for r in rs))
            xs_s.append(tags["skeptic"]); ys_n.append(mean(r["noticed"] for r in rs))
        rho1, rho2 = spearman(xs_r, ys_c), spearman(xs_s, ys_n)
        L.append(f"| {s} | {'–' if rho1 is None else f'{rho1:+.2f}'} | {'–' if rho2 is None else f'{rho2:+.2f}'} |")
    L += ["", f"(n = {len(chars)} characters; with this few characters, |ρ| < ~0.55 is not distinguishable from 0.)", ""]

    # 5. judge agreement
    if len(judges) > 1:
        L += ["## 5. Judge agreement on claim labels", ""]
        for gt in CLAIM_TAG:
            units, same = [], []
            for d in T:
                if d.get("gold_type") != gt:
                    continue
                key = f"{d['system']}|{d['character']}|{d['probe']}|{d['seed']}"
                ls = [labels.get(key, {}).get(j) for j in judges]
                if all(ls):
                    units.append(ls)
                    same.append(len(set(ls)) == 1)
            a = krippendorff_alpha(units, "nominal")
            key_lab = "noticed" if gt == "conflict_claim" else "accepted"
            ab = krippendorff_alpha([[int(x == key_lab) for x in u] for u in units], "nominal")
            L.append(f"- {gt}: exact agreement {100 * mean(same):.0f}%, α (all labels) = "
                     f"{'–' if a is None else f'{a:.2f}'}, α (binary '{key_lab}') = "
                     f"{'–' if ab is None else f'{ab:.2f}'} (n={len(units)})" if units else f"- {gt}: –")
        L.append("")

    text = "\n".join(L)
    (run_dir / "claims_analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
