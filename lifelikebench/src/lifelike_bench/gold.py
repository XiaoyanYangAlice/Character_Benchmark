"""Human gold set: export a blind, stratified sample for annotators; compare humans vs judges."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .agreement import krippendorff_alpha, spearman
from .data import load_characters
from .judges import card_summary, judge_inputs
from .runner import dialogue_key, read_jsonl

FIELDS = {"state": ["tension", "energy", "closeness"],
          "persona": ["persona_consistency", "naturalness"],
          "memory": ["label"]}


LABEL_OPTIONS = {
    "recall": "correct / partial / wrong / refused",
    "contradict": "noticed / adopted_new / ignored",
    "conflict_claim": "noticed / accepted / confabulated / ignored",
    "compatible_claim": "accepted / asked / declined / contradicted_setting",
    "canon": "consistent / partial / ignored / wrong",
}


def export_gold(out_dir: Path, characters, n: int = 250, seed: int = 0, kinds: list[str] | None = None,
                probes: list[str] | None = None, systems: list[str] | None = None,
                name: str = "default") -> Path:
    """Blind, stratified sample (by kind x system x character) of judged items.
    Writes gold/<name>/annotation_sheet.csv, key.json and GUIDELINES.md."""
    dialogues = {dialogue_key(d): d for d in read_jsonl(out_dir / "transcripts.jsonl")}
    judged = read_jsonl(out_dir / "judgments.jsonl")
    chars = load_characters(characters)
    items = sorted({(r["key"], r["kind"], r["turn"]) for r in judged})
    strata = defaultdict(list)
    for key, kind, t in items:
        d = dialogues[key]
        if kinds and kind not in kinds:
            continue
        if probes and d["probe"] not in probes:
            continue
        if systems and d["system"] not in systems:
            continue
        strata[(kind, d["system"], d["character"])].append((key, kind, t))
    if not strata:
        raise ValueError("no judged items match the filters")
    rng = random.Random(seed)
    # round-robin over strata so small strata are not starved and n is respected
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(strata.items())}
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(chosen) < n:
                chosen.append(pools[k].pop())
    rng.shuffle(chosen)

    gold_dir = out_dir / "gold" / name
    gold_dir.mkdir(parents=True, exist_ok=True)
    sheet = gold_dir / "annotation_sheet.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "task", "label_options", "material", "tension", "energy", "closeness",
                    "persona_consistency", "naturalness", "label", "comment"])
        key_map = {}
        for i, (key, kind, t) in enumerate(chosen):
            d = dialogues[key]
            iid = f"{name}-{i:04d}"
            key_map[iid] = {"key": key, "kind": kind, "turn": t}
            opts = LABEL_OPTIONS.get(d.get("gold_type"), "") if kind == "memory" else (
                "1-5 each" if kind == "state" else "1-5 each")
            w.writerow([iid, kind, opts, judge_inputs(kind, chars[d["character"]], d, t),
                        "", "", "", "", "", "", ""])
    (gold_dir / "key.json").write_text(json.dumps(key_map, indent=1), encoding="utf-8")
    (gold_dir / "GUIDELINES.md").write_text(GUIDELINES, encoding="utf-8")
    return sheet


def load_annotations(paths: list[Path]) -> dict[str, dict]:
    """annotator name (file stem) -> item_id -> {field: value}"""
    out = {}
    for p in paths:
        rows = {}
        with p.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                vals = {}
                for fs in FIELDS.values():
                    for k in fs:
                        v = (r.get(k) or "").strip()
                        if v:
                            vals[k] = v if k == "label" else float(v)
                rows[r["item_id"]] = vals
        out[p.stem] = rows
    return out


def compare(out_dir: Path, annotation_files: list[Path], name: str = "default") -> dict:
    key_map = json.loads((out_dir / "gold" / name / "key.json").read_text())
    ann = load_annotations(annotation_files)
    judged = defaultdict(dict)
    for r in read_jsonl(out_dir / "judgments.jsonl"):
        if "_error" not in r["result"]:
            judged[(r["key"], r["kind"], r["turn"])][r["judge"]] = r["result"]
    judges = sorted({j for v in judged.values() for j in v})
    humans = sorted(ann)
    report = {"humans": humans, "judges": judges, "fields": {}}
    for kind, fs in FIELDS.items():
        ids = [i for i, k in key_map.items() if k["kind"] == kind]
        for f in fs:
            level = "nominal" if f == "label" else "ordinal"
            hh = [[ann[h].get(i, {}).get(f) for h in humans] for i in ids]
            entry = {"n_items": len(ids),
                     "human_human_alpha": krippendorff_alpha(hh, level) if len(humans) > 1 else None}
            for j in judges:
                pairs_alpha, xs, ys = [], [], []
                for i in ids:
                    k = key_map[i]
                    jv = judged.get((k["key"], k["kind"], k["turn"]), {}).get(j, {}).get(f)
                    hv = [ann[h].get(i, {}).get(f) for h in humans]
                    hv = [v for v in hv if v is not None]
                    if jv is None or not hv:
                        continue
                    if f == "label":
                        pairs_alpha.append([jv, max(set(hv), key=hv.count)])
                    else:
                        hm = mean(hv)
                        pairs_alpha.append([jv, round(hm)])
                        xs.append(jv)
                        ys.append(hm)
                entry[f"{j}_vs_humans_alpha"] = krippendorff_alpha(pairs_alpha, level)
                if f != "label":
                    entry[f"{j}_vs_humans_spearman"] = spearman(xs, ys)
                    entry[f"{j}_within1"] = (mean(abs(x - y) <= 1 for x, y in zip(xs, ys))
                                             if xs else None)
            report["fields"][f"{kind}.{f}"] = entry
    return report


GUIDELINES = """# Annotation guidelines (LifelikeBench v0.1)

Each row has a `task` and the `material` the LLM judge saw. Fill ONLY the columns for that task.

## task = state  (fill tension, energy, closeness; 1–5)
You see only one line of a character's dialogue. Rate what the text SHOWS.
- tension: 1 relaxed … 3 neutral/slightly guarded … 5 clearly tense, hurt, defensive or hostile
- energy: 1 drained/minimal … 3 ordinary … 5 lively/animated
- closeness: 1 distant/cold … 3 ordinary friendly … 5 warm/intimate

## task = persona  (fill persona_consistency, naturalness; 1–5)
- persona_consistency: 5 unmistakably this character; 3 generic but not contradictory; 1 out of character / AI-assistant voice
- naturalness: would a skilled human writer plausibly write this line here? Penalise canned phrasing, over-explaining, therapy-speak, melodrama.

## task = memory  (fill label)
- recall: correct | partial | wrong | refused
- contradict: noticed | adopted_new | ignored
- conflict_claim (claim contradicts the setting): noticed | accepted | confabulated | ignored
- compatible_claim (claim fits the setting; accepting warmly is GOOD): accepted | asked | declined | contradicted_setting
- canon (user earlier added a setting fact): consistent | partial | ignored | wrong

Each row's `label_options` column lists the labels allowed for that row.
If a reply goes along with a conflicting claim but this looks like what THIS character would
plausibly do (e.g. a character described as forgetful or prone to tall tales), still choose the
label that describes the behaviour, and write `in-character` in `comment`.

Do not look at other annotators' sheets. Use `comment` for anything ambiguous.
Save your file as `<your_name>.csv`.
"""
