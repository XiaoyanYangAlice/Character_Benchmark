"""Compare annotators (Codex, humans) with the LLM judges on a blind annotation round.

  PYTHONPATH=src python scripts/analyze_annotations.py \
      --claims-run RESULTS/claims_v1 --claims-key annotation/round1/keys/claims_key.json \
      --claims codex_claims.csv [human_claims.csv ...] \
      --state-run RESULTS/local_v0 --state-key annotation/round1/keys/state_key.json \
      --state codex_state_persona.csv [human_state_persona.csv ...] \
      --out annotation/round1/analysis [--human-sheet 40]

Rater names come from the file names (codex_claims.csv -> "codex"). Judges are read from each
run's judgments.jsonl. Outputs analysis.md and, with --human-sheet N, a blind sheet of the N
items most worth a human look (largest rater disagreement + a random share) plus its key.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

from lifelike_bench.agreement import krippendorff_alpha, spearman

SCORE_FIELDS = {"state": ["tension", "energy", "closeness"],
                "persona": ["persona_consistency", "naturalness"]}
BINARY = {"conflict_claim": "noticed", "compatible_claim": "accepted"}


def rater_name(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[_-]?(claims|state_persona|state|persona)$", "", s)
    return s or path.stem


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def judge_results(run: Path) -> dict:
    """(dialogue key, kind, turn) -> judge -> result"""
    out = defaultdict(dict)
    for r in read_jsonl(run / "judgments.jsonl"):
        if "_error" not in r["result"]:
            out[(r["key"], r["kind"], r["turn"])][r["judge"]] = r["result"]
    return out


def read_sheet(p: Path) -> dict:
    with p.open(encoding="utf-8-sig") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def as_int(v):
    try:
        x = int(float(str(v).strip()))
        return x if 1 <= x <= 5 else None
    except (TypeError, ValueError):
        return None


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- claims
def claims_section(run, key_path, files, L, disagreements):
    key = json.loads(Path(key_path).read_text())
    judged = judge_results(Path(run))
    raters = {}  # name -> item -> label
    for f in files:
        rows = read_sheet(Path(f))
        raters[rater_name(Path(f))] = {i: (r.get("label") or "").strip() for i, r in rows.items()}
        if "in_character" in next(iter(rows.values()), {}):
            raters[rater_name(Path(f))]["__ic__"] = {i: (r.get("in_character") or "").strip().lower()
                                                     for i, r in rows.items()}
    sheet = read_sheet(Path(key_path).parent.parent / "claims_sheet.csv") \
        if (Path(key_path).parent.parent / "claims_sheet.csv").exists() else {}
    ptype = {}
    for iid, k in key.items():
        ptype[iid] = sheet.get(iid, {}).get("probe_type") or \
            ("conflict_claim" if "conflict" in k["key"].split("|")[2] else "compatible_claim")
    judges = sorted({j for k in key.values() for j in judged.get((k["key"], k["kind"], k["turn"]), {})})
    labels = {}  # rater -> item -> label
    for j in judges:
        labels[j] = {iid: judged.get((k["key"], k["kind"], k["turn"]), {}).get(j, {}).get("label")
                     for iid, k in key.items()}
    ic = {}
    for name, d in raters.items():
        ic[name] = d.pop("__ic__", {})
        labels[name] = d
    names = list(labels)

    L += ["## Claims (conflicting / compatible shared-past claims)", "",
          f"Items: {len(key)} · raters: {', '.join(names)}", ""]
    # validity of annotator labels
    allowed = {"conflict_claim": {"noticed", "accepted", "confabulated", "ignored"},
               "compatible_claim": {"accepted", "asked", "declined", "contradicted_setting"}}
    for n in raters:
        bad = [i for i in key if labels[n].get(i) not in allowed[ptype[i]]]
        if bad:
            L.append(f"- ⚠ {n}: {len(bad)} rows with missing/invalid labels (e.g. {', '.join(bad[:5])}); excluded pairwise")
    L.append("")

    for pt in ("conflict_claim", "compatible_claim"):
        items = [i for i in key if ptype[i] == pt]
        L += [f"### {pt} (n={len(items)})", "",
              "| pair | exact agreement | α (all labels) | α (binary '" + BINARY[pt] + "') | n |",
              "|---|---|---|---|---|"]
        for a, b in combinations(names, 2):
            pairs = [(labels[a].get(i), labels[b].get(i)) for i in items]
            pairs = [p for p in pairs if p[0] in allowed[pt] and p[1] in allowed[pt]]
            if not pairs:
                continue
            ex = mean(x == y for x, y in pairs)
            al = krippendorff_alpha([list(p) for p in pairs], "nominal")
            ab = krippendorff_alpha([[int(x == BINARY[pt]), int(y == BINARY[pt])] for x, y in pairs], "nominal")
            L.append(f"| {a} – {b} | {100 * ex:.0f}% | {fmt(al)} | {fmt(ab)} | {len(pairs)} |")
        dom = Counter(labels[names[0]].get(i) for i in items).most_common(1)
        if dom and dom[0][1] / max(1, len(items)) > 0.9:
            L += ["", f"Note: one label dominates (> 90% '{dom[0][0]}'), so α is unstable here — read exact agreement instead."]
        L += ["", "Label distribution per rater:", ""]
        L.append("| rater | " + " | ".join(sorted(allowed[pt])) + " |")
        L.append("|---|" + "---|" * len(allowed[pt]))
        for n in names:
            c = Counter(labels[n].get(i) for i in items)
            L.append(f"| {n} | " + " | ".join(str(c.get(x, 0)) for x in sorted(allowed[pt])) + " |")
        L.append("")
        # confusion: each annotator vs each judge
        for n in raters:
            for j in judges:
                cm = Counter((labels[j].get(i), labels[n].get(i)) for i in items
                             if labels[j].get(i) in allowed[pt] and labels[n].get(i) in allowed[pt])
                if not cm:
                    continue
                labs = sorted(allowed[pt])
                L += [f"Confusion ({j} rows × {n} columns):", "",
                      "| | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
                for x in labs:
                    L.append(f"| {x} | " + " | ".join(str(cm.get((x, y), 0)) for y in labs) + " |")
                L.append("")
        for i in items:
            vals = [labels[n].get(i) for n in names if labels[n].get(i) in allowed[pt]]
            if vals:
                disagreements.append((len(set(vals)) - 1 + (0.5 if BINARY[pt] in vals and len(set(vals)) > 1 else 0),
                                      "claims", i))

    # in_character
    for n, d in ic.items():
        if not d:
            continue
        L += [f"### in_character × label ({n})", "", "| probe | label | yes | no |", "|---|---|---|---|"]
        for pt in ("conflict_claim", "compatible_claim"):
            c = Counter((labels[n].get(i), d.get(i)) for i in key if ptype[i] == pt)
            for lab in sorted(allowed[pt]):
                if c.get((lab, "yes"), 0) + c.get((lab, "no"), 0):
                    L.append(f"| {pt} | {lab} | {c.get((lab, 'yes'), 0)} | {c.get((lab, 'no'), 0)} |")
        L.append("")
    return key


# ---------------------------------------------------------------- state / persona
def scores_section(run, key_path, files, L, disagreements):
    key = json.loads(Path(key_path).read_text())
    judged = judge_results(Path(run))
    raters = {rater_name(Path(f)): read_sheet(Path(f)) for f in files}
    judges = sorted({j for k in key.values() for j in judged.get((k["key"], k["kind"], k["turn"]), {})})
    L += ["## State and persona scores (1–5)", "", f"Items: {len(key)} · raters: {', '.join(judges + list(raters))}", ""]
    for kind, fields in SCORE_FIELDS.items():
        items = [i for i, k in key.items() if k["kind"] == kind]
        if not items:
            continue
        for fld in fields:
            vals = {}
            for j in judges:
                vals[j] = {i: judged.get((key[i]["key"], kind, key[i]["turn"]), {}).get(j, {}).get(fld) for i in items}
            for n, rows in raters.items():
                vals[n] = {i: as_int(rows.get(i, {}).get(fld)) for i in items}
            L += [f"### {kind}.{fld} (n={len(items)})", "",
                  "| pair | α (ordinal) | Spearman | within ±1 | mean A | mean B |", "|---|---|---|---|---|---|"]
            for a, b in combinations(vals, 2):
                pr = [(vals[a][i], vals[b][i]) for i in items if vals[a][i] is not None and vals[b][i] is not None]
                if len(pr) < 3:
                    continue
                al = krippendorff_alpha([list(p) for p in pr], "ordinal")
                rho = spearman([p[0] for p in pr], [p[1] for p in pr])
                w1 = mean(abs(x - y) <= 1 for x, y in pr)
                L.append(f"| {a} – {b} | {fmt(al)} | {fmt(rho)} | {100 * w1:.0f}% | "
                         f"{mean(p[0] for p in pr):.2f} | {mean(p[1] for p in pr):.2f} |")
            L += ["", "Score distribution:", "", "| rater | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
            for n, v in vals.items():
                c = Counter(x for x in v.values() if x is not None)
                L.append(f"| {n} | " + " | ".join(str(c.get(s, 0)) for s in range(1, 6)) + " |")
            L.append("")
            for i in items:
                xs = [v[i] for v in vals.values() if v[i] is not None]
                if len(xs) > 1:
                    disagreements.append(((max(xs) - min(xs)) / 2, "state", i))
    return key


# ---------------------------------------------------------------- human sheet
def write_human_sheet(n, disagreements, claims_key, state_key, claims_sheet, state_sheet, out: Path, seed=0):
    rng = random.Random(seed)
    by = sorted(disagreements, key=lambda x: -x[0])
    seen, top = set(), []
    for score, src, iid in by:
        if (src, iid) not in seen and len(top) < int(n * 0.75):
            seen.add((src, iid)); top.append((src, iid))
    rest = [(s, i) for _, s, i in disagreements if (s, i) not in seen]
    rng.shuffle(rest)
    for s, i in rest:
        if len(top) >= n:
            break
        if (s, i) not in seen:
            seen.add((s, i)); top.append((s, i))
    rng.shuffle(top)
    key = {}
    with (out / "human_sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "task", "label_options", "character", "material", "label", "in_character",
                    "tension", "energy", "closeness", "persona_consistency", "naturalness", "comment"])
        for k, (src, iid) in enumerate(top):
            hid = f"H{k:03d}"
            if src == "claims":
                r = claims_sheet[iid]
                key[hid] = {"source": "claims", "item_id": iid, **claims_key[iid]}
                w.writerow([hid, r["probe_type"], r["label_options"], r.get("character", ""), r["material"],
                            "", "", "", "", "", "", "", ""])
            else:
                r = state_sheet[iid]
                key[hid] = {"source": "state", "item_id": iid, **state_key[iid]}
                w.writerow([hid, r["task"], "1-5 each", "", r["material"], "", "", "", "", "", "", "", ""])
    (out / "human_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return len(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-run"); ap.add_argument("--claims-key"); ap.add_argument("--claims", nargs="*", default=[])
    ap.add_argument("--state-run"); ap.add_argument("--state-key"); ap.add_argument("--state", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--human-sheet", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = ["# Annotation analysis", "",
         "Agreement between LLM judges and annotators on the blind round. α ≥ 0.6: usable as a headline "
         "measure; 0.4–0.6: report as exploratory; < 0.4: do not rely on it.", ""]
    dis = []
    ck = sk = None
    if a.claims_run and a.claims:
        ck = claims_section(a.claims_run, a.claims_key, a.claims, L, dis)
    if a.state_run and a.state:
        sk = scores_section(a.state_run, a.state_key, a.state, L, dis)
    if a.human_sheet and ck is not None:
        base = Path(a.claims_key).parent.parent
        cs = read_sheet(base / "claims_sheet.csv")
        ss = read_sheet(base / "state_persona_sheet.csv") if (base / "state_persona_sheet.csv").exists() else {}
        n = write_human_sheet(a.human_sheet, dis, ck, sk or {}, cs, ss, out)
        L += [f"Wrote human_sheet.csv with {n} items (≈75% largest disagreement, rest random) and human_key.json.", ""]
    text = "\n".join(L)
    (out / "analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
