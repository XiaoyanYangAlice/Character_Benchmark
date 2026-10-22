"""Scoring: turns transcripts + judgments into per-system metrics for D1–D5."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from statistics import mean

from .agreement import krippendorff_alpha
from .runner import dialogue_key

DELTA = 1.0
LEAK_RE = re.compile(
    r"(as an ai|language model|i am an ai|i'm an ai|作为(一个)?(ai|人工智能|语言模型|AI)|"
    r"我是(一个)?(ai|AI|人工智能|语言模型))", re.I)
MD_RE = re.compile(r"^\s*([-*•]|\d+\.|#{1,3})\s", re.M)


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def summarize(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "ci": None, "n": 0}
    return {"mean": mean(xs), "ci": bootstrap_ci(xs), "n": len(xs)}


def _index_judgments(judgments):
    """(key, kind, turn) -> list of results across judges (errors dropped)."""
    idx = defaultdict(list)
    for r in judgments:
        if "_error" not in r["result"]:
            idx[(r["key"], r["kind"], r["turn"])].append((r["judge"], r["result"]))
    return idx


# ---------------- D1 ----------------
def d1_dialogue(d, idx):
    ev = d["event"]
    tags = [t["tag"] for t in d["turns"]]
    dim, sign = ev["target"], ev.get("sign", 1)
    traj = []
    for t in range(len(tags)):
        vals = [res[dim] for _, res in idx.get((dialogue_key(d), "state", t), [])]
        traj.append(sign * mean(vals) if vals else None)
    try:
        e = tags.index("event")
        skip = tags.index("timeskip")
    except ValueError:
        return None
    warm = [traj[i] for i, tg in enumerate(tags) if tg == "warmup" and traj[i] is not None]
    if not warm or traj[e] is None or traj[e + 1] is None or traj[skip] is None:
        return None
    b = mean(warm)
    react = traj[e] - b >= DELTA
    inertia = traj[e + 1] - b >= DELTA / 2
    recovery = react and (traj[skip] <= traj[e] - DELTA / 2)
    no_overshoot = traj[e] - b < 3
    return {"score": mean([react, inertia, recovery, no_overshoot]),
            "reactivity": react, "inertia": inertia, "recovery": recovery,
            "no_overshoot": no_overshoot, "baseline": b,
            "trajectory": [None if x is None else round(x - b, 2) for x in traj], "tags": tags,
            "event": ev["type"]}


# ---------------- D4 helpers ----------------
def ngrams(text: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", text.strip().lower())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: str, b: str) -> float:
    A, B = ngrams(a), ngrams(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def pairwise_mean(texts):
    pairs = [(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return mean(jaccard(a, b) for a, b in pairs) if pairs else None


def distinct2(texts):
    grams = [s[i:i + 2] for s in texts for i in range(len(s) - 1)]
    return len(set(grams)) / len(grams) if grams else None


# ---------------- main ----------------
def score(dialogues: list[dict], judgments: list[dict]) -> dict:
    idx = _index_judgments(judgments)
    by_sys = defaultdict(list)
    for d in dialogues:
        by_sys[d["system"]].append(d)

    out = {}
    for sysid, ds in by_sys.items():
        m = {"condition": ds[0]["condition"], "n_dialogues": len(ds)}

        # D1
        d1 = [r for d in ds if d["dimension"] == "D1" for r in [d1_dialogue(d, idx)] if r]
        m["D1"] = {"score": summarize([r["score"] for r in d1]),
                   **{k: summarize([float(r[k]) for r in d1])
                      for k in ("reactivity", "inertia", "recovery", "no_overshoot")}}
        traj = defaultdict(list)
        for r in d1:
            traj[r["event"]].append(r["trajectory"])
        m["D1"]["trajectories"] = {
            ev: [round(mean([t[i] for t in ts if t[i] is not None]), 2)
                 if any(t[i] is not None for t in ts) else None for i in range(len(ts[0]))]
            for ev, ts in traj.items()}
        m["D1"]["trajectory_tags"] = {r["event"]: r["tags"] for r in d1}

        # D2 + D5
        pers, pers_press, nat, explicit, asst, leak = [], [], [], [], [], []
        for d in ds:
            dp = []
            for t, turn in enumerate(d["turns"]):
                res = [r for _, r in idx.get((dialogue_key(d), "persona", t), [])]
                leak.append(float(bool(LEAK_RE.search(turn["reply"]) or
                                       len(MD_RE.findall(turn["reply"])) >= 2)))
                if not res:
                    continue
                pc = mean(r["persona_consistency"] for r in res)
                dp.append(pc)
                if turn["tag"] == "pressure":
                    pers_press.append(pc)
                nat.append(mean(r["naturalness"] for r in res))
                explicit.append(mean(float(bool(r.get("explicit_state_report"))) for r in res))
                asst.append(mean(float(bool(r.get("assistant_tone"))) for r in res))
            if dp:
                pers.append(mean(dp))
        m["D2"] = {"persona_consistency": summarize(pers),
                   "persona_under_pressure": summarize(pers_press),
                   "assistant_tone_rate": summarize(asst),
                   "regex_leak_rate": summarize(leak),
                   "explicit_state_report_rate": summarize(explicit)}
        m["D5"] = {"naturalness": summarize(nat)}

        # D3
        d3 = {}
        good = {"recall": {"correct": 1, "partial": 0.5},
                "contradict": {"noticed": 1},
                "conflict_claim": {"noticed": 1, "accepted": 0.25},
                "compatible_claim": {"accepted": 1, "asked": 0.75, "declined": 0.25},
                "canon": {"consistent": 1, "partial": 0.5}}
        for gt in good:
            labels, scores = defaultdict(int), []
            for d in ds:
                if d.get("gold_type") != gt:
                    continue
                for t, turn in enumerate(d["turns"]):
                    res = idx.get((dialogue_key(d), "memory", t), [])
                    for _, r in res:
                        labels[r["label"]] += 1
                    if res:
                        scores.append(mean(good[gt].get(r["label"], 0) for _, r in res))
            d3[gt] = {"score": summarize(scores), "labels": dict(labels)}
        conf = d3["conflict_claim"]["labels"]
        tot = sum(conf.values())
        d3["confabulation_rate"] = conf.get("confabulated", 0) / tot if tot else None
        m["D3"] = d3

        # D4
        groups = defaultdict(list)
        for d in ds:
            for t in d["turns"]:
                groups[(d["character"], d["probe"], t["idx"])].append(t["reply"])
        selfsim = [pairwise_mean(v) for v in groups.values() if len(v) > 1]
        voice = defaultdict(list)
        for d in ds:
            for t in d["turns"]:
                voice[(d["lang"], d["probe"], d["seed"], t["idx"])].append(t["reply"])
        vsim = [pairwise_mean(v) for v in voice.values() if len(v) > 1]
        m["D4"] = {"self_similarity": summarize([x for x in selfsim if x is not None]),
                   "voice_similarity": summarize([x for x in vsim if x is not None]),
                   "distinct2": distinct2([t["reply"] for d in ds for t in d["turns"]])}
        out[sysid] = m
    return out


def inter_judge(judgments: list[dict]) -> dict:
    """Krippendorff's alpha between LLM judges on every numeric/label field."""
    fields = {"state": ["tension", "energy", "closeness"],
              "persona": ["persona_consistency", "naturalness"], "memory": ["label"]}
    judges = sorted({r["judge"] for r in judgments})
    if len(judges) < 2:
        return {"note": "only one judge; inter-judge agreement not computed"}
    per = defaultdict(dict)
    for r in judgments:
        if "_error" not in r["result"]:
            per[(r["key"], r["kind"], r["turn"])][r["judge"]] = r["result"]
    out = {}
    for kind, fs in fields.items():
        for f in fs:
            units = [[res.get(j, {}).get(f) for j in judges]
                     for (k, kd, t), res in per.items() if kd == kind]
            level = "nominal" if f == "label" else "ordinal"
            out[f"{kind}.{f}"] = krippendorff_alpha(units, level)
    return {"judges": judges, "alpha": out}
