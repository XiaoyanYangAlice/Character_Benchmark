"""Focused comparison table for selected systems, per judge and pooled.

  python scripts/compare.py RESULTS_DIR/local_v0/metrics.json [system ...]
Without system ids it prints the recency-experiment set.
"""
import json
import sys

DEFAULT = ["base_qwen3_8b", "harness_x32b_qwen3_8b", "harness_recency_x32b_qwen3_8b",
           "harness_alert_x32b_qwen3_8b", "base_qwen3_32b", "harness_qwen3_32b",
           "harness_llama31_8b"]
ROWS = [("conflict noticed ↑", lambda m: m["D3"]["conflict_claim"]["score"]["mean"]),
        ("confabulation ↓", lambda m: m["D3"]["confabulation_rate"]),
        ("compatible accepted ↑", lambda m: m["D3"]["compatible_claim"]["score"]["mean"]),
        ("canon kept ↑", lambda m: m["D3"]["canon"]["score"]["mean"]),
        ("recall ↑", lambda m: m["D3"]["recall"]["score"]["mean"]),
        ("D1 score ↑", lambda m: m["D1"]["score"]["mean"]),
        ("persona ↑", lambda m: m["D2"]["persona_consistency"]["mean"]),
        ("naturalness ↑", lambda m: m["D5"]["naturalness"]["mean"])]


def fmt(x):
    return "  –  " if x is None else f"{x:5.2f}"


M = json.load(open(sys.argv[1], encoding="utf-8"))
systems = sys.argv[2:] or DEFAULT
systems = [s for s in systems if s in M["systems"]]
missing = [s for s in (sys.argv[2:] or DEFAULT) if s not in M["systems"]]
views = [("pooled", M["systems"])] + list(M.get("by_judge", {}).items())
w = max(len(s) for s in systems) + 2
for name, data in views:
    print(f"\n=== {name} ===")
    print("metric".ljust(24) + "".join(s[:w - 2].rjust(w) for s in systems))
    for label, get in ROWS:
        vals = []
        for s in systems:
            try:
                vals.append(fmt(get(data[s])))
            except (KeyError, TypeError):
                vals.append(fmt(None))
        print(label.ljust(24) + "".join(v.rjust(w) for v in vals))
if missing:
    print("\n(not in this metrics.json yet:", ", ".join(missing), ")")
