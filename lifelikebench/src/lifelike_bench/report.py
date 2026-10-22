"""Markdown report from scored metrics."""
from __future__ import annotations


def _f(s, pct=False):
    if s is None:
        return "–"
    if isinstance(s, dict):
        m, ci = s.get("mean"), s.get("ci")
        if m is None:
            return "–"
        if pct:
            return f"{100*m:.0f}%" + (f" [{100*ci[0]:.0f}, {100*ci[1]:.0f}]" if ci else "")
        return f"{m:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
    return f"{s:.2f}" if isinstance(s, float) else str(s)


def render(metrics: dict, agreement: dict, meta: dict, by_judge: dict | None = None) -> str:
    L = [f"# LifelikeBench report — {meta.get('run_name', '')}", ""]
    L += [f"Characters: {meta.get('n_characters')} · probes: {meta.get('n_probes')} · "
          f"seeds: {meta.get('seeds')} · judges: {', '.join(meta.get('judges', []))}",
          "Values: mean [95% bootstrap CI]. ↑ higher is better, ↓ lower is better.", ""]
    systems = list(metrics)

    def table(title, rows):
        L.append(f"## {title}")
        L.append("| metric | " + " | ".join(f"{s} ({metrics[s]['condition']})" for s in systems) + " |")
        L.append("|---|" + "---|" * len(systems))
        for name, get, pct in rows:
            L.append(f"| {name} | " + " | ".join(_f(get(metrics[s]), pct) for s in systems) + " |")
        L.append("")

    table("D1 · Affect dynamics", [
        ("score ↑", lambda m: m["D1"]["score"], False),
        ("reactivity ↑", lambda m: m["D1"]["reactivity"], True),
        ("inertia ↑", lambda m: m["D1"]["inertia"], True),
        ("recovery ↑", lambda m: m["D1"]["recovery"], True),
        ("no overshoot ↑", lambda m: m["D1"]["no_overshoot"], True)])
    table("D2 · Persona stability", [
        ("persona consistency ↑", lambda m: m["D2"]["persona_consistency"], False),
        ("…under pressure ↑", lambda m: m["D2"]["persona_under_pressure"], False),
        ("assistant tone (judge) ↓", lambda m: m["D2"]["assistant_tone_rate"], True),
        ("assistant leak (regex) ↓", lambda m: m["D2"]["regex_leak_rate"], True),
        ("explicit state report ↓", lambda m: m["D2"]["explicit_state_report_rate"], True)])
    table("D3 · Memory fidelity", [
        ("recall ↑", lambda m: m["D3"]["recall"]["score"], False),
        ("contradiction noticed ↑", lambda m: m["D3"]["contradict"]["score"], False),
        ("conflicting claim noticed ↑", lambda m: m["D3"]["conflict_claim"]["score"], False),
        ("compatible claim accepted ↑", lambda m: m["D3"]["compatible_claim"]["score"], False),
        ("user canon kept ↑", lambda m: m["D3"]["canon"]["score"], False),
        ("confabulation rate (conflicting claims) ↓", lambda m: m["D3"]["confabulation_rate"], False)])
    table("D4 · Non-templatedness", [
        ("self-similarity across seeds ↓", lambda m: m["D4"]["self_similarity"], False),
        ("voice similarity across characters ↓", lambda m: m["D4"]["voice_similarity"], False),
        ("distinct-2 ↑", lambda m: m["D4"]["distinct2"], False)])
    table("D5 · Naturalness", [("naturalness ↑", lambda m: m["D5"]["naturalness"], False)])

    L.append("## D1 trajectories (target dimension, signed, relative to warm-up baseline)")
    for s in systems:
        for ev, tr in sorted(metrics[s]["D1"]["trajectories"].items()):
            tags = metrics[s]["D1"]["trajectory_tags"][ev]
            cells = " → ".join(f"{t}:{'–' if v is None else f'{v:+.1f}'}" for t, v in zip(tags, tr))
            L.append(f"- **{s}** · {ev}: {cells}")
    L.append("")
    if by_judge and len(by_judge) > 1:
        L.append("## Headline metrics per judge (check for self-preference: a judge scoring its own family)")
        L.append("| system | judge | D1 score | persona | naturalness | recall | conflict noticed | canon kept |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in systems:
            for j, mj in by_judge.items():
                x = mj.get(s)
                if not x:
                    continue
                L.append(f"| {s} | {j} | {_f(x['D1']['score'])} | {_f(x['D2']['persona_consistency'])} | "
                         f"{_f(x['D5']['naturalness'])} | {_f(x['D3']['recall']['score'])} | "
                         f"{_f(x['D3']['conflict_claim']['score'])} | {_f(x['D3']['canon']['score'])} |")
        L.append("")
    L.append("## Inter-judge agreement (Krippendorff's α)")
    if "alpha" in agreement:
        for k, v in agreement["alpha"].items():
            L.append(f"- {k}: {_f(v)}")
    else:
        L.append(f"- {agreement.get('note')}")
    L.append("")
    return "\n".join(L)
