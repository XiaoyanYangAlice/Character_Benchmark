"""Generation and judging passes. Both are resumable: finished keys are skipped."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .backends import make_backend
from .data import instantiate, load_characters, load_probe_templates
from .judges import PROMPT_VERSION, items_for, judge_inputs, run_judge
from .systems import make_system

_lock = threading.Lock()


def _append(path: Path, row: dict):
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dialogue_key(d: dict) -> str:
    return f"{d['system']}|{d['character']}|{d['probe']}|{d['seed']}"


def backend_for(cfg: dict, entry: dict, role: str):
    """Resolve a system/judge entry to a Backend: `model:` (local registry) or explicit `backend:`."""
    from .serve import backend_spec
    if role == "helper":
        if "helper" in entry:
            return make_backend(entry["helper"])
        if "helper_model" in entry or "model" in entry:
            return make_backend(backend_spec(cfg, entry.get("helper_model", entry.get("model")), role))
        return None
    if "model" in entry:   # `adapter:` = LoRA module name on that model's server (students)
        return make_backend(backend_spec(cfg, entry["model"], role, entry.get("adapter")))
    return make_backend(entry["backend"])


def generate(cfg: dict, out_dir: Path, log=print, count_only: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transcripts.jsonl"
    done = {dialogue_key(d) for d in read_jsonl(path)}
    chars = load_characters(cfg["characters"], cfg.get("only_characters"))
    templates = load_probe_templates(cfg["probes"], cfg.get("only_probes"))
    gen = cfg.get("generation", {})

    jobs = []
    for s in cfg["systems"]:
        backend = backend_for(cfg, s, "character")
        helper = backend_for(cfg, s, "helper")
        for cid, card in chars.items():
            for tpl in templates:
                for seed in cfg.get("seeds", [0]):
                    key = f"{s['id']}|{cid}|{tpl['id']}|{seed}"
                    if key not in done:
                        jobs.append((s, backend, helper, card, tpl, seed))
    if count_only:
        return len(jobs)
    log(f"generate: {len(jobs)} dialogues to run ({len(done)} already done)")

    def run(job):
        s, backend, helper, card, tpl, seed = job
        probe = instantiate(tpl, card)
        system = make_system(s["condition"], card, backend, gen, seed, helper=helper,
                             skill_path=s.get("skill_path"))
        turns = []
        for t in probe.turns:
            reply = system.respond(t.text, t)
            row_t = {"idx": t.idx, "tag": t.tag, "user": t.text, "reply": reply,
                     "elapsed_min": t.elapsed_min,
                     "finish_reason": getattr(system, "last_finish", None)}
            if cfg.get("record_requests"):
                row_t["request"] = system.last_request   # exact (system, messages) the model saw
            turns.append(row_t)
        row = {"system": s["id"], "condition": s["condition"], "character": card["id"],
               "lang": probe.lang, "probe": probe.id, "dimension": probe.dimension,
               "event": probe.event, "gold": probe.gold, "gold_type": probe.gold_type,
               "seed": seed, "turns": turns, "trace": system.trace()}
        _append(path, row)
        return row

    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 8)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            fut.result()
            if i % 10 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)}")


def judge(cfg: dict, out_dir: Path, log=print, count_only: bool = False):
    dialogues = read_jsonl(out_dir / "transcripts.jsonl")
    path = out_dir / "judgments.jsonl"
    done = {(r["judge"], r["key"], r["kind"], r["turn"]) for r in read_jsonl(path)}
    chars = load_characters(cfg["characters"])

    jobs = []
    for j in cfg["judges"]:
        backend = backend_for(cfg, j, "judge")
        for d in dialogues:
            key = dialogue_key(d)
            for kind, t in items_for(d):
                if (j["id"], key, kind, t) not in done:
                    jobs.append((j["id"], backend, d, key, kind, t))
    if count_only:
        return len(jobs)
    log(f"judge: {len(jobs)} items to judge")

    def run(job):
        jid, backend, d, key, kind, t = job
        res = run_judge(backend, kind, judge_inputs(kind, chars[d["character"]], d, t))
        _append(path, {"judge": jid, "key": key, "system": d["system"], "character": d["character"],
                       "probe": d["probe"], "seed": d["seed"], "kind": kind, "turn": t,
                       "result": res, "prompt_version": PROMPT_VERSION})

    with ThreadPoolExecutor(max_workers=cfg.get("judge_concurrency", 8)) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
            fut.result()
            if i % 500 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)}")
