"""lifelike-bench CLI.

  lifelike-bench run       CONFIG          generate transcripts (resumable)
  lifelike-bench judge     CONFIG          run LLM judges (resumable)
  lifelike-bench score     CONFIG          metrics + report.md
  lifelike-bench all       CONFIG          run + judge + score (servers already running / APIs)
  lifelike-bench local     CONFIG          start local vLLM servers per phase, then run+judge+score
  lifelike-bench plan      CONFIG          show which models each local phase will start
  lifelike-bench export-gold CONFIG [-n 250]
  lifelike-bench gold-agreement CONFIG annotator1.csv annotator2.csv ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

from . import gold, metrics, report, runner, serve
from .data import load_characters, load_probe_templates


def _expand(obj):
    """Expand ${VAR} and ${VAR:-default} in config strings from the environment."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}",
                      lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    return obj


def load_config(path: str) -> dict:
    cfg = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    cfg.setdefault("run_name", Path(path).stem)
    cfg.setdefault("results_dir", "results")
    return cfg


def out_dir(cfg) -> Path:
    return Path(cfg["results_dir"]) / cfg["run_name"]


def do_score(cfg):
    od = out_dir(cfg)
    ds = runner.read_jsonl(od / "transcripts.jsonl")
    js = runner.read_jsonl(od / "judgments.jsonl")
    m = metrics.score(ds, js)
    ag = metrics.inter_judge(js)
    by_judge = {j["id"]: metrics.score(ds, [r for r in js if r["judge"] == j["id"]])
                for j in cfg["judges"]}
    meta = {"run_name": cfg["run_name"],
            "n_characters": len(load_characters(cfg["characters"], cfg.get("only_characters"))),
            "n_probes": len(load_probe_templates(cfg["probes"], cfg.get("only_probes"))),
            "seeds": cfg.get("seeds"), "judges": [j["id"] for j in cfg["judges"]]}
    (od / "metrics.json").write_text(json.dumps({"meta": meta, "systems": m, "inter_judge": ag,
                                                 "by_judge": by_judge},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "report.md").write_text(report.render(m, ag, meta, by_judge), encoding="utf-8")
    print(f"wrote {od/'report.md'}")


def main():
    p = argparse.ArgumentParser(prog="lifelike-bench")
    p.add_argument("cmd", choices=["run", "judge", "score", "all", "local", "plan",
                                   "export-gold", "gold-agreement"])
    p.add_argument("config")
    p.add_argument("files", nargs="*")
    p.add_argument("-n", type=int, default=250)
    p.add_argument("--systems", help="comma-separated system ids to restrict to")
    p.add_argument("--judges", help="comma-separated judge ids to restrict to")
    p.add_argument("--phases", default="generate,judge,score", help="for `local`")
    p.add_argument("--kinds", help="export-gold: comma-separated judge kinds (state,persona,memory)")
    p.add_argument("--probes", help="export-gold: comma-separated probe ids")
    p.add_argument("--name", default="default", help="export-gold / gold-agreement: sample name")
    a = p.parse_args()
    cfg = load_config(a.config)
    cfg = serve.subset(cfg, a.systems.split(",") if a.systems else None,
                       a.judges.split(",") if a.judges else None)
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)
    (od / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    if a.cmd in ("run", "all"):
        runner.generate(cfg, od)
    if a.cmd in ("judge", "all"):
        runner.judge(cfg, od)
    if a.cmd in ("score", "all"):
        do_score(cfg)
    if a.cmd == "plan":
        n = len(serve.detect_gpus(cfg)) or 1
        pl = serve.plan(cfg, n)
        print(f"GPUs available: {n}")
        for models, ids in pl["generate"]:
            print(f"[generate] start {list(models) or '(no local model)'} -> systems {ids}")
        for models, ids in pl["judge"]:
            print(f"[judge]    start {list(models) or '(no local model)'} -> judges {ids}")
    if a.cmd == "local":
        serve.run_local(cfg, od, do_score, phases=a.phases.split(","))
    if a.cmd == "export-gold":
        sheet = gold.export_gold(od, cfg["characters"], n=a.n,
                                 kinds=a.kinds.split(",") if a.kinds else None,
                                 probes=a.probes.split(",") if a.probes else None,
                                 systems=[s["id"] for s in cfg["systems"]] if a.systems else None,
                                 name=a.name)
        print(f"wrote {sheet}")
    if a.cmd == "gold-agreement":
        rep = gold.compare(od, [Path(f) for f in a.files], name=a.name)
        (od / "gold" / a.name / "agreement.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
