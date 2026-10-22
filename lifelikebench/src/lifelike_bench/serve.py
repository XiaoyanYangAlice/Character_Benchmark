"""Run the whole benchmark on local open-weight models (e.g. one SURF/Snellius GPU node).

Config: a `models:` registry, and systems/judges that refer to it by name.

    models:
      qwen3_8b:  {path: Qwen/Qwen3-8B, tp: 1, extra_body: {chat_template_kwargs: {enable_thinking: false}}}
      gemma3_27b: {path: google/gemma-3-27b-it, tp: 1}
    systems:
      - {id: base_qwen3_8b, condition: base, model: qwen3_8b}
      - {id: harness_qwen3_8b, condition: harness, model: qwen3_8b}   # helper_model defaults to model
    judges:
      - {id: judge_gemma, model: gemma3_27b}

`lifelike-bench local CONFIG` then works in phases. Each phase starts the vLLM servers it needs on
free GPUs, waits until they are healthy, does its work, and shuts them down:
  1. generation — one phase per distinct set of character/helper models
  2. judging    — judges packed onto the available GPUs, several at once if they fit
  3. scoring
Phases with nothing left to do are skipped (no server is started), so a job that hit its time
limit can simply be resubmitted.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

DEFAULT_COMMAND = ("vllm serve {path} --served-model-name {name} --host 127.0.0.1 --port {port} "
                   "--tensor-parallel-size {tp} --max-model-len {max_model_len} "
                   "--gpu-memory-utilization {gpu_memory_utilization} --seed 0 {extra_args}")


# ---------------- registry ----------------
def model_port(cfg: dict, name: str) -> int:
    if name not in cfg.get("models", {}):
        raise KeyError(f"model '{name}' is not in the config's models: registry")
    return int(cfg.get("serve", {}).get("base_port", 8100)) + list(cfg["models"]).index(name)


def backend_spec(cfg: dict, name: str, role: str, adapter: str | None = None) -> dict:
    """`adapter`: request a LoRA module served by this model's vLLM (--lora-modules adapter=path)."""
    m = cfg["models"][name]
    return {"type": "openai_compat", "model": adapter or name,
            "base_url": f"http://127.0.0.1:{model_port(cfg, name)}/v1",
            "extra_body": m.get("extra_body") or {},
            "structured_output": role == "judge" and m.get("structured_output", True),
            "timeout": m.get("timeout", 600)}


def system_models(s: dict) -> set[str]:
    if "model" not in s:
        return set()
    ms = {s["model"]}
    if s["condition"].startswith("harness"):
        ms.add(s.get("helper_model", s["model"]))
    return ms


# ---------------- servers ----------------
def detect_gpus(cfg: dict) -> list[str]:
    if cfg.get("serve", {}).get("gpus"):
        return [str(g) for g in cfg["serve"]["gpus"]]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return [str(i) for i, l in enumerate(out.splitlines()) if l.startswith("GPU")]
    except (OSError, subprocess.CalledProcessError):
        return []


class Servers:
    def __init__(self, cfg: dict, names: list[str], gpus: list[str], log_dir: Path, log=print):
        self.cfg, self.names, self.gpus, self.log_dir, self.log = cfg, names, gpus, log_dir, log
        self.procs: list[tuple[str, subprocess.Popen]] = []

    def __enter__(self):
        serve = self.cfg.get("serve", {})
        template = serve.get("command", DEFAULT_COMMAND)
        free = list(self.gpus)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for name in self.names:
                m = self.cfg["models"][name]
                tp = int(m.get("tp", 1))
                if tp > len(free):
                    raise RuntimeError(f"not enough GPUs for {name} (needs {tp}, {len(free)} free)")
                mine, free = free[:tp], free[tp:]
                cmd = template.format(
                    path=m["path"], name=name, port=model_port(self.cfg, name), tp=tp,
                    max_model_len=m.get("max_model_len", serve.get("max_model_len", 8192)),
                    gpu_memory_utilization=m.get("gpu_memory_utilization",
                                                 serve.get("gpu_memory_utilization", 0.90)),
                    extra_args=m.get("extra_args", ""))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(mine))
                logf = (self.log_dir / f"{name}.vllm.log").open("a")
                self.log(f"  starting {name} on GPU {','.join(mine)} :{model_port(self.cfg, name)}")
                p = subprocess.Popen(shlex.split(cmd), env=env, stdout=logf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                self.procs.append((name, p))
            timeout = int(serve.get("startup_timeout", 1800))
            for name, p in self.procs:
                self._wait(name, p, timeout)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _wait(self, name, p, timeout):
        url = f"http://127.0.0.1:{model_port(self.cfg, name)}/health"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                raise RuntimeError(f"server for {name} exited (code {p.returncode}); "
                                   f"see {self.log_dir / (name + '.vllm.log')}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        self.log(f"  {name} ready after {time.time() - t0:.0f}s")
                        return
            except OSError:
                pass
            time.sleep(5)
        raise TimeoutError(f"server for {name} not healthy after {timeout}s")

    def __exit__(self, *exc):
        for name, p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for name, p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        self.procs = []


# ---------------- phases ----------------
def plan(cfg: dict, n_gpus: int) -> dict:
    """Group systems by the set of local models they need; pack judges onto the GPUs."""
    gen: dict[tuple, list[str]] = {}
    for s in cfg["systems"]:
        gen.setdefault(tuple(sorted(system_models(s))), []).append(s["id"])
    tp = lambda n: int(cfg["models"][n].get("tp", 1))
    local_j = [j for j in cfg["judges"] if "model" in j]
    api_j = [j["id"] for j in cfg["judges"] if "model" not in j]
    batches, cur, used = [], [], 0
    for j in local_j:
        need = tp(j["model"])
        if cur and used + need > n_gpus:
            batches.append(cur)
            cur, used = [], 0
        cur.append(j)
        used += need
    if cur:
        batches.append(cur)
    judge_phases = [(tuple(sorted({j["model"] for j in b})), [j["id"] for j in b]) for b in batches]
    if api_j:
        judge_phases.append(((), api_j))
    return {"generate": list(gen.items()), "judge": judge_phases}


def subset(cfg: dict, systems: list[str] | None = None, judges: list[str] | None = None) -> dict:
    c = dict(cfg)
    if systems is not None:
        c["systems"] = [s for s in cfg["systems"] if s["id"] in systems]
    if judges is not None:
        c["judges"] = [j for j in cfg["judges"] if j["id"] in judges]
    return c


def run_local(cfg: dict, out_dir: Path, do_score, phases=("generate", "judge", "score"), log=print):
    from . import runner
    gpus = detect_gpus(cfg)
    p = plan(cfg, max(1, len(gpus)))
    log(f"GPUs: {gpus or 'none detected'}")
    logs = out_dir / "logs"

    if "generate" in phases:
        for models, sys_ids in p["generate"]:
            sub = subset(cfg, systems=sys_ids)
            if runner.generate(sub, out_dir, count_only=True) == 0:
                log(f"[generate] {sys_ids}: done, skipping")
                continue
            log(f"[generate] models={list(models)} systems={sys_ids}")
            with Servers(cfg, list(models), gpus, logs, log):
                runner.generate(sub, out_dir, log=log)

    if "judge" in phases:
        for models, judge_ids in p["judge"]:
            sub = subset(cfg, judges=judge_ids)
            if runner.judge(sub, out_dir, count_only=True) == 0:
                log(f"[judge] {judge_ids}: done, skipping")
                continue
            log(f"[judge] models={list(models)} judges={judge_ids}")
            with Servers(cfg, list(models), gpus, logs, log):
                runner.judge(sub, out_dir, log=log)

    if "score" in phases:
        do_score(cfg)
