#!/usr/bin/env python3
"""
state_engine.py — 角色内部状态引擎（lifelike-roleplay skill 的可选组件）

模型：每个状态维度是一个带时间基线的 Ornstein–Uhlenbeck 过程
    x_{t+dt} = x_t + θ·(μ(t) − x_t)·dt + Σ_k B[k]·e_k + σ·√dt·ε,   ε ~ N(0, 1)
  - 惯性/延续：x_t 本身
  - 向基线恢复：θ·(μ(t) − x_t)·dt          （θ 越大恢复越快）
  - 事件影响：B[k]·e_k                     （e_k 为本轮评估出的事件强度，0–1）
  - 有界随机扰动：σ·√dt·ε                  （真正的高斯噪声，模型自己“想”不出来）
  - 周期：μ(t) = μ0 + A·cos(2π(hour − peak)/24)，周期来自时间，不来自噪声

用法：
  python state_engine.py init  card.json  state.json
  python state_engine.py step  card.json  state.json --events '{"conflict":0.7}' [--minutes 5]
  python state_engine.py show  card.json  state.json
不传 --minutes 时按 state.json 中记录的上次时间与当前时间之差计算（真实流逝的时间）。
"""
import argparse, json, math, random, sys
from datetime import datetime

def now():
    return datetime.now()

def baseline(dim, hour):
    mu0 = dim["baseline"]
    rhythm = dim.get("rhythm")  # {"amplitude": 0.2, "peak_hour": 15}
    if not rhythm:
        return mu0
    return mu0 + rhythm["amplitude"] * math.cos(2 * math.pi * (hour - rhythm["peak_hour"]) / 24)

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def init(card):
    t = now()
    return {
        "last_update": t.isoformat(timespec="seconds"),
        "turn": 0,
        "state": {k: d["baseline"] for k, d in card["dimensions"].items()},
        "log": [],
    }

def step(card, st, events, minutes=None, seed=None):
    rng = random.Random(seed)
    t = now()
    if minutes is None:
        last = datetime.fromisoformat(st["last_update"])
        minutes = max(0.0, (t - last).total_seconds() / 60)
    # 时间单位：1.0 = 一小时；单轮对话即使间隔很短也至少走 0.05（约 3 分钟）
    dt = max(minutes / 60.0, 0.05)
    hour = t.hour + t.minute / 60
    changes = {}
    for name, d in card["dimensions"].items():
        x = st["state"][name]
        mu = baseline(d, hour)
        theta = d.get("recovery", 0.5)
        sigma = d.get("volatility", 0.05)
        lo, hi = d.get("range", [0.0, 1.0])
        # 指数形式的回归，dt 很大时不会过冲
        x_new = mu + (x - mu) * math.exp(-theta * dt)
        impulse = sum(d.get("triggers", {}).get(k, 0.0) * v for k, v in events.items())
        x_new += impulse
        x_new += sigma * math.sqrt(min(dt, 1.0)) * rng.gauss(0, 1)
        x_new = clip(x_new, lo, hi)
        changes[name] = {"from": round(x, 3), "to": round(x_new, 3),
                         "impulse": round(impulse, 3), "baseline_now": round(mu, 3)}
        st["state"][name] = x_new
    st["turn"] += 1
    st["last_update"] = t.isoformat(timespec="seconds")
    # provenance：每次变化都记下原因（GOLEM 式的“来源可追溯”）
    st["log"].append({"turn": st["turn"], "time": st["last_update"],
                      "elapsed_min": round(minutes, 1), "events": events})
    st["log"] = st["log"][-30:]
    return st, changes

def level(v, lo, hi):
    r = (v - lo) / (hi - lo) if hi > lo else 0.5
    return "low" if r < 0.33 else ("high" if r > 0.67 else "mid")

def render(card, st):
    """把数值翻译成行为指令。不要把数字交给生成模型——模型对数字的映射很不稳定。"""
    lines = []
    for name, d in card["dimensions"].items():
        lo, hi = d.get("range", [0.0, 1.0])
        lv = level(st["state"][name], lo, hi)
        cue = d.get("behavior", {}).get(lv)
        if cue:
            lines.append(f"- {d.get('label', name)}（{lv}）：{cue}")
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["init", "step", "show"])
    p.add_argument("card")
    p.add_argument("state")
    p.add_argument("--events", default="{}")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    card = json.load(open(a.card, encoding="utf-8"))
    if a.cmd == "init":
        st = init(card)
        changes = {}
    else:
        st = json.load(open(a.state, encoding="utf-8"))
        changes = {}
        if a.cmd == "step":
            st, changes = step(card, st, json.loads(a.events), a.minutes, a.seed)
    json.dump(st, open(a.state, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    out = {"turn": st["turn"],
           "state": {k: round(v, 3) for k, v in st["state"].items()},
           "changes": changes,
           "behavior_directives": render(card, st)}
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()

if __name__ == "__main__":
    main()
