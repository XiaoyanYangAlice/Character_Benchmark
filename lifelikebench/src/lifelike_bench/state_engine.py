"""Inner-state engine: one Ornstein–Uhlenbeck process per state dimension, with a
time-of-day baseline (rhythm), event impulses and bounded Gaussian noise.

    x <- mu(t) + (x - mu(t)) * exp(-theta * dt)      # persistence + recovery
         + sum_k trigger[k] * event_k               # event impulse
         + sigma * sqrt(min(dt,1)) * N(0,1)         # noise
dt is in hours. The clock is passed in explicitly so runs are reproducible.
"""
from __future__ import annotations

import math
import random


def baseline(dim: dict, hour: float) -> float:
    r = dim.get("rhythm")
    mu = dim["baseline"]
    if r:
        mu += r["amplitude"] * math.cos(2 * math.pi * (hour - r["peak_hour"]) / 24)
    return mu


class StateEngine:
    def __init__(self, card: dict, start_hour: float = 15.0, seed: int = 0, noise: bool = True):
        self.dims = card["dimensions"]
        self.state = {k: d["baseline"] for k, d in self.dims.items()}
        self.clock_h = start_hour          # hours since day 0 00:00
        self.rng = random.Random(seed)
        self.noise = noise
        self.log: list[dict] = []

    def step(self, events: dict[str, float], elapsed_min: float) -> dict:
        dt = max(elapsed_min / 60.0, 0.05)
        self.clock_h += elapsed_min / 60.0
        hour = self.clock_h % 24
        for name, d in self.dims.items():
            x, mu = self.state[name], baseline(d, hour)
            x = mu + (x - mu) * math.exp(-d.get("recovery", 0.5) * dt)
            x += sum(d.get("triggers", {}).get(k, 0.0) * v for k, v in events.items())
            if self.noise:
                x += d.get("volatility", 0.05) * math.sqrt(min(dt, 1.0)) * self.rng.gauss(0, 1)
            lo, hi = d.get("range", [0.0, 1.0])
            self.state[name] = max(lo, min(hi, x))
        self.log.append({"events": events, "elapsed_min": elapsed_min,
                         "state": {k: round(v, 3) for k, v in self.state.items()}})
        return dict(self.state)

    def directives(self) -> str:
        lines = []
        for name, d in self.dims.items():
            lo, hi = d.get("range", [0.0, 1.0])
            r = (self.state[name] - lo) / (hi - lo)
            lv = "low" if r < 0.33 else ("high" if r > 0.67 else "mid")
            cue = d.get("behavior", {}).get(lv)
            if cue:
                lines.append(f"- {d.get('label', name)} ({lv}): {cue}")
        return "\n".join(lines)
