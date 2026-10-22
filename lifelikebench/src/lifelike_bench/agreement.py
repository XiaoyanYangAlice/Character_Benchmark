"""Agreement statistics: Krippendorff's alpha (nominal / interval / ordinal), Spearman rho."""
from __future__ import annotations

import itertools
from collections import Counter


def krippendorff_alpha(units: list[list], level: str = "interval") -> float | None:
    """units: list of value lists (one list per item, one value per rater; None = missing)."""
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    n = len(values)
    if n < 2 or len(set(values)) < 2:
        return None

    if level == "nominal":
        delta = lambda a, b: 0.0 if a == b else 1.0
    elif level == "interval":
        delta = lambda a, b: float(a - b) ** 2
    elif level == "ordinal":
        freq = Counter(values)
        cats = sorted(freq)
        pos = {c: i for i, c in enumerate(cats)}

        def delta(a, b):
            lo, hi = sorted((pos[a], pos[b]))
            s = sum(freq[cats[g]] for g in range(lo, hi + 1)) - (freq[a] + freq[b]) / 2
            return s ** 2
    else:
        raise ValueError(level)

    d_o = 0.0
    for u in units:
        m = len(u)
        d_o += sum(delta(a, b) for a, b in itertools.permutations(u, 2)) / (m - 1)
    d_o /= n
    d_e = sum(delta(a, b) for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return None if d_e == 0 else 1 - d_o / d_e


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if vx == 0 or vy == 0 else cov / (vx * vy)
