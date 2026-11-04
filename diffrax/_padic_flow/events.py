from dataclasses import dataclass, field

import numpy as np

from .arithmetic import valuation
from .polynomial import degree, evaluate, parse_polynomial, to_dense
from .roots import Ball, vanishing_set


class PAdicEvent:
    """An event for a `PAdicFlow`, in the spirit of `diffrax.Event`: the condition
    `g(phi_t(x)) = 0` for a polynomial `g`, or several polynomials vanishing together.

    `p`-adic time is not ordered, so there is no first crossing to stop at. Instead
    the flow returns every time at which the condition holds to `depth` digits, as
    an `EventSet`: a finite union of balls in `Z_p`. The first non-negative integer
    time is then read off it.

    **Arguments:**

    - `condition`: a polynomial `{exponents: coefficient}` in the coordinates (for
        `d = 1` also `[c_0, c_1, ...]`), or a list of them, all of which must vanish.
        Coefficients are `p`-adic integers.
    - `depth`: the number of digits that must vanish; defaults to the precision of
        the flow.
    """

    def __init__(self, condition, depth: int | None = None):
        if isinstance(condition, dict) or not isinstance(condition, (list, tuple)):
            condition = [condition]
        elif condition and not isinstance(condition[0], (dict, list, tuple)):
            condition = [condition]
        if len(condition) == 0:
            raise ValueError("An event needs at least one condition.")
        self.condition = list(condition)
        self.depth = None if depth is None else int(depth)


@dataclass
class EventSet:
    """The times `t` in `Z_p` at which an event holds: a union of disjoint balls
    `t_0 + p^level Z_p`.

    - `balls`: the balls, as `Ball`s with one-coordinate centres.
    - `p`, `depth`: the prime and the number of digits required to vanish.
    - `zero_bound`: for a single condition, Strassmann's bound on the number of exact
        zeros of `t -> g(phi_t(x))` in `Z_p`, counted with multiplicity; `None` if the
        condition vanishes identically to the working precision, or for systems.
    """

    balls: list
    p: int
    depth: int
    zero_bound: int | None = field(default=None)

    def __contains__(self, t) -> bool:
        return any(ball.contains((int(t),), self.p) for ball in self.balls)

    def __bool__(self) -> bool:
        return bool(self.balls)

    def first(self) -> int | None:
        """The smallest non-negative integer time in the set (`None` if empty)."""
        return min((ball.center[0] for ball in self.balls), default=None)

    def measure(self) -> float:
        """The Haar measure of the set, `sum p^{-level}`."""
        return float(sum(float(self.p) ** -ball.level for ball in self.balls))


def _intersect(first: list, second: list, p: int) -> list:
    out = []
    for a in first:
        for b in second:
            ball = a.intersect(b, p)
            if ball is not None:
                out.append(ball)
    return out


def event_series(flow, event):
    """For each condition `g`, the dense coefficients of `G(t) = g(phi_t(x))` as a
    polynomial in `t` whose coefficients are polynomials in `x`:
    `G = sum_n t^n (D^n g)(x) / n!`.
    """
    conditions = [parse_polynomial(g, flow.dimension, flow.p) for g in event.condition]
    series = flow._divided_series(conditions)
    deg = max(degree([poly for polys in series for poly in polys]), 0)
    out = []
    for c in range(len(conditions)):
        polys = [polys[c] for polys in series]
        out.append(to_dense(polys, flow.dimension, deg))
    return np.stack(out)


def event_times(flow, event: PAdicEvent, x) -> tuple[list[EventSet], tuple]:
    p, K = flow.p, flow.precision
    depth = K if event.depth is None else event.depth
    if not 1 <= depth <= K:
        raise ValueError(f"`depth` must lie in 1, ..., {K}.")
    coefficients = event_series(flow, event)
    batch, (points,) = flow._flatten((x, True))
    res = flow.residues
    values = np.asarray(
        res.decode(evaluate(res, res.asarray(coefficients)[None], points))
    )
    out = []
    for row in values:
        polys = [{(n,): int(c) for n, c in enumerate(series) if c} for series in row]
        sets = [vanishing_set([poly], 1, p, K, depth) for poly in polys]
        balls = sets[0]
        for other in sets[1:]:
            balls = _intersect(balls, other, p)
        bound = None
        if len(polys) == 1 and polys[0]:
            lowest = min(valuation(c, p, K) for c in polys[0].values())
            if lowest < K:
                bound = max(
                    n for (n,), c in polys[0].items() if valuation(c, p, K) == lowest
                )
        out.append(EventSet(sorted(balls, key=lambda b: b.center), p, depth, bound))
    return out, batch


__all__ = ["Ball", "EventSet", "PAdicEvent", "event_times"]
