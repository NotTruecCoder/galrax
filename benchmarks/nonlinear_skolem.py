"""The Skolem problem for nonlinear maps: all `n` in `Z` with `x_n = target` along
an orbit of a generalised Henon map of `Z^2`,

    (x, y) -> (y, c x + P(y)),   c = +-1,   P(y) = a_2 y^2 + a_1 y + a_0  (a_2 != 0),

with coefficients, start and target drawn at random from small integers. The map
is invertible over `Z`, so negative `n` make sense, and `Dg` has determinant `-c`,
a unit at every prime: every orbit is within the scope of `Orbit`. SKOLEM and
other tools for linear recurrences do not apply.

The certificate is the one of `benchmarks/skolem.py`: Strassmann's bound on each
class `n = c mod N` at up to eight primes, taken in order of `p` times the period
of the orbit modulo `p`, classes that miss another prime's set dropped, compared
with the exact zeros for `|n| <= window` computed in integers. Orbits that are
periodic over `Z` are solved exactly and reported apart.

    python benchmarks/nonlinear_skolem.py --count 200 --timeout 60
"""

import argparse
import random
import time

from skolem import run, sieve


PRIMES = tuple(p for p in range(3, 200) if all(p % q for q in range(2, p)))


def cheap_primes(c, P, start, count, cap=5000):
    """The `count` primes with the least `p` times the period of the orbit modulo
    `p` (the map is invertible, so the orbit is a pure cycle).
    """
    costs = []
    for p in PRIMES:
        point = first = (start[0] % p, start[1] % p)
        for n in range(1, cap + 1):
            x, y = point
            point = (y, (c * x + sum(a * y**i for i, a in enumerate(P))) % p)
            if point == first:
                costs.append((n * p, p))
                break
    return [p for _, p in sorted(costs)[:count]]


def orbit(c, P, start, window):
    """`{n: x_n}` for `|n| <= window`, in integers."""

    def poly(y):
        return sum(a * y**i for i, a in enumerate(P))

    points, point = {0: start}, start
    for n in range(1, window + 1):
        x, y = point
        point = points[n] = (y, c * x + poly(y))
    point = start
    for n in range(1, window + 1):
        u, v = point  # (u, v) = (y, c x + P(y)): x = c (v - P(u)), y = u
        point = points[-n] = (c * (v - poly(u)), u)
    return points


def exact_hits(c, P, start, target, window):
    """`(zeros, period)`: the `n` with `|n| <= window` and `x_n = target`, and the
    period of the orbit over `Z` if it closes up within the window.
    """
    points = orbit(c, P, start, window)
    period = next((n for n in range(1, window + 1) if points[n] == start), None)
    return sorted(n for n, (x, _) in points.items() if x == target), period


def solve(case, window=14, primes=8):
    import jax

    jax.config.update("jax_enable_x64", True)
    from galrax.padic import Orbit, PAdicEvent

    kind, c, P, start, target = case
    began = time.perf_counter()
    zeros, period = exact_hits(c, P, start, target, window)
    result = {"order": kind, "zeros": zeros, "certified": False, "bound": None}
    if period is not None:
        return result | {"periodic": period, "seconds": time.perf_counter() - began}
    g = [{(0, 1): 1}, {(1, 0): c, **{(0, i): a for i, a in enumerate(P) if a}}]
    event = PAdicEvent({(1, 0): 1, (0, 0): -target})
    sets = []
    for p in cheap_primes(c, P, start, primes):
        precision = max(4, int(30 / p.bit_length()))
        events = Orbit(p, precision, g, start).times(event)
        assert all(n in events for n in zeros if n >= 0), (case, p)
        sets.append(events)
        bound = sieve(sets)
        result |= {"primes": [e.p for e in sets], "bound": bound}
        if bound is not None and bound <= len(zeros):
            assert bound == len(zeros), case
            result["certified"] = True
            break
        if len(sets) == primes:
            break
    return result | {"seconds": time.perf_counter() - began}


def make_cases(count, seed):
    """`(kind, c, P, start, target)` for `count` random orbits; half the targets are
    values of the orbit.
    """
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        P = [rng.randint(-5, 5), rng.randint(-5, 5), rng.choice([-3, -2, -1, 1, 2, 3])]
        start, c = (rng.randint(-5, 5), rng.randint(-5, 5)), rng.choice([1, -1])
        target, kind = rng.randint(-20, 20), "random"
        if rng.random() < 0.5:
            target, kind = orbit(c, P, start, 3)[rng.randint(-3, 3)][0], "orbit value"
        cases.append((kind, c, P, start, target))
    return cases


def main():
    import skolem

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases = make_cases(args.count, args.seed)
    results = run(cases, args.timeout, args.processes or skolem.mp.cpu_count(), solve)

    print(
        "| target | orbits | periodic over Z | certified, with zeros "
        "| certified, never reached | not certified | timed out | mean time (s) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for kind in ("orbit value", "random", None):
        rows = [r for r in results if kind is None or r["order"] == kind]
        done = [r for r in rows if r.get("certified")]
        hitting = sum(1 for r in done if r["zeros"])
        periodic = sum(1 for r in rows if "periodic" in r)
        late = sum(1 for r in rows if r.get("timeout"))
        rest = len(rows) - len(done) - periodic - late
        mean = sum(r["seconds"] for r in done) / len(done) if done else float("nan")
        print(
            f"| {kind or 'all'} | {len(rows)} | {periodic} | {hitting} "
            f"| {len(done) - hitting} | {rest} | {late} | {mean:.1f} |"
        )
    errors = [r for r in results if "error" in r]
    print(f"\nerrors: {len(errors)}")
    for r in errors[:5]:
        print(r)


if __name__ == "__main__":
    main()
