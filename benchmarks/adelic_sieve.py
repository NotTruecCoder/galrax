"""ABS (`galrax.padic.adelic`) against the etale sieve of `benchmarks/skolem.py`, on
three data sets: (1) random integer recurrences of order 2-10 with entries in
`[-20, 20]` (the distribution of the SKOLEM data set, drawn here since that file is
not shipped, or sampled from it with `--dataset`), (2) orbits of random quadratic
maps of `Z^2` and `Z^3` (the generator of `paper/checks/dml_etale_primes.py` with its
seeds, so the 6 + 4 orbits with no etale prime are among them), (3) the generalised
Henon maps of `benchmarks/nonlinear_skolem.py`. The question is the set of `n` with
`x_n = target`, `x` the first coordinate: `n` in `Z` for the invertible orbits (1)
and (3), `n >= 0` for (2). `ABS_REAL=0` or `1` builds the real place of recurrences
up step by step (the cone; with the dominant root; `2`, the default, with
Baker--Davenport).

    python benchmarks/adelic_sieve.py recurrences --per-order 20 --timeout 30
    python benchmarks/adelic_sieve.py quadratic --count 100 --timeout 30
    python benchmarks/adelic_sieve.py henon --count 200 --timeout 60
"""

import argparse
import multiprocessing as mp
import os
import random
import time
from itertools import combinations_with_replacement

from galrax.padic.adelic import (
    branch,
    certify,
    exact_zeros,
    henon_real,
    jacobian,
    PRIMES,
    rank,
    real_record,
    recurrence_map,
    reversed_recurrence,
    stepper,
    window_zeros,
)
from galrax.padic.cli import run
from nonlinear_skolem import exact_hits, make_cases


# (1) recurrences, (2) quadratic maps, (3) Henon maps.


def recurrence_cases(per_order, rng, dataset=None):
    """Drawn as in the SKOLEM data set, or sampled from it (`mfcs.tpls`, as in
    `benchmarks/skolem.py`), so both tools can be run on the same instances.
    """
    if dataset:
        import ast

        with open(dataset) as f:
            data = [ast.literal_eval(line) for line in f if line.strip()]
        cases = []
        for k in range(2, 11):  # the draw of skolem.py, then drop r_{k-1} = 0 or u = 0
            of_order = [c for c in data if c[0] == k]
            cases += [("recurrence", k, tuple(r), tuple(u))
                      for _, r, u in rng.sample(of_order, min(per_order, len(of_order)))
                      if r[-1] and any(u)]
        return cases
    cases = []
    for k in range(2, 11):
        while sum(1 for c in cases if c[1] == k) < per_order:
            r = tuple(rng.randint(-20, 20) for _ in range(k))
            u = tuple(rng.randint(-20, 20) for _ in range(k))
            if r[-1] and any(u):
                cases.append(("recurrence", k, r, u))
    return cases


def quadratic_maps(d, trials):
    """The maps and starts of `paper/checks/dml_etale_primes.py`, same draws."""
    rng = random.Random(d)
    monos = [()] + [(i,) for i in range(d)] + list(combinations_with_replacement(range(d), 2))

    def convert(F):
        return [{tuple(m.count(i) for i in range(d)): c for m, c in f.items() if c} for f in F]

    out = []
    while len(out) < trials:
        F = [{m: rng.randint(-3, 3) for m in monos if rng.random() < 0.6} for _ in range(d)]
        x0 = tuple(rng.randint(-3, 3) for _ in range(d))
        big = 10**9 + 7
        G = convert(F)
        if rank(jacobian(G, [rng.randrange(big) for _ in range(d)], big), big) < d:
            continue  # det dPhi vanishes identically
        step, x, seen, infinite = stepper(G), x0, set(), True
        for _ in range(40):
            if x in seen:
                infinite = False
                break
            seen.add(x)
            x = step(x)
            if max(abs(v) for v in x) > 10**60:
                break
        if infinite:
            out.append((G, x0))
    return out


def quadratic_cases(count, rng):
    cases = []
    for d, trials in ((2, 300), (3, 200)):
        maps = quadratic_maps(d, trials)
        # the orbits with no etale prime first, so a small count keeps them
        none = [i for i, (F, x0) in enumerate(maps) if not any(
            (b := branch(F, stepper(F), x0, p)) and b[0] == "etale"
            for p in PRIMES if 2 < p <= (200 if d == 2 else 100))]
        order = none + [i for i in range(len(maps)) if i not in none]
        for i in order[:count]:
            F, x0 = maps[i]
            target = rng.randint(-20, 20)
            if rng.random() < 0.5:  # half the targets are values of the orbit
                x = x0
                for _ in range(rng.randint(0, 3)):
                    x = stepper(F)(x)
                target = x[0]
            cases.append(("quadratic", d, i in none, F, x0, target))
    return cases


def henon_cases(count, rng):
    return [("henon", *case[1:]) for case in make_cases(count, rng)]


def instance(case):
    """`(F, step, x0, target, zeros(window), window, back)` of a case; `back` is the
    `(F, step, x0)` of the orbit in the time `-n` when it is an integer orbit with
    singular primes (recurrences), else `None`.
    """
    if case[0] == "recurrence":
        _, k, r, u = case
        F, step = recurrence_map(r)
        rb, s = reversed_recurrence(r, u)
        back = (*recurrence_map(rb), s)
        return F, step, u, 0, lambda w: exact_zeros(list(r), list(u), w), 200, back
    if case[0] == "quadratic":
        _, d, _, F, x0, target = case
        step = stepper(F)
        return F, step, x0, target, lambda w: window_zeros(step, x0, target, w), 120, None
    _, c, P, start, target = case
    F = [{(0, 1): 1}, {(1, 0): c, **{(0, i): a for i, a in enumerate(P) if a}}]
    zeros = lambda w: exact_hits(c, P, start, target, w)[0]  # noqa: E731
    return F, stepper(F), start, target, zeros, 14, None


def group(case):
    if case[0] == "recurrence":
        return f"order {case[1]}"
    if case[0] == "quadratic":
        return f"d = {case[1]}" + (", no etale prime" if case[2] else "")
    return "henon"


def solve(case, mode, primes=8, depth_below=50):
    """`mode`: `"sieve"` (Strassmann at etale primes) or `"abs"`. The zero set is
    certified over `Z` for invertible orbits (recurrences, Henon maps), over `n >= 0`
    for quadratic maps.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    began = time.perf_counter()
    result = {"group": group(case), "mode": mode, "certified": False, "primes": []}
    if mode == "abs" and case[0] == "henon":
        _, c, P, start, target = case
        how, zeros, period = henon_real(c, P, start, target)
        if how == "escape":
            near = exact_hits(c, P, start, target, 12)[0]
            assert [n for n in zeros if abs(n) <= 12] == near, case
        # periodic: the zero set is zeros + period Z, carried by `period`
        return result | {"certified": True, "by": f"real ({how})", "zeros": zeros,
                         "period": period, "seconds": time.perf_counter() - began}
    F, step, x0, target, zeros_in, window, back = instance(case)
    records = []
    if mode == "abs" and case[0] == "recurrence":  # the real place, both ways
        level = int(os.environ.get("ABS_REAL", "2"))
        rb, s = reversed_recurrence(case[2], case[3])
        pair = (real_record(case[2], case[3], "forward", level=level),
                real_record(rb, s, "backward", level=level))
        records = [rec for rec in pair if rec]
    result |= certify(F, step, x0, target, zeros_in, window, back=back,
                      invertible=case[0] != "quadratic", records=records, mode=mode,
                      primes=primes, depth_below=depth_below, label=case)
    return result | {"seconds": time.perf_counter() - began}


def job(task):
    (_, mode, _), case = task
    return solve(case, mode) | {"key": task[0]}


def report(results):
    def key(r):
        return r.get("key") or r["order"]  # timeouts and errors carry the key as `order`

    groups = sorted({key(r)[0] for r in results})
    print("| group | method | instances | certified | timed out | errors | mean time (s) | mean primes | settled by |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in groups + ["all"]:
        for mode in ("sieve", "abs"):
            rows = [r for r in results if key(r)[1] == mode and g in ("all", key(r)[0])]
            done = [r for r in rows if r.get("certified")]
            late = sum(1 for r in rows if r.get("timeout"))
            errors = sum(1 for r in rows if "error" in r)
            mean = sum(r["seconds"] for r in done) / len(done) if done else float("nan")
            used = sum(len(r["primes"]) for r in done) / len(done) if done else float("nan")
            by = {}
            for r in done:
                by[r["by"]] = by.get(r["by"], 0) + 1
            settled = ", ".join(f"{k} {v}" for k, v in sorted(by.items()))
            print(f"| {g} | {mode} | {len(rows)} | {len(done)} | {late} | {errors} "
                  f"| {mean:.2f} | {used:.1f} | {settled} |")
    pairs = {}
    for r in results:
        if r.get("certified"):
            pairs.setdefault(key(r)[2], {})[key(r)[1]] = r["zeros"]
    clash = [k for k, v in pairs.items() if len(v) == 2 and v["sieve"] != v["abs"]]
    gained = sum(1 for v in pairs.values() if list(v) == ["abs"])
    lost = sum(1 for v in pairs.values() if list(v) == ["sieve"])
    print(f"\ncertified by ABS only: {gained}; by the sieve only: {lost}; "
          f"both, with different zeros: {len(clash)}")
    for r in [r for r in results if "error" in r][:5]:
        print(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", choices=["recurrences", "quadratic", "henon"])
    parser.add_argument("--per-order", type=int, default=20)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--processes", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", help="mfcs.tpls of the SKOLEM tool, for recurrences")
    parser.add_argument("--modes", default="sieve,abs")
    parser.add_argument("--dump", help="write the results here, one repr per line")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.data == "recurrences":
        cases = recurrence_cases(args.per_order, rng, args.dataset)
    elif args.data == "quadratic":
        cases = quadratic_cases(args.count, rng)
    else:
        cases = henon_cases(args.count, rng)
    modes = args.modes.split(",")
    tasks = [((group(c), mode, i), c) for i, c in enumerate(cases) for mode in modes]
    results = run(tasks, args.timeout, args.processes, job)
    if args.dump:
        with open(args.dump, "w") as f:
            f.writelines(f"{r!r}\n" for r in results)
    report(results)


if __name__ == "__main__":
    main()
