"""ABS, the adelic sieve by branch type, against the etale sieve of
`benchmarks/skolem.py`, on three data sets: (1) random integer recurrences of order
2-10 with entries in `[-20, 20]` (the distribution of the SKOLEM data set, drawn here
since that file is not shipped), (2) orbits of random quadratic maps of `Z^2` and
`Z^3` (the generator of `paper/checks/dml_etale_primes.py` with its seeds, so the
6 + 4 orbits with no etale prime are among them), (3) the generalised Henon maps of
`benchmarks/nonlinear_skolem.py`. The question is the set of `n >= 0` with
`x_n = target`, `x` the first coordinate.

Branches at a prime `p`, by the Jacobian `A` of `g^L` on the cycle of the orbit mod p:

- etale (`A` invertible): `Orbit.times`, Strassmann's bound on every class
  `n = c mod N`, as in the etale sieve.
- nilpotent or mixed (`A` singular): the orbit modulo `p^k`, `k = 1, 2, ...`, read
  exactly, has a tail `T` and a cycle `P`; a zero `n >= T` lies in a class
  `n = T + r mod P` with `x_{T+r} = target mod p^k`. When no point of the cycle hits,
  every zero is below `T`. On a nilpotent branch the orbit tends to the attracting
  cycle `beta`, so this happens at `k = v(x(beta) - target) + 1`, with `T` about
  `a + d M_1 k` (the bound of the note); on a mixed branch, when the shadow on the
  centre manifold misses the target. The bound is read off the computed cycle: it
  needs neither `beta`, the shadow nor the constants `C`, `nu`, `K` of the note, and
  holds for any branch; the branch theory only says when it succeeds.
- the real place, for Henon maps: outside the square of side `R = |a_0| + |a_1| + 3`
  the orbit escapes with `|x_n|` at least doubling (forward when `|y| >= |x|`,
  backward when `|x| >= |y|`), so an exact walk both ways decides the instance.

Primes are combined as in the etale sieve: a class of one prime that meets no class
of another holds no zero `n >= max T`. Zeros below that are found exactly: in
integers for recurrences and Henon maps, modulo the prime `2^521 - 1` for quadratic
maps, with every hit there checked in integers.

    python benchmarks/adelic_sieve.py recurrences --per-order 20 --timeout 30
    python benchmarks/adelic_sieve.py quadratic --count 100 --timeout 30
    python benchmarks/adelic_sieve.py henon --count 200 --timeout 60
"""

import argparse
import math
import multiprocessing as mp
import random
import time
from itertools import combinations_with_replacement

from nonlinear_skolem import exact_hits, orbit as henon_orbit
from skolem import exact_zeros, run


PRIMES = tuple(p for p in range(2, 200) if all(p % q for q in range(2, p)))
Q = 2**521 - 1


# Maps are lists of `{exponents: coefficient}`, as for `Orbit`.


def stepper(F):
    terms = [[(c, [(i, k) for i, k in enumerate(e) if k]) for e, c in f.items()] for f in F]

    def step(x, m=None):
        out = []
        for f in terms:
            s = 0
            for c, e in f:
                for i, k in e:
                    c *= x[i] ** k
                s += c
            out.append(s % m if m else s)
        return tuple(out)

    return step


def jacobian(F, x, m):
    d = len(x)
    J = [[0] * d for _ in F]
    for row, f in zip(J, F):
        for e, c in f.items():
            for j in range(d):
                if e[j]:
                    t = c * e[j]
                    for i, (v, k) in enumerate(zip(x, e)):
                        t *= v ** (k - (i == j))
                    row[j] += t
    return [[v % m for v in row] for row in J]


def matmul(A, B, m):
    return [[sum(a * b for a, b in zip(r, c)) % m for c in zip(*B)] for r in A]


def rank(A, p):
    A, r = [[v % p for v in row] for row in A], 0
    for c in range(len(A[0])):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        inv = pow(A[r][c], -1, p)
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c] * inv
                A[i] = [(a - f * b) % p for a, b in zip(A[i], A[r])]
        r += 1
    return r


def cycle_mod(step, x0, m, cap):
    """`(orbit, T)`: the orbit modulo `m` up to its first repeat, which is at `T`;
    `None` past `cap` points.
    """
    seen, orbit, x = {}, [], tuple(v % m for v in x0)
    while x not in seen:
        if len(orbit) >= cap:
            return None
        seen[x] = len(orbit)
        orbit.append(x)
        x = step(x, m)
    return orbit, seen[x]


def branch(F, step, x0, p, cap=5000):
    """`(type, cost)` of the orbit modulo `p`: `etale`, `nilpotent` or `mixed` by
    the cycle Jacobian `A` (Fitting), and `p` times the number of points; `None`
    past `cap` points.
    """
    found = cycle_mod(step, x0, p, cap)
    if found is None:
        return None
    orbit, T = found
    d = len(x0)
    if all(sum(e) <= 1 for f in F for e in f):
        A = jacobian(F, orbit[0], p)  # constant: A = J^L has the type of J
    else:
        A = [[int(i == j) for j in range(d)] for i in range(d)]
        for y in orbit[T:]:
            A = matmul(jacobian(F, y, p), A, p)
    if rank(A, p) == d:
        return "etale", p * len(orbit)
    power = A
    for _ in range(d - 1):
        power = matmul(power, A, p)
    return ("nilpotent" if not any(map(any, power)) else "mixed"), p * len(orbit)


def depth_record(step, x0, target, p, kind, cap=20000, top=12):
    """The orbit modulo `p^k` for `k = 1, 2, ...` until no point of its cycle hits
    the target or the cycle passes `cap` points: the deepest record.
    """
    best = None
    for k in range(1, top + 1):
        m = p**k
        found = cycle_mod(step, x0, m, cap)
        if found is None:
            break
        orbit, T = found
        P = len(orbit) - T
        hits = sorted({n % P for n in range(T, len(orbit)) if (orbit[n][0] - target) % m == 0})
        best = {"p": p, "k": k, "type": kind, "T": T, "cover": [(c, P) for c in hits],
                "counted": [(c, P, None) for c in hits]}
        if not hits:
            break
    return best


def etale_record(F, x0, target, p):
    from galrax.padic import Orbit, PAdicEvent

    d = len(x0)
    orbit = Orbit(p, max(4, int(30 / p.bit_length())), F, x0)
    x = tuple(int(i == 0) for i in range(d))
    event = orbit.times(PAdicEvent({x: 1, (0,) * d: -target}))
    cover = [
        (b.center[0] % (event.period * p**b.level), event.period * p**b.level)
        for b in event.balls
    ]
    return {"p": p, "type": "etale", "T": orbit.tail, "cover": cover,
            "counted": list(event.classes)}


def verdict(records, zeros, window):
    """The record that proves `zeros` (exact for `0 <= n <= window`, and for some
    `n < 0` on invertible orbits) to be every zero `n >= 0`, or `None`.
    """
    if not records or window < max(r["T"] for r in records):
        return None
    for main in sorted(records, key=lambda r: bool(r["cover"])):  # a record with no hit settles alone
        total, found = 0, 0
        for c, N, bound in main["counted"]:
            if not all(
                any((c - c2) % math.gcd(N, N2) == 0 for c2, N2 in other["cover"])
                for other in records
                if other is not main
            ):
                continue  # no zero n >= max T in this class
            if bound is None:
                total = None
                break
            total += bound
            found += sum(1 for n in zeros if (n - c) % N == 0 and (n >= main["T"] or not main["T"]))
        if total is not None and found >= total:
            return main
    return None


def sound(record, zeros):
    """Every exact zero `n >= max(T, 0)` lies in the cover."""
    return all(
        any((n - c) % N == 0 for c, N in record["cover"])
        for n in zeros
        if n >= record["T"]
    )


# (1) recurrences, (2) quadratic maps, (3) Henon maps.


def recurrence_cases(per_order, rng):
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
    cases = []
    for _ in range(count):
        P = [rng.randint(-5, 5), rng.randint(-5, 5), rng.choice([-3, -2, -1, 1, 2, 3])]
        start, c = (rng.randint(-5, 5), rng.randint(-5, 5)), rng.choice([1, -1])
        target = rng.randint(-20, 20)
        if rng.random() < 0.5:
            target = henon_orbit(c, P, start, 3)[rng.randint(-3, 3)][0]
        cases.append(("henon", c, P, start, target))
    return cases


def map_zeros(step, x0, target, window, exact_bits=1 << 22):
    """`n` in `0..window` with `x_n = target`: modulo `Q`, each hit checked in
    integers; `None` if a hit is too far for integers.
    """
    hits, x = [], tuple(v % Q for v in x0)
    for n in range(window + 1):
        if (x[0] - target) % Q == 0:
            hits.append(n)
        x = step(x, Q)
    x, out = tuple(x0), []
    for n in range(hits[-1] + 1 if hits else 0):
        if n in hits and x[0] == target:
            out.append(n)
        x = step(x)
        if max(abs(v).bit_length() for v in x) > exact_bits:
            return None
    return out


def henon_real(c, P, start, target, cap=10**6):
    """`(how, zeros)` by the filtration: every `n` in `Z` with `x_n = target` when
    `how` is `"escape"`; for `"periodic"` the orbit is a cycle and `zeros` the hits
    on one period.
    """
    R = abs(P[0]) + abs(P[1]) + 3

    def poly(y):
        return sum(a * y**i for i, a in enumerate(P))

    zeros, seen, (x, y), n = [], set(), start, 0
    for _ in range(cap):
        if (x, y) in seen:
            return "periodic", sorted(zeros)
        seen.add((x, y))
        if x == target:
            zeros.append(n)
        if abs(y) >= max(abs(x), R) and abs(y) > abs(target):
            break  # |y| at least doubles from here on, and x_{m+1} = y_m
        x, y, n = y, c * x + poly(y), n + 1
    else:
        raise RuntimeError("no escape")
    (x, y), n = start, 0
    for _ in range(cap):
        if abs(x) >= max(abs(y), R) and abs(x) > abs(target):
            return "escape", sorted(zeros)  # |x_{m-1}| >= 2 |x_m| from here on
        x, y, n = c * (y - poly(x)), x, n - 1
        if x == target:
            zeros.append(n)
    raise RuntimeError("no escape")


def instance(case):
    """`(F, step, x0, target, zeros(window), window)` of a case."""
    if case[0] == "recurrence":
        from galrax.padic import companion

        _, k, r, u = case
        F = companion(list(reversed(r)))

        def step(x, m=None):  # u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}
            head = sum(a * b for a, b in zip(r, reversed(x)))
            return x[1:] + ((head % m if m else head),)

        return F, step, u, 0, lambda w: exact_zeros(list(r), list(u), w), 200
    if case[0] == "quadratic":
        _, d, _, F, x0, target = case
        step = stepper(F)
        return F, step, x0, target, lambda w: map_zeros(step, x0, target, w), 120
    _, c, P, start, target = case
    F = [{(0, 1): 1}, {(1, 0): c, **{(0, i): a for i, a in enumerate(P) if a}}]
    return F, stepper(F), start, target, lambda w: exact_hits(c, P, start, target, w)[0], 14


def group(case):
    if case[0] == "recurrence":
        return f"order {case[1]}"
    if case[0] == "quadratic":
        return f"d = {case[1]}" + (", no etale prime" if case[2] else "")
    return "henon"


def solve(case, mode, primes=8, depth_below=50):
    import jax

    jax.config.update("jax_enable_x64", True)
    began = time.perf_counter()
    result = {"group": group(case), "mode": mode, "certified": False, "primes": []}
    if mode == "abs" and case[0] == "henon":
        _, c, P, start, target = case
        how, zeros = henon_real(c, P, start, target)
        if how == "escape":
            near = exact_hits(c, P, start, target, 12)[0]
            assert [n for n in zeros if abs(n) <= 12] == near, case
        return result | {"certified": True, "by": f"real ({how})", "zeros": zeros,
                         "seconds": time.perf_counter() - began}
    F, step, x0, target, zeros_in, window = instance(case)
    types = {p: b for p in PRIMES if (b := branch(F, step, x0, p)) is not None}
    # cheapest first; ABS adds the singular branches, and p = 2 (Orbit.times skips it)
    queue = [
        (c, p, t) for p, (t, c) in types.items()
        if (t == "etale" and p > 2) or (mode == "abs" and p < depth_below and (t != "etale" or p == 2))
    ]
    queue.sort()
    records, zeros, seen, etale = [], None, None, 0
    while True:
        need = max([window] + [r["T"] for r in records])
        if need != seen:
            zeros, seen = zeros_in(need), need
        if zeros is None:
            result["error"] = "hit too far for integers"
            break
        for r in records:
            assert sound(r, zeros), (case, r["p"], r["type"])
        main = verdict(records, zeros, need)
        if main is not None:
            result |= {"certified": True, "by": main["type"], "zeros": zeros}
            break
        while queue and queue[0][2] == "etale" and queue[0][1] > 2 and etale == primes:
            queue.pop(0)  # no more Strassmann primes
        if not queue:
            break
        _, p, t = queue.pop(0)
        if t == "etale" and p > 2:
            records.append(etale_record(F, x0, target, p))
            etale += 1
        elif (rec := depth_record(step, x0, target, p, t if p > 2 or t != "etale" else "etale mod 2^k")):
            records.append(rec)
    result["primes"] = [r["p"] for r in records]
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
            pairs.setdefault(key(r)[2], {})[key(r)[1]] = [n for n in r["zeros"] if n >= 0]
    clash = [k for k, v in pairs.items() if len(v) == 2 and v["sieve"] != v["abs"]]
    gained = sum(1 for v in pairs.values() if list(v) == ["abs"])
    lost = sum(1 for v in pairs.values() if list(v) == ["sieve"])
    print(f"\ncertified by ABS only: {gained}; by the sieve only: {lost}; "
          f"both, with different zeros n >= 0: {len(clash)}")
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
    args = parser.parse_args()

    rng = random.Random(args.seed)
    if args.data == "recurrences":
        cases = recurrence_cases(args.per_order, rng)
    elif args.data == "quadratic":
        cases = quadratic_cases(args.count, rng)
    else:
        cases = henon_cases(args.count, rng)
    tasks = [((group(c), mode, i), c) for i, c in enumerate(cases) for mode in ("sieve", "abs")]
    report(run(tasks, args.timeout, args.processes, job))


if __name__ == "__main__":
    main()
