"""ABS, the adelic sieve by branch type, against the etale sieve of
`benchmarks/skolem.py`, on three data sets: (1) random integer recurrences of order
2-10 with entries in `[-20, 20]` (the distribution of the SKOLEM data set, drawn here
since that file is not shipped), (2) orbits of random quadratic maps of `Z^2` and
`Z^3` (the generator of `paper/checks/dml_etale_primes.py` with its seeds, so the
6 + 4 orbits with no etale prime are among them), (3) the generalised Henon maps of
`benchmarks/nonlinear_skolem.py`. The question is the set of `n` with
`x_n = target`, `x` the first coordinate: `n` in `Z` for the invertible orbits (1)
and (3), `n >= 0` for (2).

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
  backward when `|x| >= |y|`), so an exact walk both ways decides the instance. For
  recurrences: if `theta + sum_i |r_i| theta^-i <= |r_0|`, the cone
  `|u_{m+1}| >= theta |u_m|` is invariant and `u` has no zero once inside it.

For recurrences and Henon maps the zero set is certified over `Z`. The time `-n` of
a recurrence is the integer recurrence `s_m = R^m u_{-m}`, `R = r_{k-1}`
(`reversed_recurrence`): records at primes where the map is invertible modulo `p`
hold both ways (reflected for `n < 0`); the others, and the real place, hold on one
side, and each side is settled on its own.

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
import os
import random
import time
from fractions import Fraction
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


def depth_record(step, x0, target, p, kind, sides, cap=20000, top=12):
    """The orbit modulo `p^k` for `k = 1, 2, ...` until no point of its cycle hits
    the target or the cycle passes `cap` points: the deepest record. `sides` is
    `"both"` at a prime where the map is invertible modulo `p` (the cycle is pure and
    runs backwards too), else `"forward"`: the cover holds for `n >= T` only.
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
                "counted": [(c, P, None) for c in hits], "sides": sides}
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
            "counted": list(event.classes), "sides": "both"}


def reflect(record):
    """A two-sided record in the time `-n`."""
    return record | {
        "cover": [(-c % N, N) for c, N in record["cover"]],
        "counted": [(-c, N, b) for c, N, b in record["counted"]],
    }


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
    """`(how, zeros, period)` by the filtration: every `n` in `Z` with `x_n = target`
    when `how` is `"escape"` (`period` is `None`); for `"periodic"` the orbit is a
    cycle of length `period` and the zero set is `zeros + period Z`, `zeros` the hits
    on `0 <= n < period`.
    """
    assert c in (1, -1), c  # invertible over Z: the backward step divides by c
    R = abs(P[0]) + abs(P[1]) + 3

    def poly(y):
        return sum(a * y**i for i, a in enumerate(P))

    zeros, seen, (x, y), n = [], set(), start, 0
    for _ in range(cap):
        if (x, y) in seen:
            return "periodic", sorted(zeros), n  # the map is invertible: back to the start
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
            return "escape", sorted(zeros), None  # |x_{m-1}| >= 2 |x_m| from here on
        x, y, n = c * (y - poly(x)), x, n - 1
        if x == target:
            zeros.append(n)
    raise RuntimeError("no escape")


def reversed_recurrence(r, u):
    """`(r', s)` with `s_m = R^m u_{-m}`, `R = r_{k-1}`: an integer recurrence
    `s_m = r'_0 s_{m-1} + ... + r'_{k-1} s_{m-k}` whose zeros are those of `u` at
    `n = -m`. From `R u_{n-k} = u_n - r_0 u_{n-1} - ... - r_{k-2} u_{n-k+1}`, times
    `R^{m+k-1}`: `r'_j = -r_{k-2-j} R^j` for `j < k - 1` and `r'_{k-1} = R^{k-1}`.
    """
    k, R = len(r), r[-1]
    back = [-r[k - 2 - j] * R**j for j in range(k - 1)] + [R ** (k - 1)]
    v = [Fraction(x) for x in u]  # u_{1-k}, ..., u_0 at the end, built backwards
    for _ in range(k - 1):
        v.insert(0, (v[k - 1] - sum(a * v[k - 2 - i] for i, a in enumerate(r[:-1]))) / R)
    s = [R**m * v[k - 1 - m] for m in range(k)]  # s_0 = u_0, ..., s_{k-1}
    assert all(x.denominator == 1 for x in s)
    return tuple(back), tuple(int(x) for x in s)


def cone(r, grid=400):
    """The least `theta` on a grid with `theta + sum_i |r_i| theta^-i <= |r_0|`
    (`i >= 1`), or `None`. Then the cone `|u_{m+1}| >= theta |u_m|` along a state is
    invariant: `|u_{n+1}| >= |r_0| |u_n| - sum_i |r_i| |u_{n-i}| >= theta |u_n|`, so
    `u` never vanishes again once the orbit enters it with `u_n != 0`.
    """
    a = abs(r[0])
    if not a:
        return None
    for j in range(1, grid + 1):
        theta = Fraction(a * j, grid)
        if theta + sum(abs(c) / theta ** (i + 1) for i, c in enumerate(r[1:])) <= a:
            return theta
    return None


def dominant(r, u, cap=10**5, prec=128):
    """`N` with `u_n != 0` for every `n >= N`, when the characteristic polynomial is
    squarefree with one root `l` of largest modulus and `u_n = c l^n + sum_i c_i l_i^n`
    has `c != 0`; `None` otherwise or if `N > cap`. In ball arithmetic (Arb): every
    `|l_i| <= q |l|` with `q < 1`, so `|u_n| >= |l|^n (|c| - q^n sum_i |c_i|) > 0` once
    `q^N sum_i |c_i| < |c|`. Equal moduli (a complex pair, `+-l`) are left to others.
    """
    import flint

    k = len(r)
    P = flint.fmpz_poly([-c for c in reversed(r)] + [1])
    if P.gcd(P.derivative()).degree() > 0:
        return None
    old, flint.ctx.prec = flint.ctx.prec, prec
    try:
        roots = sorted((z for z, _ in P.complex_roots()), key=lambda z: -z.abs_upper().mid())
        top, rest = roots[0], roots[1:]
        rho = max(z.abs_upper() for z in rest)
        if not top.abs_lower() > rho:
            return None
        V = flint.acb_mat([[z**n for z in roots] for n in range(k)])
        c = V.solve(flint.acb_mat([[v] for v in u])).entries()
        lead, tail = c[0].abs_lower(), sum(x.abs_upper() for x in c[1:])
        if not lead > 0:
            return None
        if tail == 0:
            return 0
        N = ((tail / lead).log() / (top.abs_lower() / rho).log()).upper()
        if not N.is_finite():
            return None
        N = max(0, math.floor(float(N.mid())) + 1)  # an exact endpoint; floor + 1 is past it
    finally:
        flint.ctx.prec = old
    return N if N <= cap else None


def real_record(r, u, sides, steps=300):
    """The real place for a recurrence: `T` such that `u_n != 0` for every `n > T`,
    as a record with an empty cover, or `None` if neither the dominant root, a
    dominant complex pair (Baker--Davenport, `baker.py`) nor the cone within `steps`
    gives one.
    """
    level = int(os.environ.get("ABS_REAL", "2"))  # 0: the cone, 1: + dominant root, 2: + Baker
    N = dominant(r, u) if level >= 1 else None
    if N is None and level >= 2:
        from baker import complex_pair

        N = complex_pair(r, u)
    if N is not None:
        return {"p": None, "type": "real", "T": max(N, len(u) - 1), "cover": [],
                "counted": [], "sides": sides}
    theta = cone(r)
    if theta is None:
        return None
    x = list(u)
    for n in range(len(u) - 1, steps):
        if x[-1] and all(abs(b) >= theta * abs(a) for a, b in zip(x[-len(u):], x[-len(u) + 1:])):
            return {"p": None, "type": "real", "T": n, "cover": [], "counted": [], "sides": sides}
        x.append(sum(a * b for a, b in zip(r, reversed(x[-len(u):]))))
    return None


def recurrence_map(r):
    from galrax.padic import companion

    def step(x, m=None):  # u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}
        head = sum(a * b for a, b in zip(r, reversed(x)))
        return x[1:] + ((head % m if m else head),)

    return companion(list(reversed(r))), step


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
        return F, step, x0, target, lambda w: map_zeros(step, x0, target, w), 120, None
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


def settle(records, zeros, need, invertible):
    """The types of the records that settle `n >= 0` and, on an invertible orbit,
    `n < 0` (the reflected two-sided records with the backward ones), or `None`.
    """
    forward = verdict([r for r in records if r["sides"] != "backward"], zeros, need)
    if forward is None or not invertible:
        return forward and forward["type"]
    mirror = [reflect(r) if r["sides"] == "both" else r for r in records if r["sides"] != "forward"]
    backward = verdict(mirror, [-n for n in zeros], need)
    return backward and f"{forward['type']} / {backward['type']}"


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
    invertible = case[0] != "quadratic"
    records, zeros, seen, etale, queue = [], None, None, 0, None
    if mode == "abs" and case[0] == "recurrence":  # the real place, both ways
        rb, s = reversed_recurrence(case[2], case[3])
        pair = real_record(case[2], case[3], "forward"), real_record(rb, s, "backward")
        records += [rec for rec in pair if rec]
    while True:
        need = max([window] + [r["T"] for r in records])
        if need != seen:
            zeros, seen = zeros_in(need), need
        if zeros is None:
            result["error"] = "hit too far for integers"
            break
        for r in records:
            side = [-n for n in zeros] if r["sides"] == "backward" else zeros
            assert sound(r, side), (case, r["p"], r["type"])
        by = settle(records, zeros, need, invertible)
        if by is not None:
            result |= {"certified": True, "by": by, "zeros": zeros}
            break
        if queue is None:
            types = {p: b for p in PRIMES if (b := branch(F, step, x0, p)) is not None}
            # Cheapest first. ABS adds the orbit modulo p^k: forward (and, for `back`,
            # backward) at the singular primes, both ways at the etale ones, where it costs
            # far less than Orbit.times and gives the same exclusions; p = 2 has only this.
            queue = []
            for p, (t, c) in types.items():
                if t == "etale" and p > 2:
                    queue.append((c, 1, p, "orbit"))
                if mode == "abs" and p < depth_below:
                    queue.append((c, 0, p, "depth"))
                    if t != "etale" and back is not None:
                        queue.append((c, 0, p, "depth back"))
            queue.sort()
        queue = [q for q in queue if q[3] != "orbit" or etale < primes]
        if not queue:
            break
        _, _, p, how = queue.pop(0)
        t = types[p][0]
        if how == "orbit":
            records.append(etale_record(F, x0, target, p))
            etale += 1
        elif how == "depth":
            sides = "both" if t == "etale" else "forward"
            kind = t if t != "etale" else "etale mod p^k"
            if rec := depth_record(step, x0, target, p, kind, sides):
                records.append(rec)
        else:
            Fb, stepb, s = back
            tb = (branch(Fb, stepb, s, p) or ("mixed",))[0]
            if rec := depth_record(stepb, s, target, p, tb, "backward"):
                records.append(rec)
    result["primes"] = sorted({r["p"] for r in records if r["p"]})
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
