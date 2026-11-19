"""ABS, the adelic sieve by branch type: certified zero sets of orbits over `Z`.

The question is the set of times `n` with `g(x_n) = 0` for an integer orbit of a
polynomial map and an integer polynomial `g` (`x_n = target`, `x` the first coordinate,
is `g = x - target`), or with `u_n = 0` for an integer linear recurrence. Every place
contributes a record, a class cover of the times `n >= T` that can still hold zeros:

- at an etale prime (the Jacobian `A` of the cycle modulo `p` invertible), Strassmann's
  bound on every class `n = c mod N`, through `Orbit.times`;
- at every prime, the orbit modulo `p^k`, `k = 1, 2, ...`, read exactly: a tail `T` and
  a cycle, and when no point of the cycle hits the target every zero is below `T`. On
  a nilpotent branch (`A` nilpotent) this happens once `k` exceeds the valuation of the
  target at the attracting cycle; on a mixed branch when the shadow on the centre
  manifold misses it. The bound is read off the computed cycle and needs no constants;
- at the real place: for Henon maps the escape filtration, for recurrences, in ball
  arithmetic (python-flint), a dominant root, a dominant complex pair through Matveev's
  bound and the Baker--Davenport reduction (`baker.py`), or an invariant cone.

A class of one record that meets no class of another holds no zero past both tails;
the zeros below are found exactly, and the counted bounds of the surviving classes
close the certificate. For an invertible orbit the time `-n` is handled as a second
orbit (for a recurrence, the integer recurrence `s_m = r_{k-1}^m u_{-m}`); records at
primes where the map is invertible modulo `p` hold both ways.

Maps are lists of `{exponents: coefficient}`, as for `Orbit`. The functions need
`jax_enable_x64`; the real place of recurrences needs `python-flint`.
"""

import dataclasses
import math
from fractions import Fraction


PRIMES = tuple(p for p in range(2, 200) if all(p % q for q in range(2, p)))
Q = 2**521 - 1  # a prime: zeros are first looked for modulo Q, then checked exactly


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


def as_poly(f, d):
    """`f` as `{exponents: coefficient}` in Python integers: a dict, or anything with a
    `.dict()` such as a Sage polynomial (duck-typed, Sage is not imported)."""
    items = f.dict().items() if hasattr(f, "dict") else f.items()
    return {(int(e),) if d == 1 and not hasattr(e, "__len__") else tuple(map(int, e)):
            int(c) for e, c in items}


def as_condition(g, d):
    """The polynomial `g` of a zero condition `g(x) = 0`; an integer `target` stands
    for `x_0 - target`."""
    if isinstance(g, dict) or hasattr(g, "dict"):
        return as_poly(g, d)
    one = tuple(int(i == 0) for i in range(d))
    return {one: 1, (0,) * d: -int(g)}


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


def depth_record(step, x0, condition, p, kind, sides, cap=20000, top=12):
    """The orbit modulo `p^k` for `k = 1, 2, ...` until no point of its cycle meets
    `g = 0` modulo `p^k` (`condition` as in `as_condition`; an exact zero of `g` is
    one modulo every `p^k`) or the cycle passes `cap` points: the deepest record.
    `sides` is `"both"` at a prime where the map is invertible modulo `p` (the cycle is
    pure and runs backwards too), else `"forward"`: the cover holds for `n >= T` only.
    """
    g, best = stepper([as_condition(condition, len(x0))]), None
    for k in range(1, top + 1):
        m = p**k
        found = cycle_mod(step, x0, m, cap)
        if found is None:
            break
        orbit, T = found
        P = len(orbit) - T
        hits = sorted({n % P for n in range(T, len(orbit)) if g(orbit[n], m)[0] == 0})
        best = {"p": p, "k": k, "type": kind, "T": T, "cover": [(c, P) for c in hits],
                "counted": [(c, P, None) for c in hits], "sides": sides}
        if not hits:
            break
    return best


def etale_record(F, x0, condition, p):
    from .events import PAdicEvent
    from .return_set import Orbit

    d = len(x0)
    orbit = Orbit(p, max(4, int(30 / p.bit_length())), F, x0)
    event = orbit.times(PAdicEvent(as_condition(condition, d)))
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


def window_zeros(step, x0, condition, window, exact_bits=1 << 22):
    """`n` in `0..window` with `g(x_n) = 0` (`condition` as in `as_condition`):
    modulo `Q`, each hit checked in integers; `None` if a hit is too far for integers.
    """
    g = stepper([as_condition(condition, len(x0))])
    hits, x = [], tuple(v % Q for v in x0)
    for n in range(window + 1):
        if g(x, Q)[0] == 0:
            hits.append(n)
        x = step(x, Q)
    x, out = tuple(x0), []
    for n in range(hits[-1] + 1 if hits else 0):
        if n in hits and g(x)[0] == 0:
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
    if len(P) != 3 or not P[2]:  # the escape bound is for quadratic P; a cycle above is exact
        raise ValueError("Escape needs P = (a_0, a_1, a_2) with a_2 != 0.")
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
        rho = max((z.abs_upper() for z in rest), default=flint.arb(0))  # order 1: no rest
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


def real_record(r, u, sides, steps=300, level=2):
    """The real place for a recurrence: `T` such that `u_n != 0` for every `n > T`,
    as a record with an empty cover, or `None` if neither the dominant root, a
    dominant complex pair (Baker--Davenport, `baker.py`) nor the cone within `steps`
    gives one. `level` builds the real place up: `0` the cone only, `1` with the
    dominant root, `2` with Baker--Davenport too.
    """
    N = dominant(r, u) if level >= 1 else None
    if N is None and level >= 2:
        from .baker import complex_pair

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
    from .return_set import companion

    def step(x, m=None):  # u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}
        head = sum(a * b for a, b in zip(r, reversed(x)))
        return x[1:] + ((head % m if m else head),)

    return companion(list(reversed(r))), step


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


def exact_zeros(recurrence, initial, window):
    """The `n` with `|n| <= window` and `u_n = 0`, in exact arithmetic."""
    k = len(recurrence)
    # u_{n-k} = (u_n - r_0 u_{n-1} - ... - r_{k-2} u_{n-k+1}) / r_{k-1}
    inverse = pow(recurrence[-1], -1, Q)
    candidates, state = [], [v % Q for v in initial]
    for n in range(window + 1):
        if n >= k:
            state = state[1:] + [sum(r * state[-1 - i] for i, r in enumerate(recurrence)) % Q]
        if state[min(n, k - 1)] == 0:
            candidates.append(n)
    state = [v % Q for v in initial]
    for m in range(1, window + 1):
        head = (state[k - 1] - sum(r * state[k - 2 - i] for i, r in enumerate(recurrence[:-1])))
        state = [head * inverse % Q] + state[:-1]
        if state[0] == 0:
            candidates.append(-m)
    # each candidate checked exactly, up to the farthest one on each side
    forward = list(initial)
    while len(forward) <= max([0] + candidates):
        forward.append(sum(r * forward[-1 - i] for i, r in enumerate(recurrence)))
    backward = [Fraction(v) for v in initial]
    for _ in range(-min([0] + candidates)):
        head = backward[k - 1] - sum(
            r * backward[k - 2 - i] for i, r in enumerate(recurrence[:-1])
        )
        backward.insert(0, head / recurrence[-1])
    M = len(backward) - k  # backward = [u_{-M}, ..., u_{k-1}]
    return sorted(n for n in candidates if (forward[n] if n >= 0 else backward[M + n]) == 0)


def certify(F, step, x0, condition, zeros_in, window, *, back=None, invertible,
            records=(), mode="abs", primes=8, depth_below=50, label=None):
    """The sieve over the places. `zeros_in(w)` gives the exact zeros with `|n| <= w`
    (`n >= 0` for a map that is not invertible), or `None` if they are out of reach;
    `back = (F, step, x0)` is the orbit in the time `-n` when it has its own primes;
    `records` are records found beforehand (the real place). `mode` is `"abs"`, or
    `"sieve"` for Strassmann at etale primes only. Returns a dict with `certified`,
    and when certified the `zeros` and the record types that settle them (`by`), plus
    the primes used and the `records`; `known` are the exact zeros with `|n| <=
    searched`, certified or not. `condition` is `g` or a target (`as_condition`);
    `label` tags the internal soundness checks.
    """
    records, condition = list(records), as_condition(condition, len(x0))
    result = {"certified": False, "primes": []}
    zeros, seen, etale, queue, types = None, None, 0, None, {}
    while True:
        need = max([window] + [r["T"] for r in records])
        if need != seen:
            zeros, seen = zeros_in(need), need
        if zeros is None:
            result["error"] = "hit too far for integers"
            break
        for r in records:
            side = [-n for n in zeros] if r["sides"] == "backward" else zeros
            assert sound(r, side), (label, r["p"], r["type"])
        by = settle(records, zeros, need, invertible)
        if by is not None:
            result |= {"certified": True, "by": by, "zeros": zeros}
            break
        if queue is None:
            types = {p: b for p in PRIMES if (b := branch(F, step, x0, p)) is not None}
            # Cheapest first. ABS adds the orbit modulo p^k: forward (and, for `back`,
            # backward) at the singular primes, both ways at the etale ones, where it
            # costs far less than Orbit.times and gives the same exclusions; p = 2 has
            # only this.
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
            records.append(etale_record(F, x0, condition, p))
            etale += 1
        elif how == "depth":
            sides = "both" if t == "etale" else "forward"
            kind = t if t != "etale" else "etale mod p^k"
            if rec := depth_record(step, x0, condition, p, kind, sides):
                records.append(rec)
        else:
            Fb, stepb, s = back
            tb = (branch(Fb, stepb, s, p) or ("mixed",))[0]
            if rec := depth_record(stepb, s, condition, p, tb, "backward"):
                records.append(rec)
    result["primes"] = sorted({r["p"] for r in records if r["p"]})
    result["records"] = records
    result["known"], result["searched"] = (zeros, seen) if zeros is not None else ([], None)
    return result


@dataclasses.dataclass(frozen=True)
class ZeroSet:
    """A zero set found by ABS. When `certified`, the times are exactly `zeros`, or,
    for a periodic orbit, `zeros + period Z` with `zeros` in `[0, period)`; `by` names
    the records that settle it and `primes` the primes used. Certified or not, `known`
    are the exact zeros with `|n| <= searched` (`0 <= n` for maps), found in integers.
    """

    certified: bool
    zeros: tuple = ()
    period: int | None = None
    by: str | None = None
    primes: tuple = ()
    records: tuple = dataclasses.field(default=(), compare=False, repr=False)
    known: tuple = dataclasses.field(default=(), compare=False, repr=False)
    searched: int | None = dataclasses.field(default=None, compare=False, repr=False)


def _zero_set(found) -> ZeroSet:
    return ZeroSet(found["certified"], tuple(found.get("zeros", ())), None,
                   found.get("by"), tuple(found["primes"]), tuple(found["records"]),
                   tuple(found["known"]), found["searched"])


def recurrence_zeros(r, u=None, *, primes=8, depth_below=50, window=200,
                     real=2) -> ZeroSet:
    """Every `n` in `Z` with `u_n = 0`, for `u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}`
    with integer `r`, `r_{k-1} != 0`, and initial values `u = (u_0, ..., u_{k-1})`. `r`
    may also be a Sage `BinaryRecurrenceSequence` (`u_n = b u_{n-1} + c u_{n-2}`)."""
    if u is None and hasattr(r, "u0"):
        r, u = (r.b, r.c), (r.u0, r.u1)
    r, u = tuple(int(v) for v in r), tuple(int(v) for v in u)
    if len(r) != len(u) or not r[-1] or not any(u):
        raise ValueError("Need len(r) == len(u), r[-1] != 0 and u != 0.")
    F, step = recurrence_map(r)
    rb, s = reversed_recurrence(r, u)
    back = (*recurrence_map(rb), s)
    pair = (real_record(r, u, "forward", level=real),
            real_record(rb, s, "backward", level=real))
    found = certify(F, step, u, 0, lambda w: exact_zeros(list(r), list(u), w), window,
                    back=back, invertible=True, records=[rec for rec in pair if rec],
                    primes=primes, depth_below=depth_below, label=(r, u))
    return _zero_set(found)


def map_zeros(F, x0, target, *, primes=8, depth_below=50, window=120) -> ZeroSet:
    """Every `n >= 0` with `x_n = target` along the integer orbit of `x0` under the
    polynomial map `F` of `Z^d` (`x` the first coordinate)."""
    return orbit_zeros(F, x0, target, primes=primes, depth_below=depth_below,
                       window=window)


def orbit_zeros(F, x0, g, *, primes=8, depth_below=50, window=120) -> ZeroSet:
    """Every `n >= 0` with `g(x_n) = 0`, `g` an integer polynomial or a target as in
    `as_condition`."""
    x0 = tuple(int(v) for v in x0)
    F, g = [as_poly(f, len(x0)) for f in F], as_condition(g, len(x0))
    step = stepper(F)
    found = certify(F, step, x0, g, lambda w: window_zeros(step, x0, g, w),
                    window, invertible=False, primes=primes, depth_below=depth_below,
                    label=(F, x0, g))
    return _zero_set(found)


def henon_zeros(c, P, start, target) -> ZeroSet:
    """Every `n` in `Z` with `x_n = target` for the Henon map
    `(x, y) -> (y, c x + P(y))`, `c = +-1`, `P` given by its coefficients, decided at
    the real place by the escape filtration."""
    how, zeros, period = henon_real(c, list(P), tuple(start), target)
    return ZeroSet(True, tuple(zeros), period, f"real ({how})", ())
