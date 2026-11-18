"""The Skolem problem with time in the Gaussian integers: all `n` in `Z[i]` with

    h(n) = x([n] P_1) + x([n] P_2) - c = 0

on `E_5 x E_6`, `E_D : y^2 = x^3 - D^2 x`, for points `P_1 in E_5(Q)`,
`P_2 in E_6(Q)` and `c` in `Q(i)`. Both curves have CM by `Z[i]`,
`[i](x, y) = (-x, i y)`, so `n` acts on both factors at once, and `[i] z = i z` in
the parameter `z = -x/y`.

At a prime `p = 3 mod 4` of good reduction `p` is inert, both reductions are
supersingular and `E(F_{p^2}) = E[p + 1]`, so `[n] P` reduces to `[n_0] P` on the
class `n = n_0 + (p + 1) t`, `t in Z[i]`. On it

    [n] P = [n_0] P + exp(t log [p + 1] P)

with `log` linear over `Z_p[i] = Z_{p^2}` (CM), so `h` is a power series in ONE
variable `t` over `Z_{p^2}`: by Strassmann its zeros in the class, Gaussian
integers or not, are at most its Weierstrass degree. Its coefficients come from the
values at `t` in `Z` alone (Mahler differences, then Stirling numbers): the
direction `i t` is free, by holomorphy. At a split prime the same function is one
of `a + b iota`, `iota^2 = -1` in `Z_p`, whose zeros in `Z_p^2` are lines, and no
finite count exists: the inert primes are what makes the certificate possible.

Classes whose reduction misses `h = 0` drop out at once; classes with one factor in
the kernel of reduction have a pole and no zero; classes with both use
`h / (x_1 x_2)`. A class of a prime meeting no live class of another prime holds no
zero (the sieve). The instance is certified when the remaining bounds add up to
the exact zeros found with `|Re n|, |Im n| <= window`.

    python benchmarks/cm_skolem.py --count 20
"""

import argparse
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from galrax.padic.arithmetic import factorial_valuation, is_prime, valuation  # noqa: E402
from galrax.padic.formal import EllipticCurve  # noqa: E402


class Gauss:
    """`re + im i` with rational parts."""

    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re, self.im = Fraction(re), Fraction(im)

    @staticmethod
    def of(v):
        return v if isinstance(v, Gauss) else Gauss(v)

    def __add__(self, o):
        o = Gauss.of(o)
        return Gauss(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __neg__(self):
        return Gauss(-self.re, -self.im)

    def __sub__(self, o):
        return self + -Gauss.of(o)

    def __rsub__(self, o):
        return Gauss.of(o) - self

    def __mul__(self, o):
        o = Gauss.of(o)
        return Gauss(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Gauss.of(o)
        norm = o.re * o.re + o.im * o.im
        return self * Gauss(o.re / norm, -o.im / norm)

    def __rtruediv__(self, o):
        return Gauss.of(o) / self

    def __eq__(self, o):
        o = Gauss.of(o)
        return self.re == o.re and self.im == o.im

    def __hash__(self):
        return hash((self.re, self.im))

    def __repr__(self):
        return f"({self.re} + {self.im} i)"


I = Gauss(0, 1)
CURVES = (5, 6)  # E_5, E_6: congruent numbers, rank 1 over Q
POINTS = ((-4, 6), (-3, 9))
BAD = {2, 3, 5}


def curve(D):
    return EllipticCurve((0, 0, 0, -D * D, 0))


def times(E, n, P):
    """`[a + b i] P`, exact."""
    a, b = n
    iP = None if P is None else (-P[0], I * P[1])
    return E.add(E.multiply(a, P), E.multiply(b, iP))


def residue(v, M):
    """`v mod M` as a pair `(re, im)`, or `None` when `v` is not integral at `M`."""
    v = Gauss.of(v)
    out = []
    for part in (v.re, v.im):
        if math.gcd(part.denominator, M) != 1:
            return None
        out.append(part.numerator * pow(part.denominator, -1, M) % M)
    return tuple(out)


def gvaluation(v, p, cap):
    return min(valuation(v[0], p, cap), valuation(v[1], p, cap))


def class_series(p, K, n0, c, setup):
    """Taylor coefficients modulo `p^K`, at the centre of the samples, of `h` on the
    class `n = n0 + (p + 1) t` (pairs over `Z_p[i]`), or a string for a class that
    holds no zero: `"miss"` (reduction off `h = 0`), `"pole"`.
    """
    m = p + 1
    J = -(-K * (p - 1) // (p - 2))  # J (p - 2)/(p - 1) >= K: the rest vanishes mod p^K
    work = p ** (K + factorial_valuation(J - 1, p))
    shift = -(J // 2)
    values, kind = [], None
    for t in range(shift, shift + J):
        n = (n0[0] + m * t, n0[1])  # real t only: the direction i t is free
        xs = []
        for E, P in setup:
            Q = times(E, n, P)
            xs.append(None if Q is None else Q[0])
        integral = [x is not None and residue(x, p) is not None for x in xs]
        here = "A" if all(integral) else "C" if not any(integral) else "pole"
        assert kind in (None, here), "the reduction is constant on a class"
        kind = here
        if kind == "pole":
            return "pole"
        if kind == "A":
            value = residue(xs[0] + xs[1] - c, work)
            if value[0] % p or value[1] % p:  # h mod p is constant on the class
                return "miss"
        else:  # both in the kernel: h / (x_1 x_2) = 1/x_2 + 1/x_1 - c/(x_1 x_2)
            inv = [Gauss(0) if x is None else 1 / x for x in xs]
            value = residue(inv[0] + inv[1] - c * inv[0] * inv[1], work)
        values.append(value)
    # Delta^k h / k!, exact modulo p^K
    d = []
    row = values
    for k in range(J):
        fk = factorial_valuation(k, p)
        unit = math.factorial(k) // p**fk
        assert all(v % p**fk == 0 for v in row[0]), "Mahler coefficient not divisible"
        inv = pow(unit, -1, p**K)
        d.append(tuple(v // p**fk * inv % p**K for v in row[0]))
        row = [tuple((b - a) % work for a, b in zip(u, w)) for u, w in zip(row, row[1:])]
    # binomial(s, k) = sum_j stirling(k, j) s^j / k!: a_j = sum_k d_k stirling(k, j)
    coefficients = [[0, 0] for _ in range(J)]
    stirling = [1]
    for k in range(J):
        for j, s in enumerate(stirling):
            for r in range(2):
                coefficients[j][r] = (coefficients[j][r] + d[k][r] * s) % p**K
        stirling = [0] + stirling  # times (s - k)
        for j in range(len(stirling) - 1):
            stirling[j] -= k * stirling[j + 1]
    coefficients = [tuple(a) for a in coefficients]
    if kind == "C" and n0 == (0, 0):
        # n = 0 sits at s = -shift; h / (x_1 x_2) has a double zero there that is
        # not a zero of h: recentre at s = -shift and divide by s^2.
        coefficients = taylor_shift(coefficients, -shift, p**K)
        assert coefficients[0] == (0, 0) and coefficients[1] == (0, 0)
        coefficients = coefficients[2:]
    return coefficients


def taylor_shift(a, s, M):
    """Coefficients of `f(x + s)` from those of `f`, modulo `M`."""
    a = [list(v) for v in a]
    n = len(a)
    for i in range(n):
        for j in range(n - 2, i - 1, -1):
            for r in range(2):
                a[j][r] = (a[j][r] + s * a[j + 1][r]) % M
    return [tuple(v) for v in a]


def strassmann(coefficients, p, K):
    """The Weierstrass degree (a bound on the zeros in `Z_p[i]`), or `None` when
    every coefficient vanishes modulo `p^K`.
    """
    v = [gvaluation(a, p, K) for a in coefficients]
    low = min(v)
    if low >= K:
        return None
    return max(j for j, w in enumerate(v) if w == low)


def evaluate(coefficients, t, M):
    """`sum a_j t^j` modulo `M` for `t = (re, im)` in `Z[i]`."""
    out, power = (0, 0), (1, 0)
    for a in coefficients:
        out = (
            (out[0] + a[0] * power[0] - a[1] * power[1]) % M,
            (out[1] + a[0] * power[1] + a[1] * power[0]) % M,
        )
        power = (
            (power[0] * t[0] - power[1] * t[1]) % M,
            (power[0] * t[1] + power[1] * t[0]) % M,
        )
    return out


def prime_classes(p, c, setup, K=4):
    """`{n0: bound}` over `n0 mod p + 1` (both parts), or `None` if the bound
    needs more precision somewhere.
    """
    m = p + 1
    tables = [reduction_table(E, P, p) for E, P in setup]
    cbar = residue(c, p)
    out = {}
    for a in range(m):
        for b in range(m):
            R = [table[a][b] for table in tables]
            if (R[0] is None) != (R[1] is None):
                out[(a, b)] = 0  # one pole: no zero
                continue
            if R[0] is not None and any(
                (R[0][0][r] + R[1][0][r] - cbar[r]) % p for r in range(2)
            ):
                out[(a, b)] = 0  # the reduction misses h = 0
                continue
            n0 = (a - m if 2 * a > m else a, b - m if 2 * b > m else b)
            series = class_series(p, K, n0, c, setup)
            if isinstance(series, str):
                out[(a, b)] = 0
                continue
            bound = strassmann(series, p, K)
            if bound is None:
                return None
            out[(a, b)] = bound
    return out


def _f2(u, v, p):
    return ((u[0] * v[0] - u[1] * v[1]) % p, (u[0] * v[1] + u[1] * v[0]) % p)


def _f2_inverse(u, p):
    norm = pow((u[0] * u[0] + u[1] * u[1]) % p, -1, p)
    return (u[0] * norm % p, -u[1] * norm % p)


def _add_f2(a4, P, R, p):
    """The group law of `y^2 = x^3 + a4 x` over `F_{p^2} = F_p[i]`."""
    if P is None:
        return R
    if R is None:
        return P
    (x1, y1), (x2, y2) = P, R
    if x1 == x2 and all((u + v) % p == 0 for u, v in zip(y1, y2)):
        return None
    if x1 == x2:
        top = tuple((3 * w + a4 * (r == 0)) % p for r, w in enumerate(_f2(x1, x1, p)))
        s = _f2(top, _f2_inverse(tuple(2 * v % p for v in y1), p), p)
    else:
        dx = tuple((u - v) % p for u, v in zip(x2, x1))
        s = _f2(tuple((u - v) % p for u, v in zip(y2, y1)), _f2_inverse(dx, p), p)
    x3 = tuple((w - u - v) % p for w, u, v in zip(_f2(s, s, p), x1, x2))
    y3 = tuple((w - v) % p for w, v in zip(_f2(s, tuple((u - w) % p for u, w in zip(x1, x3)), p), y1))
    return (x3, y3)


def reduction_table(E, P, p):
    """`[a + b i] P mod p` over `F_{p^2}` for `0 <= a, b <= p`."""
    a4 = int(E.a[3]) % p
    x, y = int(P[0]) % p, int(P[1]) % p
    Pbar, iPbar = ((x, 0), (y, 0)), ((-x % p, 0), (0, y))
    table, column = [], None
    for _ in range(p + 1):
        row, point = [], column
        for _ in range(p + 1):
            row.append(point)
            point = _add_f2(a4, point, iPbar, p)
        table.append(row)
        column = _add_f2(a4, column, Pbar, p)
    assert column is None, "E(F_{p^2}) = E[p + 1] for supersingular p"
    return table


def sieve(sets):
    """The least, over the primes, of the bounds summed over that prime's classes
    that meet a live class of every other prime.
    """
    best = None
    for p, classes in sets.items():
        total = 0
        for n0, bound in classes.items():
            if bound and all(
                any(
                    b
                    and (n0[0] - m0[0]) % math.gcd(p + 1, q + 1) == 0
                    and (n0[1] - m0[1]) % math.gcd(p + 1, q + 1) == 0
                    for m0, b in other.items()
                )
                for q, other in sets.items()
                if q != p
            ):
                total += bound
        best = total if best is None else min(best, total)
    return best


Q = 10**18 + 9  # a prime = 1 mod 4: candidates are found modulo Q, then checked exactly


def _sqrt_minus_one(q):
    for g in range(2, 100):
        r = pow(g, (q - 1) // 4, q)
        if r * r % q == q - 1:
            return r
    raise ValueError


def _add_mod(a4, P, R, q):
    if P is None:
        return R
    if R is None:
        return P
    (x1, y1), (x2, y2) = P, R
    if x1 == x2 and (y1 + y2) % q == 0:
        return None
    if x1 == x2:
        s = (3 * x1 * x1 + a4) * pow(2 * y1, -1, q) % q
    else:
        s = (y2 - y1) * pow(x2 - x1, -1, q) % q
    x3 = (s * s - x1 - x2) % q
    return (x3, (s * (x1 - x3) - y1) % q)


def _times_mod(a4, k, P, q):
    if k < 0:
        k, P = -k, (P[0], -P[1] % q)
    out = None
    while k:
        if k & 1:
            out = _add_mod(a4, out, P, q)
        P = _add_mod(a4, P, P, q)
        k >>= 1
    return out


def exact_zeros(c, setup, window):
    """The `n` with `|Re n|, |Im n| <= window` and `h(n) = 0`, exact."""
    q, iota = Q, _sqrt_minus_one(Q)
    cq = (c.re.numerator * pow(c.re.denominator, -1, q) + iota * c.im.numerator * pow(c.im.denominator, -1, q)) % q
    rows = []
    for E, P in setup:
        a4 = int(E.a[3])
        Pq = (int(P[0]) % q, int(P[1]) % q)
        iPq = (-int(P[0]) % q, iota * int(P[1]) % q)
        rows.append((a4, Pq, iPq))
    candidates = []
    for b in range(-window, window + 1):
        points = [
            _add_mod(D, _times_mod(D, -window, Pq, q), _times_mod(D, b, iPq, q), q)
            for D, Pq, iPq in rows
        ]
        for a in range(-window, window + 1):
            if any(R is None for R in points) or (points[0][0] + points[1][0] - cq) % q == 0:
                candidates.append((a, b))
            points = [_add_mod(D, R, Pq, q) for R, (D, Pq, _) in zip(points, rows)]
    zeros = []
    for n in candidates:
        xs = [times(E, n, P) for E, P in setup]
        if all(R is not None for R in xs) and xs[0][0] + xs[1][0] == c:
            zeros.append(n)
    return zeros


INERT = tuple(p for p in range(7, 100) if p % 4 == 3 and is_prime(p) and p not in BAD)


def solve(c, setup, window=12, primes=6):
    start, c = time.perf_counter(), Gauss.of(c)
    zeros = exact_zeros(c, setup, window)
    result = {"zeros": zeros, "certified": False, "bound": None, "primes": []}
    sets = {}
    for p in INERT:
        if residue(c, p) is None:
            continue
        classes = prime_classes(p, c, setup)
        if classes is None:
            continue
        sets[p] = classes
        bound = sieve(sets)
        result |= {"primes": list(sets), "bound": bound}
        if bound <= len(zeros):
            assert bound == len(zeros), "the bound counts every zero"
            result["certified"] = True
            break
        if len(sets) == primes:
            break
    return result | {"seconds": time.perf_counter() - start}


def setup():
    return [(curve(D), tuple(Fraction(v) for v in P)) for D, P in zip(CURVES, POINTS)]


def cases(count, rng, spread=3):
    """Targets `c = h`-value at a random `n*`, so that `n*` (and `-n*`) are zeros,
    and as many targets `c` in `Z[i]` drawn at random.
    """
    S = setup()
    out = []
    for k in range(count):
        if k % 2 == 0:
            n = (rng.randint(-spread, spread), rng.randint(-spread, spread))
            xs = [times(E, n, P) for E, P in S]
            if any(R is None for R in xs):
                continue
            out.append(xs[0][0] + xs[1][0])
        else:
            out.append(Gauss(rng.randint(-20, 20), rng.randint(-20, 20)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    S = setup()
    done = 0
    for c in cases(args.count, random.Random(args.seed)):
        r = solve(c, S, args.window)
        done += r["certified"]
        print(
            f"zeros={r['zeros']} bound={r['bound']} primes={r['primes']} "
            f"certified={r['certified']} {r['seconds']:.1f}s"
        )
    print(f"certified {done}")


if __name__ == "__main__":
    main()
