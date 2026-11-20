"""Orbit closures on surfaces with three involutions: Markoff and Wehler.

A surface F = 0 in A^3 of degree <= 2 in each variable, F = A_i x_i^2 + B_i x_i + C_i,
carries the involutions s_i: x_i -> -B_i / A_i - x_i (the other root). Two families:

    markoff D:  x^2 + y^2 + z^2 - xyz - D                         (A_i = 1)
    wehler a:   (1 + x^2)(1 + y^2)(1 + z^2) + a xyz - 2           (a (2, 2, 2) surface)

For the Wehler family p = 3 mod 4, so that A_i = (1 + x_j^2)(1 + x_k^2) is a unit and
the s_i are analytic on all of Z_p^3.

Near a smooth point q, u -> q + p u (two free coordinates, the third by Hensel) is a
chart of the residue disc of q onto Z_p^2. For t = s_i o s_j, n the return time of q
modulo p under t, and N the order modulo p of the affine map t^n in the chart, t^{nN}
is the identity modulo p in the chart, coefficientwise. OrbitClosure on these three
maps (one per pair) gives levels from the generators alone (depth 1) and with their
commutators; when the rank is 2 at level k, the orbit closure of q contains the ball
of radius p^-k in the chart, that is the points of the surface congruent to q modulo
p^(k + 1).

Each lift is printed with three level sets: the generators alone, with their
commutators, and with the conjugates by s_0, s_1, s_2 as well. Commutators are the
identity modulo p^(2c), so they only lower levels above 2c; conjugates can fill
level 1. Seed 0 gives, for example:

    python orbit_closure.py markoff 5 1 1,5,9,19,21,24
        (1, 2) -> (1, 2) -> (1, 1) at 6 of 118 lifts, (2, 2) throughout at 1
    python orbit_closure.py wehler 3 2 1,2,4,5,10
        (1, 3) -> (1, 2) -> (1, 1) at 3 of 160 lifts: [g1, g2] lowers the level
        (1, 2) -> (1, 2) -> (1, 1) at 34, (2, 2) throughout at 3
    python orbit_closure.py wehler 7 1 1,3
        (1, 2) -> (1, 2) -> (1, 1) at 8 of 108 lifts

Usage: python orbit_closure.py markoff|wehler p lifts parameter1,parameter2,...
"""

import random
import sys
from collections import Counter

from galrax.padic import OrbitClosure
from galrax.padic.normal_form import _multiply


K = 6  # digits in the chart
PAIRS = ((1, 2), (2, 0), (0, 1))  # t = s_i o s_j, one per fixed coordinate


def _add(*terms, m):
    """A sum of `(sign, polynomial)` pairs modulo `m`."""
    out = {}
    for sign, poly in terms:
        for e, v in poly.items():
            out[e] = (out.get(e, 0) + sign * v) % m
    return {e: v for e, v in out.items() if v}


def _product(*factors, top, m):
    out = {(0, 0): 1}
    for factor in factors:
        out = _multiply(out, factor, top, m)
    return out


def markoff(D):
    return {(2, 0, 0): 1, (0, 2, 0): 1, (0, 0, 2): 1, (1, 1, 1): -1, (0, 0, 0): -D}


def wehler(a):
    F = {(1, 1, 1): a, (0, 0, 0): -1}  # (1 + x^2)(1 + y^2)(1 + z^2) - 2 + a xyz
    for e in ((i, j, k) for i in (0, 2) for j in (0, 2) for k in (0, 2)):
        F[e] = F.get(e, 0) + 1
    return {e: v for e, v in F.items() if v}


class Surface:
    def __init__(self, F, p, top=K + 1):
        self.F, self.p, self.top = F, p, top
        self.m = p ** (top + 1)
        self.parts = []  # (A_i, B_i) as polynomials in all three variables
        for i in range(3):
            A, B = {}, {}
            for e, v in F.items():
                rest = e[:i] + (0,) + e[i + 1 :]
                if e[i] == 2:
                    A[rest] = v
                elif e[i] == 1:
                    B[rest] = v
            self.parts.append((A, B))

    # Points, modulo m.
    def value(self, poly, q, m):
        return (
            sum(v * q[0] ** e[0] * q[1] ** e[1] * q[2] ** e[2] for e, v in poly.items())
            % m
        )

    def gradient(self, q, m):
        out = []
        for i in range(3):
            d = {
                e[:i] + (e[i] - 1,) + e[i + 1 :]: v * e[i]
                for e, v in self.F.items()
                if e[i]
            }
            out.append(self.value(d, q, m))
        return out

    def involution(self, q, i, m):
        A, B = (self.value(f, q, m) for f in self.parts[i])
        r = list(q)
        r[i] = (-B * pow(A, -1, m) - q[i]) % m
        return r

    def points(self):
        p = self.p
        grid = [(x, y, z) for x in range(p) for y in range(p) for z in range(p)]
        return [
            q
            for q in grid
            if self.value(self.F, q, p) == 0 and any(self.gradient(q, p))
        ]

    def lift(self, point, rng):
        p, m = self.p, self.m
        j = next(i for i, g in enumerate(self.gradient(point, p)) if g)
        q = [(v + p * rng.randrange(m)) % m for v in point]
        for _ in range(self.top + 3):
            q[j] = (
                q[j] - self.value(self.F, q, m) * pow(self.gradient(q, m)[j], -1, m)
            ) % m
        assert self.value(self.F, q, m) == 0
        return q

    # Polynomials in the chart u, modulo m, of degree <= top.
    def substitute(self, poly, P):
        terms = []
        for e, v in poly.items():
            factors = [P[j] for j in range(3) for _ in range(e[j])]
            terms.append((v, _product(*factors, top=self.top, m=self.m)))
        return _add(*terms, m=self.m)

    def inverse(self, A):
        """`1 / A` for `A` with a unit constant term and the others divisible by p."""
        m, a = self.m, pow(A[(0, 0)], -1, self.m)
        r = _add((a, A), (-1, {(0, 0): 1}), m=m)
        out, power = {(0, 0): a}, {(0, 0): a}
        for _ in range(self.top + 1):
            power = _multiply(power, r, self.top, m)
            power = {e: -v % m for e, v in power.items()}
            out = _add((1, out), (1, power), m=m)
        return out

    def chart_involution(self, P, i):
        A, B = (self.substitute(f, P) for f in self.parts[i])
        out = list(P)
        out[i] = _add(
            (-1, _multiply(B, self.inverse(A), self.top, self.m)), (-1, P[i]), m=self.m
        )
        return out

    def chart(self, q):
        """`u -> q + p u` on the free coordinates, and the solved one by Newton."""
        p, m = self.p, self.m
        solved = next(j for j, g in enumerate(self.gradient(q, p)) if g)
        free = [j for j in range(3) if j != solved]
        P = [{(0, 0): v % m} for v in q]
        P[free[0]][(1, 0)], P[free[1]][(0, 1)] = p, p
        inverse = pow(self.gradient(q, m)[solved], -1, m)
        for _ in range(self.top + 2):
            P[solved] = _add(
                (1, P[solved]), (-inverse, self.substitute(self.F, P)), m=m
            )
        return P, free

    def generators(self, q, conjugate=False):
        """The maps `h^-1 t^{nN} h` of the chart of `q`, one per pair, for `h = id`
        and, with `conjugate`, the three involutions; modulo p^K. `words` keeps
        `(h, (i, j), nN)` for each.
        """
        p = self.p
        start, free = self.chart(q)
        out, self.words = [], []
        for h in [()] + ([(0,), (1,), (2,)] if conjugate else []):
            for i, j in PAIRS:
                home, P = [v % p for v in q], start
                for a in h:
                    home, P = self.involution(home, a, p), self.chart_involution(P, a)
                r, n = self.involution(self.involution(home, j, p), i, p), 1
                while r != home:
                    r, n = self.involution(self.involution(r, j, p), i, p), n + 1
                for _ in range(n):
                    P = self.chart_involution(self.chart_involution(P, j), i)
                for a in reversed(h):
                    P = self.chart_involution(P, a)
                g, steps = self._power_to_identity(P, q, free)
                out.append(g)
                self.words.append((h, (i, j), n * steps))
        return out

    def _power_to_identity(self, P, q, free):
        """The chart map of `P`, a map of the disc of `q` to itself, raised to the
        order of its affine part modulo p."""
        p, M = self.p, self.p**K
        phi = [
            {
                e: v // p % M
                for e, v in _add((1, P[f]), (-1, {(0, 0): q[f]}), m=self.m).items()
            }
            for f in free
        ]
        phi = [{e: v for e, v in f.items() if v} for f in phi]
        g, steps = phi, 1
        while any(_affine(f, a, p) for a, f in enumerate(g)):
            g, steps = _compose(phi, g, K, M), steps + 1
        return g, steps


def _affine(f, a, p):
    """Whether component `a` of a chart map differs from `u_a` modulo p (its terms of
    degree >= 2 are divisible by p)."""
    unit = (int(a == 0), int(a == 1))
    return any((f.get(e, 0) - (e == unit)) % p for e in ((0, 0), (1, 0), (0, 1)))


def _compose(f, g, top, m):
    """`f o g` for maps of the chart."""
    out = []
    for poly in f:
        terms = [
            (v, _product(*[g[j] for j in range(2) for _ in range(e[j])], top=top, m=m))
            for e, v in poly.items()
        ]
        out.append(_add(*terms, m=m))
    return out


def run(kind, p, lifts, parameters, seed=0):
    rng = random.Random(seed)
    family = {"markoff": markoff, "wehler": wehler}[kind]
    print(f"=== {kind}, p = {p}, {lifts} lift(s) per point ===")
    total = Counter()
    for parameter in parameters:
        surface = Surface(family(parameter), p)
        found = Counter()
        for point in surface.points():
            for _ in range(lifts):
                q = surface.lift(point, rng)
                maps = surface.generators(q)
                alone = OrbitClosure(p, K, maps, (0, 0), depth=1).levels
                commutators = OrbitClosure(p, K, maps, (0, 0))
                conjugates = OrbitClosure(p, K, surface.generators(q, True), (0, 0))
                found[(alone, commutators.levels, conjugates.levels)] += 1
                if alone != commutators.levels:
                    near = tuple(v % p**2 for v in q)
                    print(
                        f"  {parameter}: q = {near} mod p^2, generators {alone}, "
                        f"with commutators {commutators.levels}: {commutators.words}"
                    )
        print(f"{parameter}: {dict(found)}")
        total.update(found)
    return total


if __name__ == "__main__":
    kind, p, lifts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    run(kind, p, lifts, [int(v) for v in sys.argv[4].split(",")])
