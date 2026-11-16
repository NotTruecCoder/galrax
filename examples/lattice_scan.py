"""Minimality scanner for the four-holed sphere family

    X: x^2 + y^2 + z^2 + xyz = a x + b y + c z + d,

acted on by the Vieta involutions s_i (q_i -> l_i - q_j q_k - q_i, l = (a, b, c)).
The rotation rot_i = s_j o s_k fixes q_i and is affine in (q_j, q_k); its power
rot_i^{n_i}, n_i its order modulo p on the fibre, is the identity modulo p near q,
with field X_i(q) = log(rot_i^{n_i}) (q_j, q_k, 1). Transported along reduced words h,
X_{h^-1 g h}(q) = Dh(q)^{-1} X_g(h q). The Smith exponents (k1, k2) of the span of
these fields in T_q X are printed as in markoff_smith.py (a = b = c = 0 is the
Markoff surface up to z -> -z).

When k1 = k2 = k, two of the maps h^-1 rot_i^{n_i p} h are written in the chart
u -> (q + p u, solved for the last coordinate) of the disc of q, where they are the
identity modulo p, and a MinimalAction is built from them. Its level must be k: it
acts minimally on the ball of radius p^-k in the chart, which is the ball of radius
p^-(k+1) around q on the surface. Odd p only: for p = 2 the flows need rot^n = I
modulo 4, and the order is taken modulo 2.

Usage: python lattice_scan.py p depth lifts a,b,c d1,d2,... [sample]
"""

import random
import sys
from collections import Counter

import jax


jax.config.update("jax_enable_x64", True)

from diffrax import padic_flow as pf  # noqa: E402
from diffrax._padic_flow.normal_form import (  # noqa: E402
    _matmul,
    _matpow,
    _multiply,
    smith,
)


W = 16  # p-adic digits of the fields
K = 4  # digits of the minimal actions


def _others(i):
    return (i + 1) % 3, (i + 2) % 3


def _add(*terms, m):
    """A sum of `(sign, polynomial)` pairs modulo `m`."""
    out = {}
    for sign, poly in terms:
        for e, v in poly.items():
            out[e] = (out.get(e, 0) + sign * v) % m
    return {e: v for e, v in out.items() if v}


def log_unipotent(M, p, digits):
    """log M modulo p^digits for M = I mod p."""
    n, extra = len(M), 4
    m = p ** (digits + extra)
    N = [[(M[i][j] - (i == j)) % m for j in range(n)] for i in range(n)]
    out, power = (
        [[0] * n for _ in range(n)],
        [[int(i == j) for j in range(n)] for i in range(n)],
    )
    for j in range(1, digits + extra + 2):
        power = _matmul(power, N, m)
        v, u = 0, j
        while u % p == 0:
            u, v = u // p, v + 1
        c = (-1) ** (j - 1) * pow(u, -1, p**digits)
        for r in range(n):
            for s in range(n):
                out[r][s] = (out[r][s] + c * (power[r][s] // p**v)) % p**digits
    return out


class Surface:
    """`x^2 + y^2 + z^2 + xyz = a x + b y + c z + d` and its Vieta involutions."""

    def __init__(self, a=0, b=0, c=0, d=0):
        self.linear, self.d = (a, b, c), d
        self._logs = {}

    def equation(self, q, m):
        (x, y, z), (a, b, c) = q, self.linear
        return (x * x + y * y + z * z + x * y * z - a * x - b * y - c * z - self.d) % m

    def gradient(self, q, m):
        return [
            (2 * q[i] + q[_others(i)[0]] * q[_others(i)[1]] - self.linear[i]) % m
            for i in range(3)
        ]

    def involution(self, q, i, m):
        j, k = _others(i)
        r = list(q)
        r[i] = (self.linear[i] - q[j] * q[k] - q[i]) % m
        return r

    def jacobian(self, q, i, m):
        """`Ds_i(q)`, its own inverse."""
        j, k = _others(i)
        J = [[int(r == s) for s in range(3)] for r in range(3)]
        J[i][i], J[i][j], J[i][k] = m - 1, -q[k] % m, -q[j] % m
        return J

    def rotation(self, t, i, m):
        """`rot_i = s_j o s_k` on the fibre `q_i = t`, on `(q_j, q_k, 1)`."""
        j, k = _others(i)
        s_k = [[1, 0, 0], [-t % m, m - 1, self.linear[k] % m], [0, 0, 1]]
        s_j = [[m - 1, -t % m, self.linear[j] % m], [0, 1, 0], [0, 0, 1]]
        return _matmul(s_j, s_k, m)

    def order(self, t, i, p):
        R, M, n = self.rotation(t, i, p), self.rotation(t, i, p), 1
        while M != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            M, n = _matmul(M, R, p), n + 1
        return n

    def field(self, q, i, p):
        """`X_i(q)`, the field of `rot_i^{n_i}`, modulo `p^W`."""
        j, k = _others(i)
        key = (i, q[i] % p ** (W + 4))
        if key not in self._logs:
            n, m = self.order(q[i], i, p), p ** (W + 4)
            self._logs[key] = log_unipotent(
                _matpow(self.rotation(q[i], i, m), n, m), p, W
            )
        A = self._logs[key]
        X = [0, 0, 0]
        X[j], X[k] = [
            (A[r][0] * q[j] + A[r][1] * q[k] + A[r][2]) % p**W for r in (0, 1)
        ]
        return X

    def points(self, p):
        """The smooth points of `X(F_p)`."""
        grid = [(x, y, z) for x in range(p) for y in range(p) for z in range(p)]
        return [
            q for q in grid if self.equation(q, p) == 0 and any(self.gradient(q, p))
        ]

    def lift(self, point, p, rng, digits=W, near=1):
        """A random point of `X(Z_p)` congruent to `point` modulo `p^near`, given
        modulo `p^digits`, by Hensel.
        """
        m = p**digits
        j = next(i for i, g in enumerate(self.gradient(point, p)) if g)
        q = [(v + p**near * rng.randrange(m)) % m for v in point]
        for _ in range(digits + 2):
            q[j] = (q[j] - self.equation(q, m) * pow(self.gradient(q, m)[j], -1, m)) % m
        assert self.equation(q, m) == 0
        return q


def lattice(surface, q, p, depth):
    """Smith exponents of the span of the transported fields at `q`, for words of
    length `<= depth`, and the fields with their `(word, axis)`.
    """
    P = p**W
    identity = [[int(i == j) for j in range(3)] for i in range(3)]
    stack, found = [(q, identity, (), -1)], []
    while stack:
        r, inverse, word, last = stack.pop()
        for i in range(3):
            X = surface.field(r, i, p)
            found.append(
                (
                    [sum(inverse[s][t] * X[t] for t in range(3)) % P for s in range(3)],
                    word,
                    i,
                )
            )
        if len(word) < depth:
            for i in range(3):
                if i != last:
                    J = surface.jacobian(r, i, P)
                    stack.append(
                        (
                            surface.involution(r, i, P),
                            _matmul(inverse, J, P),
                            word + (i,),
                            i,
                        )
                    )
    out = {}
    for L in range(depth + 1):
        rows = [v for v, word, _ in found if len(word) <= L]
        out[L] = sorted(smith(rows, p, W)[3])
    return out, found


def chart_map(surface, q, p, word, i, digits=K):
    """`h^-1 rot_i^{n p} h` in the chart `u` of the disc of `q` (`n` the order of
    `rot_i` at `h q`), modulo `p^digits`: two polynomials in `u`.
    """
    if p == 2:
        raise ValueError("p = 2 needs rot^n = I mod 4; the scanner takes n mod p.")
    m, top = p ** (digits + 1), digits
    solved = next(j for j, g in enumerate(surface.gradient(q, p)) if g)
    free = [j for j in range(3) if j != solved]
    P = [{(0, 0): v % m} for v in q]
    P[free[0]][(1, 0)], P[free[1]][(0, 1)] = p, p
    point = list(q)

    def involution(P, a):
        b, c = _others(a)
        out = list(P)
        out[a] = _add(
            (1, {(0, 0): surface.linear[a]}),
            (-1, _multiply(P[b], P[c], top, m)),
            (-1, P[a]),
            m=m,
        )
        return out

    def equation(P):
        x, y, z = P
        cubic = _multiply(_multiply(x, y, top, m), z, top, m)
        squares = [(1, _multiply(v, v, top, m)) for v in P]
        linear = [(-c, v) for c, v in zip(surface.linear, P)]
        return _add(*squares, (1, cubic), *linear, (-1, {(0, 0): surface.d}), m=m)

    inverse = pow(surface.gradient(q, m)[solved], -1, m)
    for _ in range(digits + 2):
        P[solved] = _add((1, P[solved]), (-inverse, equation(P)), m=m)
    for a in word:
        P, point = involution(P, a), surface.involution(point, a, m)
    j, k = _others(i)
    for _ in range(surface.order(point[i], i, p) * p):
        P = involution(involution(P, k), j)
    for a in reversed(word):
        P = involution(P, a)
    return [
        {e: v // p for e, v in _add((1, P[f]), (-1, {(0, 0): q[f]}), m=m).items()}
        for f in free
    ]


def minimal_action(surface, q, p, found, k):
    """A `MinimalAction` from two transported maps whose fields span `p^k T_q`."""
    for a in range(len(found)):
        for b in range(a):
            if sorted(smith([found[a][0], found[b][0]], p, W)[3]) == [k, k]:
                maps = [chart_map(surface, q, p, *found[c][1:]) for c in (a, b)]
                return pf.MinimalAction(p, K, maps, (0, 0))
    raise ValueError(f"No two fields span p^{k} T_q.")


def run(p, linear, ds, depth, lifts, sample=10, seed=0):
    rng = random.Random(seed)
    print(f"\n=== p = {p}, (a, b, c) = {linear}, words <= {depth}, {lifts} lift(s) ===")
    total, checked = Counter(), Counter()
    for d in ds:
        surface = Surface(*linear, d)
        glob, count = Counter(), 0
        for point in surface.points(p):
            for _ in range(lifts):
                q = surface.lift(point, p, rng)
                exponents, found = lattice(surface, q, p, depth)
                assert exponents[depth][2] >= W - 2, ("not tangent", exponents)
                k1, k2 = exponents[depth][:2]
                glob[(k1, k2)] += 1
                if k1 == k2 < K and count % sample == 0:
                    level = minimal_action(surface, q, p, found, k1).level
                    checked["agree" if level == k1 else "disagree"] += 1
                if (k1, k2) != (1, 1):
                    print(f"  d={d} q={tuple(v % p**2 for v in q)} mod p^2 {(k1, k2)}")
                count += 1
        total.update(glob)
        print(
            f"d={d:4d} points={sum(glob.values()):4d} "
            f"(k1, k2): {dict(sorted(glob.items()))}"
        )
        sys.stdout.flush()
    print(
        "TOTAL", dict(sorted(total.items())), "MinimalAction level = k:", dict(checked)
    )
    return total, checked


if __name__ == "__main__":
    p, depth, lifts = (int(v) for v in sys.argv[1:4])
    linear = tuple(int(v) for v in sys.argv[4].split(","))
    ds = [int(v) for v in sys.argv[5].split(",")]
    run(p, linear, ds, depth, lifts, *(int(v) for v in sys.argv[6:7]))
