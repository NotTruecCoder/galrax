"""The p-adic Eliasson--Vey normal form of an integrable system near a non-degenerate
critical point of rank 0, `F o Phi = G(q)`, with a certificate.

Coordinates are `(x_1, ..., x_n, y_1, ..., y_n)` with `{x_a, y_b} = delta_ab`, so
`{f, g} = sum_a df/dx_a dg/dy_a - df/dy_a dg/dx_a`. Degree by degree the scheme
solves the joint homological equation `{q_i, W} - H_i(q) = -R_i` for all `i` at once,
in the original coordinates and through the Smith form of the stacked operator over
`Z_p`: no eigencoordinates and no extension of `Q_p` are needed, so every Williamson
type (elliptic, hyperbolic, focus-focus, ...) is handled alike. The joint kernel is
`Q_p[q]` exactly when the quadratic parts `q` span a Cartan subalgebra.
"""

import math
from fractions import Fraction

import numpy as np

from .arithmetic import factorial_valuation, valuation
from .normal_form import _monomials, _multiply, _solve, _substitute, smith
from .polynomial import parse_polynomial, reduce


class _Retry(Exception):
    pass


def _bracket(f: dict, g: dict, n: int, m: int | None = None, top: int | None = None) -> dict:
    """`{f, g}`, exactly or modulo `m`, without the terms of degree above `top`."""
    out = {}
    for a in range(n):
        for sign, i, j in ((1, a, n + a), (-1, n + a, a)):
            for e1, c1 in f.items():
                if not e1[i]:
                    continue
                for e2, c2 in g.items():
                    if not e2[j]:
                        continue
                    e = [u + v for u, v in zip(e1, e2)]
                    e[i] -= 1
                    e[j] -= 1
                    if top is not None and sum(e) > top:
                        continue
                    e = tuple(e)
                    out[e] = out.get(e, 0) + sign * c1 * e1[i] * c2 * e2[j]
    if m is not None:
        out = {e: c % m for e, c in out.items()}
    return {e: c for e, c in out.items() if c}


def _add(f: dict, g: dict, m: int, scale: int = 1) -> dict:
    out = dict(f)
    for e, c in g.items():
        out[e] = (out.get(e, 0) + scale * c) % m
    return {e: c for e, c in out.items() if c}


def _part(f: dict, D: int) -> dict:
    return {e: c for e, c in f.items() if sum(e) == D}


def _upto(f: dict, D: int) -> dict:
    return {e: c for e, c in f.items() if sum(e) <= D}


def _slope(polys, p: int, K: int, weight) -> Fraction | None:
    """The least `v(c) / weight(e)` over the terms with `weight(e) > 0`."""
    ratios = [Fraction(valuation(c, p, K), weight(e))
              for f in polys for e, c in f.items() if weight(e) > 0 and c % p**K]
    return min(ratios, default=None)


class EliassonVey:
    """`F o Phi = G(q)` modulo `p^K` on the unit polydisc, in the coordinates
    `x = p^rho z`: with `F~(z) = p^{-2 rho} F(p^rho z)`,

    `F~(Phi(z)) = G(q(z))`  modulo `p^K`  for every `z` in `Z_p^{2n}`,

    where `q` are the quadratic parts of `F`, `Phi(z) = z + O(z^2)` is symplectic
    modulo `p^K` and `G(u) = u + O(u^2)`. So `F(p^rho Phi(z)) = p^{2 rho} G(q(z))`
    modulo `p^{K + 2 rho}`, on the ball `|x| <= p^{-rho}`.

    **Attributes:** `p`, `precision` (`K`), `radius` (`rho`), `degree` (`M`, the
    degree of `Phi`), `quadratic` (`q`), `transformation` (`Phi - z`, one dict per
    coordinate), `G` (one dict in `u` per function), `divisors` (the largest Smith
    exponent of the joint operator in each degree) and `certificate`.
    """

    def __init__(self, system, precision: int = 6, radius: int | None = None,
                 max_degree: int = 10):
        self.system, self.p, self.precision = system, system.p, int(precision)
        self._work = 2 * self.precision + 2  # digits lost to the divisors stay below K
        self._smith = {}
        for rho in range(1, 2 * self.precision + 1) if radius is None else [int(radius)]:
            try:
                self._build(rho, max_degree)
                break
            except _Retry:
                continue
        else:
            raise ValueError("No radius certifies the normal form up to `max_degree`; "
                             "raise it, or the critical point is degenerate.")
        self.radius = rho
        self.certificate = Certificate(self)
        if not self.certificate.verify():
            raise RuntimeError("The normal form failed its own certificate.")

    def _operator(self, D: int):
        """The stacked matrix of `(W, h_1, ..., h_n) -> ({q_i, W} - h_i(q))_i` on
        degree `D`, with its monomials and Smith decomposition (cached)."""
        if D not in self._smith:
            p, K, n = self.p, self._work, self.system.n
            m = p**K
            rows = _monomials(2 * n, D)
            index = {e: k for k, e in enumerate(rows)}
            gammas = _monomials(n, D // 2) if D % 2 == 0 else []
            P, G = len(rows), len(gammas)
            M = np.zeros((n * P, P + n * G), dtype=object)
            for i, q in enumerate(self.quadratic):
                for col, w in enumerate(rows):
                    for e, c in _bracket(q, {w: 1}, n, m).items():
                        M[i * P + index[e], col] = c
                for col, g in enumerate(gammas):
                    for e, c in _substitute({g: 1}, self.quadratic, D, m).items():
                        M[i * P + index[e], P + i * G + col] = -c % m
            self._smith[D] = (rows, gammas, M, smith(M, p, K))
        return self._smith[D]

    def _build(self, rho: int, top: int):
        p, K, n, work = self.p, self.precision, self.system.n, self._work
        m, d = p**work, 2 * self.system.n
        F = [{e: c * p ** (rho * (sum(e) - 2)) % m for e, c in f.items()}
             for f in reduce(self.system.F, m)]
        F = [{e: c for e, c in f.items() if c} for f in F]
        self.quadratic = [_part(f, 2) for f in F]
        unit = [tuple(int(k == j) for k in range(d)) for j in range(d)]
        phi = [{unit[j]: 1} for j in range(d)]
        G = [{tuple(int(k == i) for k in range(n)): 1} for i in range(n)]
        composed = [_upto(f, top) for f in F]  # F~ o Phi
        Gq = [dict(q) for q in self.quadratic]  # G(q)
        self.divisors = {}
        for D in range(3, top + 1):
            rows, gammas, M, decomposition = self._operator(D)
            R = [_part(_add(c, g, m, -1), D) for c, g in zip(composed, Gq)]
            vector = [-R[i].get(e, 0) % m for i in range(n) for e in rows]
            solution = _solve(M, vector, p, work, decomposition)
            if solution is None:
                raise _Retry
            P, Gn = len(rows), len(gammas)
            W = {e: c for e, c in zip(rows, solution[:P]) if c}
            if any(c % p for c in W.values()):
                raise _Retry  # exp(ad_W) divides by k!: needs W = 0 mod p
            self.divisors[D] = max((s for s in decomposition[3] if s < work), default=0)
            Wp = {e: c // p for e, c in W.items()}
            composed = [self._exp(Wp, f, top) for f in composed]
            phi = [self._exp(Wp, f, top) for f in phi]
            for i in range(n):
                H = {g: c for g, c in zip(gammas, solution[P + i * Gn: P + (i + 1) * Gn]) if c}
                G[i] = _add(G[i], H, m)
                Gq[i] = _add(Gq[i], _substitute(H, self.quadratic, D, m), m)
            # certified at degree D?
            k = p**K
            H = [{e: c % k for e, c in f.items() if 2 <= sum(e) <= D and c % k} for f in phi]
            Gc = [{g: c % k for g, c in f.items() if 2 * sum(g) <= D and c % k} for f in G]
            eta = [_slope(H, p, K, lambda e: sum(e) - 1),
                   _slope(F, p, K, lambda e: sum(e) - 2 if sum(e) > 2 else 0),
                   _slope(Gc, p, K, lambda g: 2 * sum(g) - 2)]
            eta = min((s for s in eta if s is not None), default=Fraction(K))
            if eta > 0 and eta * (D - 1) >= K:
                self.degree, self.slope = D, eta
                self.transformation, self.G = H, Gc
                return
        raise _Retry

    def _exp(self, Wp: dict, f: dict, top: int) -> dict:
        """`exp(ad_W) f = f o phi_W` up to degree `top`, for `W = p Wp`."""
        p, n, m = self.p, self.system.n, self.p**self._work
        out, term, k = dict(f), f, 0
        while term:
            k += 1
            term = _bracket(term, Wp, n, m, top)
            v = factorial_valuation(k, p)
            coefficient = p ** (k - v) * pow(math.factorial(k) // p**v, -1, m) % m  # p^k/k!
            out = _add(out, term, m, coefficient)
        return out

    def __repr__(self) -> str:
        return (f"EliassonVey(p={self.p}, precision={self.precision}, "
                f"radius={self.radius}, degree={self.degree})")


class Certificate:
    """A checkable proof of an `EliassonVey` normal form. With `F~` the scaled system,
    `Phi = z + H` and `G` the computed polynomials (`H` of degrees `2..M`, `G - u` of
    `u`-degrees `2..M/2`), if

    1. `F~ o Phi - G(q)` vanishes modulo `p^K` in every degree `<= M`;
    2. `{Phi_a, Phi_b}` equals the standard form modulo `p^K` in every degree `< M`;
    3. `v(H_n) >= eta (n - 1)`, `v(F~_D) >= eta (D - 2)` for `D >= 3` and
       `v(G_gamma) >= eta (2|gamma| - 2)`, with `eta > 0` and `eta (M - 1) >= K`,

    then every term of `F~ o Phi - G(q)` of degree `D > M` has valuation at least
    `eta (D - 2) >= K`, and every term of `{Phi_a, Phi_b}` of degree `D >= M` at least
    `eta D >= K`: the identity and symplecticity hold modulo `p^K` on `Z_p^{2n}`.
    `verify` checks 1-3 from the system itself, in exact integer arithmetic.
    """

    def __init__(self, form):
        self._form = form
        self.p, self.precision, self.radius = form.p, form.precision, form.radius
        self.degree, self.slope = form.degree, form.slope

    def verify(self) -> bool:
        form = self._form
        p, K, M, rho = self.p, self.precision, self.degree, self.radius
        n, m = form.system.n, p**K
        d = 2 * n
        F = [{e: c * p ** (rho * (sum(e) - 2)) % m for e, c in f.items()}
             for f in reduce(form.system.F, m)]
        q = [_part(f, 2) for f in F]
        unit = [tuple(int(k == j) for k in range(d)) for j in range(d)]
        if any(sum(e) < 2 or sum(e) > M for h in form.transformation for e in h):
            return False
        for i, g in enumerate(form.G):
            u = tuple(int(k == i) for k in range(n))
            if g.get(u) != 1 or any(2 * sum(e) > M or (sum(e) < 2 and e != u) for e in g):
                return False
        phi = [_add({unit[j]: 1}, h, m) for j, h in enumerate(form.transformation)]
        for f, g in zip(F, form.G):
            if _add(_substitute(f, phi, M, m), _substitute(g, q, M, m), m, -1):
                return False
        for a in range(d):
            for b in range(a + 1, d):
                target = {(0,) * d: 1} if b == a + n else {}
                if _add(_bracket(phi[a], phi[b], n, m, M - 1), target, m, -1):
                    return False
        Gh = [{g: c for g, c in f.items() if sum(g) >= 2} for f in form.G]
        eta = self.slope
        bounds = [_slope(form.transformation, p, K, lambda e: sum(e) - 1),
                  _slope(F, p, K, lambda e: sum(e) - 2 if sum(e) > 2 else 0),
                  _slope(Gh, p, K, lambda g: 2 * sum(g) - 2)]
        return eta > 0 and eta * (M - 1) >= K and all(b is None or b >= eta for b in bounds)

    def __repr__(self) -> str:
        return (f"Certificate(p={self.p}, precision={self.precision}, radius={self.radius}, "
                f"degree={self.degree}, slope={self.slope})")


class IntegrableSystem:
    """`n` Poisson-commuting polynomials `F = (f_1, ..., f_n)` in
    `(x_1, ..., x_n, y_1, ..., y_n)`, `{x_a, y_b} = delta_ab`, with `p`-adic integer
    coefficients and a critical point of rank 0 at the origin (`F(0) = 0`,
    `dF(0) = 0`). Each `f_i` is `{exponents: coefficient}` with `2n` exponents.
    Commutation is checked exactly.
    """

    def __init__(self, p: int, F):
        F = list(F)
        n = len(F)
        self.p, self.n = int(p), n
        self.F = [parse_polynomial(f, 2 * n, self.p) for f in F]
        if any(sum(e) < 2 for f in self.F for e in f):
            raise ValueError("The origin must be a critical point: F(0) = 0, dF(0) = 0.")
        for i in range(n):
            for j in range(i + 1, n):
                if _bracket(self.F[i], self.F[j], n):
                    raise ValueError(f"{{f_{i}, f_{j}}} != 0: the system is not integrable.")

    def normal_form(self, precision: int = 6, radius: int | None = None,
                    max_degree: int = 10) -> EliassonVey:
        """`F o Phi = G(q)` near the origin (p-adic Eliasson--Vey, rank 0), certified
        modulo `p^precision`; see `EliassonVey`. `radius` is found by default.
        """
        return EliassonVey(self, precision, radius, max_degree)
