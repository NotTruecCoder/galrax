"""Formal groups as flows, and logarithmic eigenvalues.

A one-dimensional formal group law over `Z_p` has an invariant differential
`omega(z) dz` with `omega in 1 + z Z_p[[z]]` and a logarithm `log(z) = int_0^z omega`.
The field `f(u) = p^c / omega(p u)` on `Z_p` converges, satisfies the hypothesis of
`PAdicFlow`, and its flow is translation in the formal group:

    log(p phi_s(u)) = log(p u) + p^{1 + c} s.

So the rectifying coordinate of this flow around `0` is the formal logarithm, and the
flow-box coordinates of `PAdicFlow.rectify` compute it exactly. For the multiplicative
group, `omega = 1 / (1 + z)` and the field is the linear `p^c + p^{c + 1} u`; for an
elliptic curve it is a power series computed from the Weierstrass coefficients.

The second part concerns linear parts `A = diag(log P_1, ..., log P_d)` whose
eigenvalues are logarithms of points of `G_m` or of an elliptic curve over `Q`. The
small divisors `<m, lambda> - lambda_i` are then logarithms of the points
`sum_j m_j P_j - P_i`, and their valuations, the Smith exponents of the homological
operator, are read off exactly from rational arithmetic on the group.
"""

import math
from fractions import Fraction

import numpy as np

from .arithmetic import simplex, valuation
from .events import EventSet
from .flow import PAdicFlow
from .roots import Ball, vanishing_set
from .series import PowerSeries


def _valuation(c, p: int) -> int | float:
    c = Fraction(c)
    if c == 0:
        return math.inf
    cap = max(abs(c.numerator).bit_length(), abs(c.denominator).bit_length()) + 1
    return valuation(abs(c.numerator), p, cap) - valuation(c.denominator, p, cap)


def _residue(c, modulus: int) -> int:
    c = Fraction(c)
    return c.numerator * pow(c.denominator, -1, modulus) % modulus


def _multiply(a: list, b: list, n: int) -> list:
    out = [Fraction(0)] * n
    for i, ai in enumerate(a[:n]):
        if ai:
            for j, bj in enumerate(b[: n - i]):
                out[i + j] += ai * bj
    return out


def _inverse(a: list, n: int) -> list:
    out = [Fraction(0)] * n
    out[0] = 1 / Fraction(a[0])
    for k in range(1, n):
        total = sum(a[j] * out[k - j] for j in range(1, min(k, len(a) - 1) + 1))
        out[k] = -total * out[0]
    return out


def weierstrass_differential(a, terms: int) -> list[Fraction]:
    """The first `terms` coefficients of `omega(z)`, where `omega(z) dz` is the
    invariant differential `dx / (2y + a_1 x + a_3)` of
    `y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6` in the parameter `z = -x/y`
    (Silverman, *The Arithmetic of Elliptic Curves*, IV.1). For integral `a_i` the
    coefficients are integers and `omega(0) = 1`.
    """
    a1, a2, a3, a4, a6 = (Fraction(c) for c in a)
    n = terms + 4
    w = [Fraction(0)] * n
    for _ in range(n):
        w2 = _multiply(w, w, n)
        w3 = _multiply(w2, w, n)
        new = [Fraction(0)] * n
        new[3] = Fraction(1)
        for k in range(n):
            new[k] += (
                a1 * (w[k - 1] if k >= 1 else 0)
                + a2 * (w[k - 2] if k >= 2 else 0)
                + a3 * w2[k]
                + a4 * (w2[k - 1] if k >= 1 else 0)
                + a6 * w3[k]
            )
        w = new
    # w = z^3 u(z), x = z^{-2} / u, y = -z^{-3} / u.
    v = _inverse(w[3:], n - 3)
    dx = [(k - 2) * v[k] for k in range(len(v))]
    den = [
        -2 * v[k] + (a1 * v[k - 1] if k >= 1 else 0) + (a3 if k == 3 else 0)
        for k in range(len(v))
    ]
    return _multiply(dx, _inverse(den, len(den)), len(den))[:terms]


def weierstrass_omega(a):
    """`n -> ` the coefficient of `z^n` in `omega` of `weierstrass_differential`,
    computed in growing blocks.
    """
    cache = []

    def omega(n):
        if n >= len(cache):
            cache[:] = weierstrass_differential(a, 2 * n + 16)
        return cache[n]

    return omega


class FormalGroupFlow:
    """The flow of `du/ds = p^c / omega(p u)` on `Z_p`, for a one-dimensional formal
    group law over `Z_p` with invariant differential `omega(z) dz`. It is translation
    in the formal group: `log(p phi_s(u)) = log(p u) + p^{1 + c} s`, so
    `p phi_s(u) = F(p u, exp(p^{1 + c} s))`.

    **Arguments:**

    - `p`, `precision`: as for `PAdicFlow`; `u` is read modulo `p^K`, so that the group
        parameter `z = p u` is known modulo `p^{K + 1}`.
    - `omega`: a function `n -> coefficient of z^n in omega`, integers or fractions with
        denominators prime to `p`, with `omega(0) = 1`.
    - `c`: the valuation of the field; at least `1` (`2` for `p = 2`).
    """

    def __init__(self, p: int, precision: int, omega, c: int | None = None, **kwargs):
        p, K = int(p), int(precision)
        if c is None:
            c = 2 if p == 2 else 1
        c = int(c)
        if Fraction(omega(0)) != 1:
            raise ValueError("The invariant differential must have omega(0) = 1.")
        self.p, self.precision, self.c = p, K, c
        self._omega = omega
        self._inverse = [Fraction(1)]

        def inverse(n):
            while len(self._inverse) <= n:
                k = len(self._inverse)
                total = sum(
                    Fraction(omega(j)) * self._inverse[k - j] for j in range(1, k + 1)
                )
                self._inverse.append(-total)
            return self._inverse[n]

        series = PowerSeries(
            lambda e: Fraction(p) ** (c + e[0]) * inverse(e[0]), lambda n, _: c + n
        )
        self.flow = PAdicFlow(p, K, [series], **kwargs)

    @classmethod
    def multiplicative(cls, p: int, precision: int, **kwargs) -> "FormalGroupFlow":
        """`F(X, Y) = X + Y + XY`, `omega = 1 / (1 + z)` and `log = log(1 + z)`; the
        field is the linear `p^c (1 + p u)`.
        """
        return cls(p, precision, lambda n: (-1) ** n, **kwargs)

    @classmethod
    def weierstrass(cls, p: int, precision: int, a, **kwargs) -> "FormalGroupFlow":
        """The formal group of `y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6`,
        with integral `a = (a_1, a_2, a_3, a_4, a_6)`, in the parameter `z = -x/y`.
        """
        return cls(p, precision, weierstrass_omega(a), **kwargs)

    def logarithm(self, z):
        """`log(z)` modulo `p^{K + 1}` for group parameters `z` of valuation at least
        `1 + c`, given modulo `p^{K + 1}`: the rectifying coordinate of `z / p` around
        `0`, times `p^{1 + c}`. No digit is lost: `log` is an isometry there.
        """
        p, c, K = self.p, self.c, self.precision
        z = np.asarray(z, dtype=object) % p ** (K + 1)
        if any(_valuation(int(v), p) < 1 + c for v in z.reshape(-1) if v):
            raise ValueError(f"The parameters must be divisible by p^{1 + c}.")
        u = np.vectorize(lambda v: int(v) // p, otypes=[object])(z)
        s, _ = self.flow.rectify(np.zeros(u.shape, dtype=object), u)
        s = np.asarray(s, dtype=object)
        return s * p ** (1 + c) % p ** (K + 1)

    def exponential(self, w):
        """`exp(w)` modulo `p^{K + 1}` for `w` of valuation at least `1 + c`: the
        point `p phi_{w / p^{1 + c}}(0)`.
        """
        p, c, K = self.p, self.c, self.precision
        w = np.asarray(w, dtype=object) % p ** (K + 1)
        if any(_valuation(int(v), p) < 1 + c for v in w.reshape(-1) if v):
            raise ValueError(f"The logarithms must be divisible by p^{1 + c}.")
        s = np.vectorize(lambda v: int(v) // p ** (1 + c), otypes=[object])(w)
        u = self.flow.flow(s, np.zeros(s.shape, dtype=object))
        return np.asarray(u, dtype=object) * p % p ** (K + 1)

    def multiple(self, t, z):
        """`[t] z = exp(t log z)` modulo `p^{K + 1}`, for every `t` in `Z_p`."""
        p, K = self.p, self.precision
        w = self.logarithm(z)
        t = np.asarray(t, dtype=object) % p ** (K + 1)
        return self.exponential(t * w % p ** (K + 1))


def _product(a: dict, b: dict, top: int) -> dict:
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(i + j for i, j in zip(ea, eb))
            if sum(e) <= top:
                out[e] = out.get(e, 0) + ca * cb
    return out


class FormalGroup:
    """A `d`-dimensional formal group over `Z_p` from its invariant differentials
    `omega_i = sum_j Omega_ij(z) dz_j`, `Omega(0) = I`, so that `d log = Omega dz`.

    The fields `X_i`, the columns of `p^c Omega(p u)^{-1}`, satisfy the hypothesis of
    `PAdicFlow`, commute, and translate the logarithm:
    `log(p phi^i_s(u)) = log(p u) + p^{1 + c} s e_i`. So the chart
    `F(t) = phi^1_{t_1} o ... o phi^d_{t_d}(0)` has `log(p F(t)) = p^{1 + c} t`, and the
    logarithm is `p^{1 + c}` times its coordinates.

    **Arguments:**

    - `p`, `precision`: as for `PAdicFlow`; parameters `z = p u` are known modulo
        `p^{K + 1}`.
    - `omega`: `omega(i, j, e)` is the coefficient of `z^e` in `Omega_ij`, a `p`-adic
        integer.
    - `dimension`: `d`.
    - `c`: the valuation of the fields; at least `1` (`2` for `p = 2`).

    **Attributes:** `fields` (one list of `PowerSeries` per `X_i`), `flows`.
    """

    def __init__(self, p, precision, omega, dimension, c=None, **kwargs):
        p, K, d = int(p), int(precision), int(dimension)
        c = (2 if p == 2 else 1) if c is None else int(c)
        zero = (0,) * d
        if any(
            Fraction(omega(i, j, zero)) != (i == j) for i in range(d) for j in range(d)
        ):
            raise ValueError("The invariant differentials must have Omega(0) = I.")
        self.p, self.precision, self.dimension, self.c = p, K, d, c
        self._omega, self._inverses = omega, {}

        def component(k, i):
            return PowerSeries._build(
                d, lambda q, digits: self._inverse(int(digits))[k][i], lambda q: c
            )

        self.fields = [[component(k, i) for k in range(d)] for i in range(d)]
        self.flows = [PAdicFlow(p, K, field, **kwargs) for field in self.fields]

    @classmethod
    def product(cls, p, precision, omegas, **kwargs) -> "FormalGroup":
        """The product of one-dimensional groups, `omegas[i](n)` the coefficient of
        `z^n` in the `i`-th invariant differential (as for `FormalGroupFlow`).
        """
        d = len(omegas)

        def omega(i, j, e):
            if i != j or any(e[k] for k in range(d) if k != i):
                return 0
            return omegas[i](e[i])

        return cls(p, precision, omega, d, **kwargs)

    def _inverse(self, digits: int):
        """`p^c Omega(p u)^{-1}` cut to the terms that can be non-zero modulo
        `p^digits`: a degree-`n` term has valuation at least `c + n`.
        """
        if digits not in self._inverses:
            p, c, d = self.p, self.c, self.dimension
            top = max(digits - c - 1, 0)
            N = [
                [
                    {
                        e: Fraction(self._omega(i, j, e)) * p ** sum(e)
                        for e in simplex(d, top)
                        if any(e) and self._omega(i, j, e)
                    }
                    for j in range(d)
                ]
                for i in range(d)
            ]
            one = {(0,) * d: Fraction(1)}
            B = [[one if i == j else {} for j in range(d)] for i in range(d)]
            for _ in range(top + 1):  # B = I - N B, one more degree each time
                B = [
                    [
                        _subtract_all(
                            one if i == j else {},
                            [_product(N[i][k], B[k][j], top) for k in range(d)],
                        )
                        for j in range(d)
                    ]
                    for i in range(d)
                ]
            self._inverses[digits] = [
                [{e: v * p**c for e, v in entry.items() if v} for entry in row]
                for row in B
            ]
        return self._inverses[digits]

    def _rows(self, values):
        return np.asarray(values, dtype=object).reshape(-1, self.dimension)

    def chart(self, t):
        """`F(t) = phi^1_{t_1} o ... o phi^d_{t_d}(0)` modulo `p^K`, for times of
        shape `(..., d)`.
        """
        shape = np.shape(np.asarray(t, dtype=object))
        times, d = self._rows(t), self.dimension
        y = np.zeros_like(times)
        for i in reversed(range(d)):
            moved = self.flows[i].flow(times[:, i], y[:, 0] if d == 1 else y)
            y = self._rows(np.asarray(np.asarray(moved), dtype=object))
        return y.reshape(shape)

    def logarithm(self, z):
        """`log(z)` modulo `p^{K + 1}` for parameters `z` (shape `(..., d)`) of
        valuation at least `1 + c`, given modulo `p^{K + 1}`: Newton's method on the
        chart, `t <- t + (z / p - F(t)) / p^c`, gains a digit a step.
        """
        p, c, K = self.p, self.c, self.precision
        shape = np.shape(np.asarray(z, dtype=object))
        z = self._rows(z) % p ** (K + 1)
        if np.any(z % p ** (1 + c)):
            raise ValueError(f"The parameters must be divisible by p^{1 + c}.")
        u, t = z // p, np.zeros_like(z)
        for _ in range(K - c):
            residual = (u - self._rows(self.chart(t))) % p**K
            t = (t + residual // p**c) % p ** (K - c)
        return (t * p ** (1 + c) % p ** (K + 1)).reshape(shape)

    def exponential(self, w):
        """`exp(w)` modulo `p^{K + 1}` for `w` of valuation at least `1 + c`."""
        p, c, K = self.p, self.c, self.precision
        shape = np.shape(np.asarray(w, dtype=object))
        w = self._rows(w) % p ** (K + 1)
        if np.any(w % p ** (1 + c)):
            raise ValueError(f"The logarithms must be divisible by p^{1 + c}.")
        u = self._rows(self.chart(w // p ** (1 + c)))
        return (u * p % p ** (K + 1)).reshape(shape)


def _subtract_all(a: dict, terms: list) -> dict:
    out = dict(a)
    for poly in terms:
        for e, v in poly.items():
            out[e] = out.get(e, 0) - v
    return {e: v for e, v in out.items() if v}


class EllipticCurve:
    """`y^2 + a_1 xy + a_3 y = x^3 + a_2 x^2 + a_4 x + a_6` over `Q`, with exact group
    law on points `(x, y)` of fractions; `None` is the identity.
    """

    def __init__(self, a):
        self.a = tuple(Fraction(c) for c in a)

    def contains(self, P) -> bool:
        if P is None:
            return True
        a1, a2, a3, a4, a6 = self.a
        x, y = P
        return y * y + a1 * x * y + a3 * y == x**3 + a2 * x * x + a4 * x + a6

    def negate(self, P):
        if P is None:
            return None
        a1, _, a3, _, _ = self.a
        x, y = P
        return (x, -y - a1 * x - a3)

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        a1, a2, a3, a4, _ = self.a
        (x1, y1), (x2, y2) = P, Q
        if x1 == x2 and y1 + y2 + a1 * x2 + a3 == 0:
            return None
        if x1 == x2:
            slope = (3 * x1 * x1 + 2 * a2 * x1 + a4 - a1 * y1) / (2 * y1 + a1 * x1 + a3)
        else:
            slope = (y2 - y1) / (x2 - x1)
        x3 = slope * slope + a1 * slope - a2 - x1 - x2
        y3 = -(slope + a1) * x3 - (y1 - slope * x1) - a3
        return (x3, y3)

    def multiply(self, n: int, P):
        if n < 0:
            n, P = -n, self.negate(P)
        out = None
        while n:
            if n & 1:
                out = self.add(out, P)
            P = self.add(P, P)
            n >>= 1
        return out

    def combination(self, coefficients, points):
        out = None
        for k, P in zip(coefficients, points):
            out = self.add(out, self.multiply(int(k), P))
        return out

    @staticmethod
    def parameter(P) -> Fraction:
        """`z = -x/y`."""
        x, y = P
        return -x / y

    @staticmethod
    def depth(P, p: int) -> int | float:
        """`v_p(z(P)) = -v_p(x(P)) / 2` for `P` in the kernel of reduction, `0`
        outside it, and `inf` at the identity.
        """
        if P is None:
            return math.inf
        v = _valuation(P[0], p)
        return -v // 2 if v < 0 else 0

    @staticmethod
    def naive_height(P) -> float:
        """`h(x(P)) = log max(|num|, |den|)`."""
        x = Fraction(P[0])
        return math.log(max(abs(x.numerator), abs(x.denominator), 1))


def height_bound(
    points,
    n: int,
    p: int,
    curve: EllipticCurve | None = None,
    gram_eigenvalue: float | int | None = None,
    mu: float | int | None = None,
) -> int:
    """An upper bound for every finite exponent of `logarithmic_exponents(points, n,
    p, curve)`, from heights alone (points over `Q`, so `D = 1`).

    - For the multiplicative group, `s <= ((n + 1) h_* + log 2) / log p` with
      `h_* = max_j h(alpha_j)`: a non-zero `beta = 1/(gamma - 1)` with
      `v_p(beta) = -s` has `h(beta) >= s log p`, and `h(gamma - 1) <= h(gamma) + log 2`
      with `gamma = prod alpha_j^{k_j}`, `sum |k_j| <= n + 1`. Computed exactly as the
      largest `s` with `p^s <= 2 H_*^{n + 1}`.
    - For an elliptic curve, `s <= (rho(H) (n + 1)^2 + mu_E) / (2 log p)`, where
      `rho(H)` (`gram_eigenvalue`) bounds the largest eigenvalue of the Gram matrix of
      the Neron--Tate height `h^ = lim 4^{-k} h(x([2^k] Q))` on the points, and `mu_E`
      (`mu`) bounds `h(x(Q)) - h^(Q)`. Both must be supplied as upper bounds, for
      example from Silverman's explicit estimates.
    """
    p, n = int(p), int(n)
    if p == 2:
        raise ValueError("The height certificate needs an odd p.")
    if curve is None:
        top = max(
            max(abs(Fraction(a).numerator), Fraction(a).denominator) for a in points
        )
        target = 2 * top ** (n + 1)
        s = 0
        while p ** (s + 1) <= target:
            s += 1
        return s
    if gram_eigenvalue is None or mu is None:
        raise ValueError(
            "The elliptic bound needs `gram_eigenvalue` (rho(H)) and `mu` (mu_E)."
        )
    bound = (float(gram_eigenvalue) * (n + 1) ** 2 + float(mu)) / (2 * math.log(p))
    # Round up a hair so that floating point can only make the bound weaker.
    return math.floor(bound * (1 + 1e-12) + 1e-12)


def linear_degree(
    points,
    p: int,
    threshold: int,
    curve: EllipticCurve | None = None,
    gram_eigenvalue: float | int | None = None,
    mu: float | int | None = None,
) -> int:
    """The last degree `n_sigma` up to which every exponent is at most the threshold
    `sigma` of `NormalForm`, by `height_bound`: for independent points, the normal form
    has no terms of degree `2, ..., n_sigma` besides the linear part. Returns `1` when
    already degree `2` may keep terms.
    """
    n = 1
    while height_bound(points, n + 1, p, curve, gram_eigenvalue, mu) <= threshold:
        n += 1
        if n > 10**6:
            raise RuntimeError("The height bound does not grow.")
    return n


def logarithmic_exponents(points, n: int, p: int, curve: EllipticCurve | None = None):
    """The Smith exponents of the homological operator `L_n` of
    `A = diag(log P_1, ..., log P_d)`, computed from the group law alone.

    For the multiplicative group (`curve=None`) the points are rationals `alpha_j`
    congruent to `1` modulo `p`, and `lambda_j = log_p(alpha_j)`; for an elliptic curve
    they are rational points in the kernel of reduction modulo `p`, and
    `lambda_j = log_E(P_j)`. For `p` odd, `<m, lambda> - lambda_i` is the logarithm of
    `Q = sum_j m_j P_j - P_i` and has the valuation of its group parameter. Returns
    `{(m, i): exponent}` over `|m| = n`, with `inf` exactly when `Q` is the identity.
    """
    from .normal_form import _monomials

    if p == 2:
        raise ValueError("The logarithm is an isometry on the kernel only for odd p.")
    d = len(points)
    out = {}
    for m in _monomials(d, n):
        for i in range(d):
            k = [m[j] - (j == i) for j in range(d)]
            if curve is None:
                q = Fraction(1)
                for kj, alpha in zip(k, points):
                    q *= Fraction(alpha) ** kj
                out[(m, i)] = _valuation(q - 1, p)
            else:
                out[(m, i)] = curve.depth(curve.combination(k, points), p)
    return out


#
# Chabauty--Coleman in residue discs of y^2 = f(x)
#


def _compose(f: list, x: list, n: int) -> list:
    """`f(x(s))` for a polynomial `f` (coefficients) and a series `x`."""
    out = [Fraction(0)] * n
    for c in reversed(f):
        out = _multiply(out, x, n)
        out[0] += c
    return out


def disc_differentials(f, p: int, point, n: int) -> tuple[list, list]:
    """The parameter `s` in `Z_p` of the residue disc of a rational point
    `P_0 = (x_0, y_0)` of `y^2 = f(x)` (good reduction, `p` odd): `x = x_0 + p s` if
    `p` does not divide `y_0`, else `y = y_0 + p s`. Returns the series `x(s)` and
    `w(s)` with `dx / y = w(s) ds`, `n` terms each, in exact rationals.
    """
    f = [Fraction(c) for c in f]
    x0, y0 = (Fraction(v) for v in point)
    if sum(c * x0**k for k, c in enumerate(f)) != y0 * y0:
        raise ValueError("The point is not on the curve.")
    pad = [Fraction(0)] * n
    df = [k * c for k, c in enumerate(f)][1:]
    if _valuation(y0, p) == 0:
        x = ([x0, Fraction(p)] + pad)[:n]
        u = [c / y0**2 for c in _compose(f, x, n)]
        u[0] -= 1
        root, power, binomial = list(pad), [Fraction(1)] + pad[1:], Fraction(1)
        for k in range(n):  # (1 + u)^{-1/2}
            root = [a + binomial * b for a, b in zip(root, power)]
            power, binomial = (
                _multiply(power, u, n),
                binomial * (-Fraction(1, 2) - k) / (k + 1),
            )
        return x, [p * c / y0 for c in root]
    square = _multiply([y0, Fraction(p)], [y0, Fraction(p)], n)
    x = [x0] + pad[1:]
    for _ in range(n.bit_length() + 2):  # Newton on f(x) = y^2
        step = _multiply(
            [a - b for a, b in zip(_compose(f, x, n), square)],
            _inverse(_compose(df, x, n), n),
            n,
        )
        x = [a - b for a, b in zip(x, step)]
    return x, [2 * p * c for c in _inverse(_compose(df, x, n), n)]


def _tiny(f, p: int, point, n: int) -> list[list]:
    """`int_{P_0}^{P(s)} x^k dx / y`, `k < g`, as series in `s`."""
    genus = (len(f) - 2) // 2
    x, w = disc_differentials(f, p, point, n)
    out, power = [], [Fraction(1)] + [Fraction(0)] * (n - 1)
    for _ in range(genus):
        integrand = _multiply(power, w, n)
        out.append([Fraction(0)] + [c / (m + 1) for m, c in enumerate(integrand[:-1])])
        power = _multiply(power, x, n)
    return out


def _parameter(p: int, start, end) -> Fraction:
    (x0, y0), (x1, y1) = start, end
    s = (Fraction(x1) - x0) / p if _valuation(y0, p) == 0 else (Fraction(y1) - y0) / p
    if min(_valuation(Fraction(x1) - x0, p), _valuation(Fraction(y1) - y0, p)) < 1:
        raise ValueError("The points are not in the same residue disc.")
    return s


def tiny_integrals(f, p: int, precision: int, start, end) -> list[int]:
    """`int_start^end x^k dx / y`, `k < g`, modulo `p^precision`, for rational points
    in the same residue disc: the logarithm on the Jacobian of `[end - start]`.
    """
    m, n = p**precision, 2 * precision + 8
    s = _residue(_parameter(p, start, end), m)
    return [
        sum(_residue(c, m) * pow(s, k, m) for k, c in enumerate(series)) % m
        for series in _tiny(f, p, start, n)
    ]


def chabauty_zeros(f, p: int, precision: int, point, annihilator) -> EventSet:
    """The zeros `s` in `Z_p` of `int_{P_0}^{P(s)} omega` in the residue disc of the
    rational point `P_0` (parameter as in `disc_differentials`), for
    `omega = sum_k a_k x^k dx / y` with `p`-adic integer `a_k`. After dividing out
    the content, `zero_bound` is Strassmann's bound, which counts the zeros when
    the balls isolate them. With `omega` killing the Mordell--Weil group, every
    rational point of the disc is among the zeros.
    """
    m, n = p**precision, 2 * precision + 8
    total = [0] * n
    for a, series in zip(annihilator, _tiny(f, p, point, n)):
        a = _residue(a, m)
        total = [(t + a * _residue(c, m)) % m for t, c in zip(total, series)]
    content = min(valuation(c, p, precision) for c in total)
    if content >= precision:
        return EventSet([Ball((0,), 0)], p, 0, None)
    digits = precision - content
    poly = {(k,): c // p**content % p**digits for k, c in enumerate(total) if c}
    bound = max(k for (k,), c in poly.items() if c % p)
    balls = vanishing_set([poly], 1, p, digits, digits)
    return EventSet(sorted(balls, key=lambda b: b.center), p, digits, bound)
