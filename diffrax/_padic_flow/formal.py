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

from .arithmetic import valuation
from .flow import PAdicFlow
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
        cache = {"coefficients": []}

        def omega(n):
            if n >= len(cache["coefficients"]):
                cache["coefficients"] = weierstrass_differential(a, 2 * n + 16)
            return cache["coefficients"][n]

        return cls(p, precision, omega, **kwargs)

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
