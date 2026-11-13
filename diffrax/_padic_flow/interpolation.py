"""Interpolated iterates and minimal group actions.

An analytic map `g = id + p^c h` of `Z_p^d`, with `c >= 1` (`c >= 2` for `p = 2`), is
the time-one map of exactly one field `X` with coefficients divisible by `p^c`:

    X = sum_{j >= 1} (-1)^{j - 1} / j * Delta_g^j(id),    Delta_g h = h o g - h,

and the flow of `X` is the Mahler interpolation `sum_j C(t, j) Delta_g^j(id)` of the
iterates of `g` (Bell, Poonen). `X = g - id` modulo `p^{c + 1}`.

Several such maps whose displacements `g_i(x_0) - x_0` form a basis of `p^k Z_p^d`
generate a group that acts minimally on the ball `x_0 + p^k Z_p^d`, and the product of
their flows is an explicit chart of that ball.
"""

import math
from functools import cache

import numpy as np

from .arithmetic import is_prime, valuation
from .equilibria import _determinant
from .normal_form import _multiply, smith
from .polynomial import evaluate_exact, parse_field, reduce
from .roots import Ball
from .series import PowerSeries


def _parse(g, p: int, digits: int) -> list[dict]:
    """The map `g` as integer polynomials modulo `p^digits`, one per component."""
    return reduce(parse_field(g, p, digits=digits), p**digits)


def _displacement(g: list[dict], m: int) -> list[dict]:
    """`g - id` modulo `m`."""
    d = len(g)
    out = []
    for j, component in enumerate(g):
        unit = tuple(int(i == j) for i in range(d))
        moved = dict(component)
        moved[unit] = (moved.get(unit, 0) - 1) % m
        out.append({e: c for e, c in moved.items() if c})
    return out


def map_valuation(g, p: int, cap: int) -> int:
    """`v_p(g - id)`, the least valuation of a coefficient of `g - id`, capped at
    `cap`.
    """
    moved = _displacement(_parse(g, p, cap), p**cap)
    return min(
        (valuation(c, p, cap) for component in moved for c in component.values()),
        default=cap,
    )


def time_one_field(p: int, digits: int, g) -> list[dict]:
    """The field `X` whose flow has time-one map `g`, modulo `p^digits`, one
    `{exponents: integer}` dictionary per component.

    `g` is given like a vector field (`[c_0, c_1, ...]` for `d = 1`, or one
    `{exponents: coefficient}` dictionary or `PowerSeries` per component) and must be
    congruent to the identity modulo `p^c`, `c >= 1` (`c >= 2` for `p = 2`).

    The series `sum_j (-1)^{j-1} Delta^j(id) / j` is summed while its terms can be
    non-zero: `v(Delta^j(id)) >= j c` and the division by `j` loses `v_p(j)` digits,
    so the terms with `j c - floor(log_p j) >= digits` vanish, and the others are
    computed modulo `p^{digits + max v_p(j)}`, where the divisions are exact.
    Monomials of degree `e` in the iterates have valuation at least
    `c (e - 1) / (D - 1)`, `D` the degree of `g - id`, so higher degrees are dropped.
    """
    p, digits = int(p), int(digits)
    if not is_prime(p):
        raise ValueError("`p` must be a prime.")
    if digits < 1:
        raise ValueError("`digits` must be at least 1.")
    c = map_valuation(g, p, digits)
    needed = 2 if p == 2 else 1
    d = len(parse_field(g, p, digits=1))
    if c < needed:
        raise ValueError(
            f"The map must be congruent to the identity modulo {p**needed}: "
            f"v_p(g - id) = {c}."
        )
    if c >= digits:
        return [{} for _ in range(d)]
    terms = 1
    while terms * c - (len(np.base_repr(terms, p)) - 1) < digits:
        terms += 1
    guard = max(valuation(j, p, digits) for j in range(1, terms))
    work = digits + guard
    m = p**work
    g = _parse(g, p, work)
    moved = _displacement(g, m)
    top = max((sum(e) for component in moved for e in component), default=1)
    cap = 1 + math.ceil(work * max(top - 1, 0) / c)
    zero = (0,) * d

    @cache
    def power(j: int, k: int) -> dict:
        if k == 0:
            return {zero: 1}
        return _multiply(power(j, k - 1), g[j], cap, m)

    def compose(poly: dict) -> dict:
        total = {}
        for e, coefficient in poly.items():
            term = {zero: coefficient}
            for j, k in enumerate(e):
                if k:
                    term = _multiply(term, power(j, k), cap, m)
            for f, value in term.items():
                total[f] = (total.get(f, 0) + value) % m
        return total

    delta = [{tuple(int(i == j) for i in range(d)): 1} for j in range(d)]
    field = [{} for _ in range(d)]
    for j in range(1, terms):
        new = []
        for poly in delta:
            image = compose(poly)
            for e, value in poly.items():
                image[e] = (image.get(e, 0) - value) % m
            new.append({e: v for e, v in image.items() if v})
        delta = new
        lowest = min(
            (valuation(v, p, work) for poly in delta for v in poly.values()),
            default=work,
        )
        if lowest < min(j * c, work):
            raise RuntimeError(f"Delta^{j}(id) is not divisible by p^{j * c}.")
        v = valuation(j, p, work)
        inverse = pow(j // p**v, -1, m)
        sign = 1 if j % 2 else -1
        for out, poly in zip(field, delta):
            for e, value in poly.items():
                out[e] = (out.get(e, 0) + sign * (value // p**v) * inverse) % m
    modulus = p**digits
    return [
        {e: v % modulus for e, v in component.items() if v % modulus}
        for component in field
    ]


def time_one_series(p: int, g) -> list[PowerSeries]:
    """The field of `time_one_field` as `PowerSeries`, known modulo every power of
    `p`, so that a `PAdicFlow` (and a `NormalForm` built on it) read as many digits as
    they need.
    """
    p = int(p)
    c = map_valuation(g, p, 64)
    needed = 2 if p == 2 else 1
    if c < needed:
        raise ValueError(
            f"The map must be congruent to the identity modulo {p**needed}: "
            f"v_p(g - id) = {c}."
        )
    d = len(parse_field(g, p, digits=1))

    @cache
    def field(digits: int):
        return time_one_field(p, digits, g)

    def component(i: int) -> PowerSeries:
        def terms(q, digits):
            if q != p:
                raise ValueError(f"This field was built for p = {p}, not {q}.")
            return field(max(int(digits), 1))[i]

        def lowest(q):
            if q != p:
                raise ValueError(f"This field was built for p = {p}, not {q}.")
            return c

        return PowerSeries._build(d, terms, lowest)

    return [component(i) for i in range(d)]


class MinimalAction:
    """The group generated by maps `g_1, ..., g_d` of `Z_p^d` congruent to the
    identity modulo `p^c`, near a point `x_0` whose displacements form a basis of
    `p^k Z_p^d`: the matrix with columns `(g_i(x_0) - x_0) / p^k` is invertible over
    `Z_p`.

    Then the group maps the ball `B = x_0 + p^k Z_p^d` to itself and acts minimally on
    it, and with `phi^(i)` the flow of the field of `g_i`,

    `F(t) = phi^(1)_{t_1} o ... o phi^(d)_{t_d}(x_0)`

    is a bijection of `Z_p^d` onto `B` with `|F(t) - F(t')| = p^{-k} |t - t'|`. The
    hypothesis is checked on the values `g_i(x_0)` alone; a `ValueError` says when it
    fails.

    **Arguments:**

    - `p`, `precision`: as for `PAdicFlow`.
    - `maps`: the `d` maps, each given like a vector field.
    - `point`: `x_0`.
    - `**kwargs`: passed to the `PAdicFlow` of each map.

    **Attributes:** `level` (`k`), `ball` (a `Ball`), `flows` (one `PAdicFlow` per map,
    from `PAdicFlow.from_time_one`), `point`.
    """

    def __init__(self, p: int, precision: int, maps, point, **kwargs):
        from .flow import PAdicFlow

        maps = list(maps)
        if not maps:
            raise ValueError("At least one map is needed.")
        flows = [PAdicFlow.from_time_one(p, precision, g, **kwargs) for g in maps]
        p, K, d = flows[0].p, flows[0].precision, flows[0].dimension
        if len(flows) != d or any(f.dimension != d for f in flows):
            raise ValueError(f"Minimality on a ball of Z_p^{d} needs {d} maps of it.")
        x0 = [int(v) for v in np.atleast_1d(np.asarray(point, dtype=object))]
        if len(x0) != d:
            raise ValueError(f"`point` must be a point of Z_p^{d}.")
        m = p**K
        x0 = [v % m for v in x0]
        columns = []
        for g in maps:
            image = [evaluate_exact(f, x0, m) for f in _parse(g, p, K)]
            columns.append([(y - x) % m for y, x in zip(image, x0)])
        k = min(valuation(v, p, K) for column in columns for v in column)
        if k >= K:
            raise ValueError("The maps fix the point to the working precision.")
        residual = [[columns[i][j] // p**k % p for i in range(d)] for j in range(d)]
        if _determinant(residual, p) % p == 0:
            raise ValueError(
                "The displacements g_i(x_0) - x_0 do not form a basis of "
                f"p^{k} Z_p^{d}, so the ball is not shown to be minimal."
            )
        digits = K - k
        speeds = []
        for flow in flows:
            value = np.asarray(flow.vector_field(x0[0] if d == 1 else x0), dtype=object)
            speeds.append([int(v) for v in np.atleast_1d(value)])
        matrix = [[speeds[i][j] // p**k for i in range(d)] for j in range(d)]
        left, _, right, s = smith(matrix, p, digits)
        if any(s):
            raise RuntimeError("X_i(x_0) / p^k is not invertible modulo p.")
        self.p, self.precision, self.dimension = p, K, d
        self.level = k
        self.point = tuple(x0)
        self.ball = Ball(tuple(v % p**k for v in x0), k)
        self.flows = flows
        self._inverse = right @ left % p**digits

    def __repr__(self) -> str:
        return (
            f"MinimalAction(p={self.p}, precision={self.precision}, "
            f"dimension={self.dimension}, level={self.level})"
        )

    def _rows(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        if self.dimension == 1:
            return values.reshape(-1, 1)
        return values.reshape(-1, self.dimension)

    def chart(self, t):
        """`F(t) = phi^(1)_{t_1} o ... o phi^(d)_{t_d}(x_0)` modulo `p^K`. Times have
        the shape of points: `(...)` for `d = 1`, `(..., d)` otherwise.
        """
        d = self.dimension
        times = self._rows(t)
        shape = np.shape(np.asarray(t, dtype=object))
        point = np.array(self.point, dtype=object)
        y = np.broadcast_to(point, times.shape).copy()
        for i in reversed(range(d)):
            flow = self.flows[i]
            moved = flow.flow(times[:, i], y[:, 0] if d == 1 else y)
            y = self._rows(moved)
        return y.reshape(shape)

    def coordinates(self, z):
        """The times `t` with `F(t) = z`, modulo `p^{K - k}`, for `z` in the ball.

        Since `F(t) - F(t') = sum_i (t_i - t'_i) X_i(x_0) + e` with
        `|e| <= p^{-k-1} |t - t'|`, the step `t <- t + C^{-1} (z - F(t)) / p^k`, where
        `C` has columns `X_i(x_0) / p^k`, gains one digit each time.
        """
        p, K, k, d = self.p, self.precision, self.level, self.dimension
        m = p**K
        shape = np.shape(np.asarray(z, dtype=object))
        z = self._rows(z) % m
        if np.any((z - np.array(self.point, dtype=object)) % p**k):
            raise ValueError(f"The point is outside the ball x_0 + p^{k} Z_p^{d}.")
        digits = K - k
        t = np.zeros_like(z)
        for _ in range(digits):
            residual = (z - self._rows(self.chart(t.reshape(shape)))) % m // p**k
            t = (t + residual @ self._inverse.T) % p**digits
        return t.reshape(shape)
