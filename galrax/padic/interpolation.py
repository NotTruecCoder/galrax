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

import heapq
import math
from functools import cache

import numpy as np

from .arithmetic import is_prime, valuation
from .equilibria import _determinant
from .normal_form import _multiply, _solve, _substitute, smith
from .polynomial import (
    _derivative,
    _multiply_add,
    evaluate_exact,
    parse_field,
    reduce,
)
from .roots import Ball
from .series import _digits_of_log, PowerSeries


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
    while terms * c - _digits_of_log(terms, p) < digits:
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


def lie_bracket(p: int, digits: int, X, Y) -> list[dict]:
    """`[X, Y] = (X . grad) Y - (Y . grad) X` modulo `p^digits`, for fields given
    like a vector field.
    """
    m = p**digits
    X, Y = _parse(X, p, digits), _parse(Y, p, digits)
    if len(X) != len(Y):
        raise ValueError("The fields must live on the same Z_p^d.")
    out = []
    for i in range(len(X)):
        total = {}
        for j in range(len(X)):
            _multiply_add(total, X[j], _derivative(Y[i], j, m), m)
            minus = {e: -v % m for e, v in Y[j].items()}
            _multiply_add(total, minus, _derivative(X[i], j, m), m)
        out.append({e: v for e, v in total.items() if v})
    return out


def _subtract(a: dict, b: dict, m: int) -> dict:
    out = dict(a)
    for e, v in b.items():
        out[e] = (out.get(e, 0) - v) % m
    return {e: v for e, v in out.items() if v}


def commutator(p: int, digits: int, g, h) -> list[dict]:
    """`g o h o g^{-1} o h^{-1}` modulo `p^digits`, for maps `g = id + p^c ...`,
    `h = id + p^c ...` (`c >= 1`) given like vector fields. It is
    `id - [X, Y]` for the fields `X`, `Y` of `g` and `h`, modulo `p^{3c}` for odd
    `p` and only modulo `p^{3c - 1}` for `p = 2`: the third-order terms carry `1/2`.

    `g^{-1}` is the limit of `y <- id - (g - id) o y`, which gains `c` digits a
    step. A monomial of degree `e` in any word in `g`, `h` and their inverses has
    valuation `>= c (e - 1) / (D - 1)`, `D` the largest degree of `g - id` and
    `h - id`, so higher degrees are dropped.
    """
    m = p**digits
    c = min(map_valuation(g, p, digits), map_valuation(h, p, digits))
    if c < 1:
        raise ValueError("The maps must be congruent to the identity modulo p.")
    g, h = _parse(g, p, digits), _parse(h, p, digits)
    if len(g) != len(h):
        raise ValueError("The maps must act on the same Z_p^d.")
    d = len(g)
    identity = [{tuple(int(i == j) for i in range(d)): 1} for j in range(d)]
    moved = [_displacement(f, m) for f in (g, h)]
    top = max((sum(e) for f in moved for poly in f for e in poly), default=1)
    cap = 1 + math.ceil(digits * max(top - 1, 0) / c)

    def compose(f, k):
        return [_substitute(poly, k, cap, m) for poly in f]

    def inverse(shift):
        y = identity
        for _ in range(-(-digits // c)):
            y = [_subtract(i, f, m) for i, f in zip(identity, compose(shift, y))]
        return y

    words = compose(h, compose(inverse(moved[0]), inverse(moved[1])))
    return compose(g, words)


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


def _binomial(n: int, j: int) -> int:
    """`C(n, j)` for any integer `n`."""
    out = 1
    for i in range(j):
        out *= n - i
    return out // math.factorial(j)


def _inverse(word: tuple) -> tuple:
    return tuple((item, -n) for item, n in reversed(word))


class OrbitClosure:
    """A ball or a disc in the orbit closure of `x_0` under the group generated by
    maps `g_1, ..., g_r` of `Z_p^d` congruent to the identity modulo `p^c`
    (coefficientwise, `c >= 1`, `c >= 2` for `p = 2`), which need not commute and
    whose displacements at `x_0` need not span `Z_p^d`.

    The candidates are the generators and their left-normed commutators
    `[w, g_i] = w g_i w^{-1} g_i^{-1}`, congruent to the identity modulo `p^{n c}` at
    depth `n`. They are taken by increasing level `k = v_p(w(x_0) - x_0)`. When the
    residue of `(w(x_0) - x_0) / p^k` modulo `p` is a combination `sum a_j u_j` of the
    residues of the elements `h_j` kept so far, at levels `k_j <= k`, `w` is replaced
    by `w h_1^{-a_1 p^{k - k_1}} ...`, whose level is larger: with `X` the field of
    `h` and `v_p(n) = k - k_h`, `h^n(x_0) - x_0 = n X(x_0) + O(p^{k + 1})` and
    `X(x_0) = h(x_0) - x_0` modulo `p^{k_h + 1}`. Displacements are computed exactly
    at `x_0`, so the kept elements are words in the group and no estimate on
    Baker-Campbell-Hausdorff terms is needed.

    The kept elements `h_1, ..., h_s` have levels `k_1 <= ... <= k_s` and residues
    independent modulo `p`, so the span of their displacements has Smith exponents
    `levels`. With `k = k_s`, the maps `h_j^{p^{k - k_j}}` satisfy the hypothesis of
    `MinimalAction` on `x_0 + p^k Z_p^d` if `s = d`, and in any case

    `F(t) = (h_1^{p^{k - k_1}})^{t_1} o ... o (h_s^{p^{k - k_s}})^{t_s} (x_0)`

    satisfies `|F(t) - F(t')| = p^{-k} |t - t'|` on `Z_p^s`, with image in the orbit
    closure: the ball `x_0 + p^k Z_p^d` if `s = d`, an `s`-dimensional analytic disc
    (an arc for `s = 1`) otherwise. This is a lower bound on the closure. The depth
    stops when `s = d` and deeper commutators, of level `>= (n + 1) c`, cannot lower
    `k`, or at `depth`. A rank `s < d` is the rank to the working precision, not a
    proof that the closure has dimension `s`.

    **Arguments:**

    - `p`, `precision`: the prime and the digits `K` of all computations.
    - `maps`: the `r` maps, each given like a vector field.
    - `point`: `x_0`.
    - `depth`: the deepest commutators tried.

    **Attributes:** `rank` (`s`), `levels`, `words` (the `h_j` as strings, generators
    numbered from 1), `level` (`k`), `ball` (a `Ball` if `s = d`, else `None`),
    `depth` (the depth reached), `point`.
    """

    def __init__(self, p: int, precision: int, maps, point, depth: int = 4):
        p, K = int(p), int(precision)
        if not is_prime(p):
            raise ValueError("`p` must be a prime.")
        maps = list(maps)
        if depth < 1:
            raise ValueError("`depth` must be at least 1.")
        if not maps:
            raise ValueError("At least one map is needed.")
        needed = 2 if p == 2 else 1
        c = min(map_valuation(g, p, K) for g in maps)
        if c < needed:
            raise ValueError(
                f"The maps must be congruent to the identity modulo {p**needed}."
            )
        self._polys = [_parse(g, p, K) for g in maps]
        d = len(self._polys[0])
        if any(len(g) != d for g in self._polys):
            raise ValueError("The maps must act on the same Z_p^d.")
        x0 = [int(v) % p**K for v in np.atleast_1d(np.asarray(point, dtype=object))]
        if len(x0) != d:
            raise ValueError(f"`point` must be a point of Z_p^{d}.")
        self.p, self.precision, self.dimension, self.point = p, K, d, tuple(x0)
        self._terms = -(-K // c)  # v(Delta^j) >= j c: Mahler terms j < K / c

        generators = [(((i, 1),), f"g{i + 1}") for i in range(len(maps))]
        candidates, layer = [], generators
        for n in range(1, depth + 1):
            candidates += [(w, name, self._displacement(w)) for w, name in layer]
            kept = self._eliminate(candidates)
            top = max((level for level, *_ in kept), default=K)
            if len(kept) == d and (n + 1) * c >= top:
                break
            layer = [
                (w + g + _inverse(w) + _inverse(g), f"[{name}, {gname}]")
                for w, name in layer
                for g, gname in generators
                if w != g
            ]
        self.depth = n
        self.rank = len(kept)
        self.levels = tuple(level for level, *_ in kept)
        self.words = tuple(name for *_, name in kept)
        self.level = top
        self.ball = Ball(tuple(v % p**top for v in x0), top) if self.rank == d else None
        self._chart = [(h, p ** (top - level)) for level, _, h, _ in kept]

    def __repr__(self) -> str:
        return (
            f"OrbitClosure(p={self.p}, precision={self.precision}, "
            f"rank={self.rank}, levels={self.levels})"
        )

    def _apply(self, item, y):
        if isinstance(item, int):
            m = self.p**self.precision
            return [evaluate_exact(f, y, m) for f in self._polys[item]]
        for letter, n in reversed(item):
            y = self._power(letter, n, y)
        return y

    def _power(self, item, n: int, y):
        """`item^n (y)` for any integer `n`, by Mahler interpolation of the iterates."""
        if n == 1:
            return self._apply(item, y)
        m = self.p**self.precision
        rows = [y]
        for _ in range(self._terms - 1):
            rows.append(self._apply(item, rows[-1]))
        if 0 <= n < len(rows):
            return rows[n]
        total = [0] * self.dimension
        for j in range(self._terms):
            b = _binomial(n, j) % m
            total = [(s + b * v) % m for s, v in zip(total, rows[0])]
            rows = [[(v - u) % m for u, v in zip(x, y)] for x, y in zip(rows, rows[1:])]
        return total

    def _displacement(self, word) -> list:
        m = self.p**self.precision
        image = self._apply(word, list(self.point))
        return [(y - x) % m for y, x in zip(image, self.point)]

    def _level(self, v) -> int:
        return min(valuation(x, self.p, self.precision) for x in v)

    def _eliminate(self, candidates) -> list:
        """`(level, residue, word, name)` of the kept elements, by increasing level."""
        p, K = self.p, self.precision
        heap = [
            (self._level(v), i, w, name, v) for i, (w, name, v) in enumerate(candidates)
        ]
        heap = [entry for entry in heap if entry[0] < K]
        heapq.heapify(heap)
        count, kept = len(candidates), []
        while heap and len(kept) < self.dimension:
            k, _, word, name, v = heapq.heappop(heap)
            u = [x // p**k % p for x in v]
            a = None
            if kept:
                columns = np.array([r for _, r, _, _ in kept], dtype=object).T
                a = _solve(columns, u, p, 1)
            if a is None:
                kept.append((k, u, word, name))
                continue
            for (level, _, h, hname), coefficient in zip(kept, a):
                if coefficient:
                    n = -coefficient * p ** (k - level)
                    word, name = word + ((h, n),), f"{name} ({hname})^{n}"
            v = self._displacement(word)
            new = self._level(v)
            if new <= k:
                raise RuntimeError("A reduction did not raise the level.")
            if new < K:
                count += 1
                heapq.heappush(heap, (new, count, word, name, v))
        return kept

    def chart(self, t) -> np.ndarray:
        """`F(t)` modulo `p^K`. Times have shape `(..., rank)` (or `(...)` for rank
        one) and the points shape `(..., d)`.
        """
        t = np.asarray(t, dtype=object)
        if self.rank == 1 and (t.ndim == 0 or t.shape[-1] != 1):
            t = t[..., None]
        if t.ndim == 0 or t.shape[-1] != self.rank:
            raise ValueError(f"`t` must have {self.rank} coordinates.")
        out = np.empty(t.shape[:-1] + (self.dimension,), dtype=object)
        for index in np.ndindex(t.shape[:-1]):
            y = list(self.point)
            for (h, scale), s in reversed(list(zip(self._chart, t[index]))):
                y = self._power(h, scale * int(s), y)
            out[index] = y
        return out
