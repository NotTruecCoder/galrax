"""Return sets and jump-ahead for polynomial maps of `Z_p^d`.

For `x_n = g^n(x)`, modulo `p^e` (`e = 2` for `p = 2`, else `1`) the orbit is a tail
of length `s` and a cycle of length `L`. In the coordinates `x = c + p^e u` of the
ball of `x_s`, `g^L` is affine modulo `p^e`, `u -> A u + b` with `A = Dg^L(x_s)`, so
for `A` invertible modulo `p` the map `G = g^N`, `N = L ord(A) p^e`, is the identity
modulo `p^e` and the time-one map of a flow `phi` (`PAdicFlow.from_time_one`). Then

    x_{s + iL + j + N q} = g^j(c + p^e phi_q(u_i)),   u_i = (x_{s + iL} - c) / p^e,

for every `q` in `Z_p`: the times of an event along the orbit are, for each
`r = iL + j < N`, an `EventSet` of the flow, and the classes `n = s + r + N q` glue
into a union of classes modulo `M p^k`, `M` the part of `N` prime to `p`.

This needs an orbit without attracting directions: `Dg^L` invertible modulo `p` on
the cycle. A random polynomial map often has a singular `Dg^L` modulo `p`; then the
orbit modulo `p^K` is read exactly instead, a tail and a cycle of at most
`EXACT_CAP` points. On a nilpotent `Dg^L` the cycle has `L` points, the attracting
cycle modulo `p^K`; on a mixed one it grows with `K`. The times of an event are then
classes modulo the cycle length, and a class holds no exact zero unless its point
meets the event modulo `p^K`.

For jump-ahead the flow is not needed: `q -> x_{s + r + N q}` is a power series whose
`q^m` coefficient has valuation `>= m (e - 1/(p - 1))`, so modulo `p^K` it is the
polynomial `sum_{j < J} C(q, j) Delta^j x_{s + r}` with `J (e - 1/(p - 1)) >= K`,
read off the first `J` samples. The flow is built only for return sets.
"""

import math
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from .arithmetic import factorial_valuation
from .equilibria import _determinant
from .events import EventSet, PAdicEvent
from .flow import PAdicFlow
from .normal_form import _matmul, _substitute
from .polynomial import (
    _derivative,
    evaluate_exact,
    parse_field,
    parse_polynomial,
    reduce,
)
from .roots import _merge, Ball


EXACT_CAP = 200_000  # points of an orbit read exactly modulo p^K


def _unit(i: int, d: int) -> tuple:
    return tuple(int(j == i) for j in range(d))


def _trim(poly: dict, top: int) -> dict:
    return {k: v for k, v in poly.items() if v and sum(k) <= top}


def _degree(polys) -> int:
    return max((sum(k) for poly in polys for k in poly), default=0)


def companion(coefficients) -> list[dict]:
    """The shift `(a_n, ..., a_{n+k-1}) -> (a_{n+1}, ..., a_{n+k})` of the recurrence
    `a_{n+k} = c_0 a_n + ... + c_{k-1} a_{n+k-1}`, as a map of `Z_p^k`.
    """
    k = len(coefficients)
    shift = [{_unit(i + 1, k): 1} for i in range(k - 1)]
    return shift + [{_unit(i, k): c for i, c in enumerate(coefficients) if c}]


def linear(matrix) -> list[dict]:
    """The map `x -> M x`."""
    d = len(matrix)
    return [{_unit(j, d): c for j, c in enumerate(row) if c} for row in matrix]


class Orbit:
    """The orbit `x_n = g^n(x)` of a point under a polynomial map `g` of `Z_p^d`,
    modulo `p^K`: `x_n` for any integer `n` at a cost independent of `n`, and the
    return set of an event.

    Without attracting directions (`Dg^L` invertible modulo `p` on the cycle of `x`
    modulo `p^e`) through the flow of `g^N`. With them the orbit modulo `p^K` is read
    exactly: then `cycle = step` is its period modulo `p^K`, there is no `flow`, the
    classes of `times` count `n >= tail` only, and a `ValueError` is raised when the
    orbit has more than `EXACT_CAP` points.

    **Arguments:**

    - `p`, `precision`: a prime and `K`.
    - `g`: the map, given like a vector field (`[c_0, c_1, ...]` for `d = 1`, else
        one `{exponents: coefficient}` per component), with `p`-adic integer
        coefficients.
    - `x`: the starting point.

    **Attributes:** `tail` (`s`), `cycle` (`L`), `step` (`N`), `flow` (the flow of
    `g^N` in the coordinates `u`).
    """

    def __init__(self, p: int, precision: int, g, x):
        p, K = int(p), int(precision)
        field = parse_field(g, p)
        d = len(field)
        e = 2 if p == 2 else 1
        m = p ** (K + e)
        g = reduce(field, m)
        x = [int(v) % m for v in np.atleast_1d(np.asarray(x, dtype=object))]
        if len(x) != d:
            raise ValueError(f"`x` must be a point of Z_p^{d}.")
        seen, y = {}, tuple(v % p**e for v in x)
        while y not in seen:
            seen[y] = len(seen)
            y = tuple(evaluate_exact(f, y, p**e) for f in g)
        s = seen[y]
        cycle = list(seen)[s:]
        L = len(cycle)
        identity = [[int(i == j) for j in range(d)] for i in range(d)]
        A = identity
        for point in cycle:
            jacobian = [
                [evaluate_exact(_derivative(f, j, p**e), point, p**e) for j in range(d)]
                for f in g
            ]
            A = _matmul(jacobian, A, p**e)
        if _determinant(A, p) == 0:
            self._read_exactly(p, K, field, g, x, e)
            return
        order, power = 1, A
        while power != identity:
            power, order = _matmul(power, A, p**e), order + 1
        N = L * order * p**e

        orbit = [x]
        for _ in range(s + N - 1):
            orbit.append([evaluate_exact(f, orbit[-1], m) for f in g])

        # In u, a degree-k coefficient is divisible by p^{e k}: higher ones vanish.
        top = (K + e - 1) // e
        zero = (0,) * d
        c = list(cycle[0])
        chart = [_trim({zero: c[i], _unit(i, d): p**e}, top) for i in range(d)]
        charts = [chart]
        for _ in range(L):
            cap = max(top, _degree(g))
            chart = [_trim(_substitute(f, chart, cap, m), top) for f in g]
            charts.append(chart)
        H = []
        for i, f in enumerate(charts.pop()):
            f = {**f, zero: (f.get(zero, 0) - c[i]) % m}
            H.append(_trim({k: v // p**e for k, v in f.items()}, top))

        def compose(a, b):
            return [_substitute(f, b, top, p**K) for f in a]

        G, n = [{_unit(i, d): 1} for i in range(d)], order * p**e
        while n:
            if n & 1:
                G = compose(H, G)
            H, n = compose(H, H), n >> 1

        self.p, self.precision, self.dimension = p, K, d
        self.tail, self.cycle, self.step = s, L, N
        self._field, self._start, self._G = field, x, G
        self._g, self._e, self._top, self._centre = g, e, top, c
        self._orbit, self._charts = orbit, charts
        self._u = np.array(
            [
                [(v - w) // p**e % p**K for v, w in zip(orbit[s + r], c)]
                for r in range(0, N, L)
            ],
            dtype=object,
        )

    def _read_exactly(self, p, K, field, g, x, e):
        """With attracting directions: the orbit modulo `p^K` up to its first repeat."""
        m, seen, orbit, y = p ** (K + e), {}, [], x
        while (key := tuple(v % p**K for v in y)) not in seen:
            if len(orbit) >= EXACT_CAP:
                raise ValueError(
                    f"Dg^L is not invertible modulo p on the cycle of `x` (attracting "
                    f"directions), and the orbit modulo p^K has more than {EXACT_CAP} "
                    "points: lower the precision."
                )
            seen[key] = len(orbit)
            orbit.append(y)
            y = [evaluate_exact(f, y, m) for f in g]
        s = seen[key]
        self.p, self.precision, self.dimension = p, K, len(x)
        self.tail, self.cycle = s, len(orbit) - s
        self.step = self.cycle
        self._field, self._start, self._G = field, x, None
        self._g, self._e, self._orbit = g, e, orbit

    def __repr__(self) -> str:
        return (
            f"Orbit(p={self.p}, precision={self.precision}, dimension="
            f"{self.dimension}, tail={self.tail}, cycle={self.cycle}, step={self.step})"
        )

    @cached_property
    def flow(self) -> PAdicFlow:
        """The flow of `g^N` in the coordinates `u`, built on first use."""
        if self._G is None:
            raise ValueError("No flow: the orbit has attracting directions.")
        digits = min(self.precision, max(1, int(math.log(32) / math.log(self.p))))
        return PAdicFlow.from_time_one(
            self.p, self.precision, self._G, table_digits=digits
        )

    @cached_property
    def _tables(self) -> np.ndarray:
        """For every `r < N`, the polynomial in `q` equal to `x_{s + r + N q}` modulo
        `p^K`: integers of shape `(N, J, d)`, constant term first.
        """
        p, K, d, e = self.p, self.precision, self.dimension, self._e
        s, N = self.tail, self.step
        J = -(-K * (p - 1) // (e * (p - 1) - 1))
        work = p ** (K + factorial_valuation(J - 1, p))
        g = reduce(self._field, work)
        y = [v % work for v in self._start]
        samples = []
        for n in range(s + N * J):
            if n >= s:
                samples.append(y)
            y = [evaluate_exact(f, y, work) for f in g]
        m = p**K
        out = np.zeros((N, J, d), dtype=object)
        for r in range(N):
            differences = samples[r::N]
            falling = [1]  # q (q - 1) ... (q - j + 1), constant term first
            for j in range(J):
                v = factorial_valuation(j, p)
                unit = pow(math.factorial(j) // p**v, -1, m)
                for i in range(d):
                    if differences[0][i] % p**v:
                        raise RuntimeError(f"Delta^{j} is not divisible by {j}!.")
                    b = differences[0][i] // p**v * unit
                    for k, c in enumerate(falling):
                        out[r, k, i] = (out[r, k, i] + b * c) % m
                falling = [0] + falling
                for k in range(j + 1):
                    falling[k] = (falling[k] - j * falling[k + 1]) % m
                differences = [
                    [b - a for a, b in zip(u, w)]
                    for u, w in zip(differences, differences[1:])
                ]
        return out

    def _points(self, rows):
        u = self._u[rows]
        return u[:, 0] if self.dimension == 1 else u

    def __call__(self, n):
        """`x_n` modulo `p^K` for integers `n` of any shape; negative `n` when the
        tail is empty (the inverse on the orbit closure). Large arrays run as
        `int64` JAX arrays while `p^{2K} < 2^62`.
        """
        d, s, N, m = self.dimension, self.tail, self.step, self.p**self.precision
        shape = np.shape(n)
        flat = [int(v) for v in np.asarray(n, dtype=object).reshape(-1)]
        if s and min(flat, default=0) < 0:
            raise ValueError("Negative iterates need a point periodic modulo p.")
        out = np.empty((len(flat), d), dtype=object)
        if self._G is None:
            for k, v in enumerate(flat):
                i = v if 0 <= v < s else s + (v - s) % N
                out[k] = [w % m for w in self._orbit[i]]
            out = out.reshape(shape + ((d,) if d > 1 else ()))
            return out[()] if out.ndim == 0 else out
        early = np.array([0 <= v < s for v in flat], dtype=bool)
        for k in np.flatnonzero(early):
            out[k] = [v % m for v in self._orbit[flat[k]]]
        late = np.flatnonzero(~early)
        if len(late):
            k = np.array([flat[i] - s for i in late], dtype=object)
            q, r = k // N % m, (k % N).astype(np.int64)
            tables = self._tables
            fast = len(late) > 64 and m < 2**31 and jax.config.jax_enable_x64
            if fast:
                tables, q = (
                    jnp.asarray(tables.astype(np.int64)),
                    jnp.asarray(q.astype(np.int64)),
                )
                r = jnp.asarray(r)
            total = (jnp if fast else np).zeros((len(late), d), dtype=q.dtype)
            for j in reversed(range(tables.shape[1])):
                total = (total * q[:, None] + tables[r, j]) % m
            out[late] = np.asarray(total).astype(object)
        out = out.reshape(shape + ((d,) if d > 1 else ()))
        return out[()] if out.ndim == 0 else out

    def times(self, event: PAdicEvent) -> EventSet:
        """All `n >= 0` with `v_p(h(x_n)) >= depth` for every condition `h` of the
        event, as an `EventSet` of classes `n = c mod M p^k` valid from `n = tail`
        on, with the earlier hits in `tail`. For a single condition, `zero_bound`
        adds Strassmann's bound over every `r < N` to the hits in the tail: an upper
        bound on the number of `n >= 0` with `h(x_n) = 0` exactly; `classes` splits
        it over the classes `n = s + r mod N`.
        """
        p, K, d, e = self.p, self.precision, self.dimension, self._e
        s, L, N = self.tail, self.cycle, self.step
        depth = K if event.depth is None else event.depth
        if not 1 <= depth <= K:
            raise ValueError(f"`depth` must lie in 1, ..., {K}.")
        m = p ** (K + e)
        conditions = reduce([parse_polynomial(h, d, p) for h in event.condition], m)

        def hit(n, digits):
            return all(
                evaluate_exact(h, self._orbit[n], p**digits) == 0 for h in conditions
            )

        tail = tuple(n for n in range(s) if hit(n, depth))
        a = 0
        while N % p ** (a + 1) == 0:
            a += 1
        M = N // p**a
        bound, balls, classes = len(tail), [], []
        if self._G is None:
            # x_n = x_{s + r} mod p^K for n = s + r mod N: exact zeros need h = 0 there.
            for r in range(N):
                if hit(s + r, depth):
                    found = None if hit(s + r, K) else 0
                    classes.append((s + r, N, found))
                    bound = None if bound is None or found is None else bound + found
                    balls.append(Ball(((s + r) % (M * p**a),), a))
            balls = sorted(_merge(balls, p, 1, M), key=lambda b: b.center)
            return EventSet(balls, p, depth, bound, M, s or None, tail, tuple(classes))
        # x_{s + r + N q} = x_{s + r} mod p^{2e}, so most r are ruled out at once.
        cut = min(depth, 2 * e)
        for j in range(L):
            rs = [r for r in range(j, N, L) if hit(s + r, cut)]
            if not rs:
                continue
            cap = max(self._top, _degree(conditions))
            pulled = [
                {
                    k: v % p**K
                    for k, v in _substitute(h, self._charts[j], cap, m).items()
                    if sum(k) <= self._top and v % p**K
                }
                for h in conditions
            ]
            sets = self.flow.event_times(
                PAdicEvent(pulled, depth), self._points([r // L for r in rs])
            )
            for r, found in zip(rs, sets):
                classes.append((s + r, N, found.zero_bound))
                if bound is not None:
                    bound = (
                        None if found.zero_bound is None else bound + found.zero_bound
                    )
                for b in found.balls:
                    level = a + b.level
                    centre = (s + r + N * b.center[0]) % (M * p**level)
                    balls.append(Ball((centre,), level))
        balls = sorted(_merge(balls, p, 1, M), key=lambda b: b.center)
        return EventSet(balls, p, depth, bound, M, s or None, tail, tuple(classes))

    def index(self, y) -> int | None:
        """The smallest `n >= 0` with `x_n = y` modulo `p^K`, or `None`."""
        d = self.dimension
        y = [int(v) for v in np.atleast_1d(np.asarray(y, dtype=object))]
        conditions = [{_unit(i, d): 1, (0,) * d: -v} for i, v in enumerate(y)]
        return self.times(PAdicEvent(conditions)).first()


def recurrence_times(
    p: int, precision: int, coefficients, initial, value=0, depth: int | None = None
) -> EventSet:
    """The `n >= 0` with `a_n = value` to `depth` digits, for the linear recurrence
    `a_{n+k} = c_0 a_n + ... + c_{k-1} a_{n+k-1}` started at `initial`, through the
    orbit of its companion matrix.
    """
    k = len(coefficients)
    event = PAdicEvent({_unit(0, k): 1, (0,) * k: -value}, depth)
    return Orbit(p, precision, companion(coefficients), initial).times(event)
