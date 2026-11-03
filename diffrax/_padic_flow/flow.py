import math

import jax.numpy as jnp
import numpy as np

from .arithmetic import factorial_valuation, Residues, scan_horner, valuation
from .equilibria import equilibria, Equilibrium, jacobian, Linearization
from .events import event_times, PAdicEvent
from .polynomial import (
    degree,
    evaluate,
    field_valuation,
    lie_series,
    parse_field,
    reduce,
    to_dense,
)


def _terms(p: int, precision: int, rate: int) -> int:
    """The smallest `N` such that `n rate - (n - 1) / (p - 1) >= precision` for all
    `n >= N`: past it, `t^n D^n x / n!` vanishes modulo `p^precision` whenever
    `v_p(t^n D^n x) >= n rate`.
    """
    n = 1
    while n * rate - (n - 1) // (p - 1) < precision:
        n += 1
    return n


class PAdicFlow:
    """The flow of a polynomial differential equation on the `p`-adic integers,

    `dx/dt = f(x)`,  `x` in `Z_p^d`,  `t` in `Z_p`,

    computed exactly modulo `p^K` for every initial point and every time at once.

    The coefficients of `f` must be divisible by `p` (by `4` when `p = 2`). Then
    `D^n x` is divisible by `p^n` for the derivation `D = f . grad`, the Lie series
    `phi_t(x) = sum_n t^n D^n x / n!` converges for all `t` and `x`, and modulo `p^K`
    it is a polynomial in `(t, x)` with integer coefficients. A field that is only
    integral can be rescaled: the flow of `p f` at time `s` is the flow of `f` at time
    `p s`.

    The flow is evaluated in one of two ways, which agree exactly:

    - `"taylor"`: the polynomial `sum_{n < N} t^n E_n(x)`, with `E_n = D^n x / n!`.
    - `"table"`: write `t = t_lo + sigma` with `0 <= t_lo < p^J` and `p^J` dividing
        `sigma`. Then `phi_t = phi_sigma o phi_{t_lo}`. The `p^J` polynomials
        `phi_{t_lo}` are tabulated, and `phi_sigma` needs only a few Lie terms because
        `sigma` is `p`-adically small. This costs two polynomial evaluations, in the
        spirit of table-driven `exp`.

    For `d = 1` points are arrays of shape `(...)`; for `d > 1` they have shape
    `(..., d)`. Times and points are integers, read modulo `p^K`, and broadcast
    against each other.

    **Arguments:**

    - `p`: a prime.
    - `precision`: `K`, the number of `p`-adic digits; `p^K` must be below `2^62`
        (see `Residues`: beyond `2^31`, products use two half-length digits).
    - `field`: the vector field. For `d = 1`, the coefficients `[c_0, c_1, ...]` of
        `c_0 + c_1 x + ...`. In general, one dictionary `{exponents: coefficient}` per
        component, e.g. `[{(0, 1): 3}, {(1, 0): -3}]` for `(3y, -3x)`. Coefficients
        are integers or fractions with denominators prime to `p`.
    - `table_digits`: `J`. Defaults to the largest `J` with `p^J <= 32`.
    - `reduction`: `"barrett"`, `"remainder"` or `"double"`; chosen automatically by
        default, see `Residues`.
    """

    def __init__(
        self,
        p: int,
        precision: int,
        field,
        *,
        table_digits: int | None = None,
        reduction: str | None = None,
    ):
        residues = Residues(p, precision, reduction)
        p, K = residues.p, residues.precision
        field = parse_field(field, p)
        d = len(field)
        rate = field_valuation(field, p, cap=K)
        needed = 2 if p == 2 else 1
        if rate < needed:
            raise ValueError(
                f"The coefficients of the vector field must be divisible by "
                f"{p**needed}, so that the flow exists for all times in Z_{p}. "
                f"Multiply the field by {p**needed} and divide time by {p**needed}."
            )
        if table_digits is None:
            table_digits = max(1, int(math.floor(math.log(32) / math.log(p))))
        table_digits = int(table_digits)
        if not 1 <= table_digits <= K:
            raise ValueError(f"`table_digits` must lie in 1, ..., {K}.")

        self.residues = residues
        self.p = p
        self.precision = K
        self.modulus = residues.modulus
        self.dimension = d
        self.field = field
        self.field_valuation = rate
        self.table_digits = table_digits

        terms = _terms(p, K, rate)
        self.terms = terms
        taylor = self._divided_series(None)
        self.degree = max(degree(polys) for polys in taylor)
        self._taylor = np.stack([to_dense(polys, d, self.degree) for polys in taylor])
        self._field = to_dense(reduce(field, self.modulus), d, degree(field))

        tail = _terms(p, K, rate + table_digits)
        tail_degree = max(degree(polys) for polys in taylor[:tail])
        self.tail_terms = tail
        self._tail = self._taylor[
            (slice(0, tail), slice(None)) + (slice(0, tail_degree + 1),) * d
        ]

        size = p**table_digits
        s = jnp.arange(size, dtype=jnp.int64).reshape((size,) + (1,) * (d + 1))
        table = jnp.zeros((size,) + self._taylor.shape[1:], dtype=jnp.int64)
        for coefficients in self._taylor[::-1]:
            table = residues.add(residues.mul(table, s), jnp.asarray(coefficients))
        self._table = np.asarray(table)

    def _divided_series(self, start):
        """`[g, Dg, D^2 g / 2!, ...]` modulo `p^K`, `self.terms` entries, for the
        polynomials `g = start` (default: the coordinates). Each `D^n g` is computed
        exactly modulo `p^{K + v_p((N-1)!)}`, so that the division by `n!` is exact.
        """
        p, K = self.p, self.precision
        guard = factorial_valuation(self.terms - 1, p)
        big = p ** (K + guard)
        if start is not None:
            start = reduce(start, big)
        series = lie_series(reduce(self.field, big), self.terms, big, start)
        out = []
        for n, polys in enumerate(series):
            factorial = math.factorial(n)
            v = valuation(factorial, p, cap=K + guard)
            inverse = pow(factorial // p**v, -1, self.modulus)
            divided = []
            for poly in polys:
                terms = {e: c // p**v * inverse % self.modulus for e, c in poly.items()}
                divided.append({e: c for e, c in terms.items() if c})
            out.append(divided)
        return out

    def __repr__(self) -> str:
        return (
            f"PAdicFlow(p={self.p}, precision={self.precision}, "
            f"dimension={self.dimension})"
        )

    #
    # Points and times
    #

    def _flatten(self, *arrays):
        """Broadcast times or points (given as `(times, is_point)` pairs) to a common
        batch shape, and flatten them to `(B,)` and `(B, d)`.
        """
        values = []
        shapes = []
        for array, is_point in arrays:
            array = self.residues.asarray(array)
            if is_point and self.dimension > 1:
                if array.ndim == 0 or array.shape[-1] != self.dimension:
                    raise ValueError(
                        f"Points of Z_p^{self.dimension} must have shape "
                        f"(..., {self.dimension})."
                    )
                shapes.append(array.shape[:-1])
            else:
                shapes.append(array.shape)
            values.append((array, is_point))
        batch = jnp.broadcast_shapes(*shapes)
        out = []
        for array, is_point in values:
            if is_point:
                if self.dimension == 1:
                    array = array[..., None]
                array = jnp.broadcast_to(array, batch + (self.dimension,))
                out.append(array.reshape(-1, self.dimension))
            else:
                out.append(jnp.broadcast_to(array, batch).reshape(-1))
        return batch, out

    def _unflatten(self, batch, x):
        if self.dimension == 1:
            return x.reshape(batch)
        return x.reshape(batch + (self.dimension,))

    #
    # The flow
    #

    def _flow(self, t, x, method: str = "table"):
        res = self.residues
        if method == "taylor":
            values = evaluate(res, self._taylor[None], x)
            return scan_horner(res, jnp.moveaxis(values, 1, -1), t[:, None])
        if method != "table":
            raise ValueError("`method` must be 'table' or 'taylor'.")
        low = jnp.remainder(t, self.p**self.table_digits)
        sigma = res.sub(t, low)
        y = evaluate(res, jnp.asarray(self._table)[low], x)
        values = evaluate(res, self._tail[None], y)
        return scan_horner(res, jnp.moveaxis(values, 1, -1), sigma[:, None])

    def flow(self, t, x, method: str = "table"):
        """`phi_t(x)` modulo `p^K`."""
        batch, (t, x) = self._flatten((t, False), (x, True))
        return self._unflatten(batch, self._flow(t, x, method))

    def vector_field(self, x):
        """`f(x)` modulo `p^K`."""
        batch, (x,) = self._flatten((x, True))
        return self._unflatten(batch, self._vector_field(x))

    def _vector_field(self, x):
        return evaluate(self.residues, self._field[None], x)

    def mahler_coefficients(self, x, count: int | None = None):
        """`Delta^n(x) = sum_j (-1)^{n-j} C(n, j) phi_j(x)` for `n < count` (default
        `K`), of shape `(count,) + x.shape`. `Delta^n(x)` is divisible by `p^n`, and
        `phi_t(x) = sum_n C(t, n) Delta^n(x)` for every `t` in `Z_p`: the flow is the
        Mahler interpolation of the iterates of `phi_1`.
        """
        count = self.precision if count is None else int(count)
        res = self.residues
        batch, (x,) = self._flatten((x, True))
        size = x.shape[0]
        times = jnp.repeat(jnp.arange(count, dtype=jnp.int64), size)
        iterates = self._flow(times, jnp.tile(x, (count, 1))).reshape(count, size, -1)
        out = []
        for n in range(count):
            total = jnp.zeros_like(iterates[0])
            for j in range(n + 1):
                weight = (-1) ** (n - j) * math.comb(n, j) % self.modulus
                total = res.add(total, res.mul(iterates[j], weight))
            out.append(self._unflatten(batch, total))
        return jnp.stack(out)

    #
    # Orbits
    #

    def _levels(self, x):
        values = self.residues.valuation(self._vector_field(x))
        return values.min(axis=-1), values.argmin(axis=-1)

    def orbit_level(self, x):
        """`k = v_p(f(x))`, so that the orbit of `x` has radius `p^{-k}`; `K` for an
        equilibrium (to the working precision).

        The orbit `{phi_t(x) : t in Z_p}` of a point with `f(x) != 0` is a curve
        through `x`, and `t -> phi_t(x)` multiplies distances by exactly `|f(x)|_p`.
        For `d = 1` the orbit is the whole ball `B(x, |f(x)|_p)`.
        """
        batch, (x,) = self._flatten((x, True))
        return self._levels(x)[0].reshape(batch)

    def _rectify(self, x, z):
        res = self.residues
        K = self.precision
        level, axis = self._levels(x)
        powers = jnp.asarray(res.powers)
        x_axis = jnp.take_along_axis(x, axis[:, None], axis=1)[:, 0]
        time = jnp.zeros(x.shape[0], dtype=jnp.int64)
        foot = z
        for _ in range(math.ceil(math.log2(K)) + 2):
            value = jnp.take_along_axis(foot, axis[:, None], axis=1)[:, 0]
            slope = jnp.take_along_axis(
                self._vector_field(foot), axis[:, None], axis=1
            )[:, 0]
            step = res.divide(res.sub(value, x_axis), slope, level)
            time = jnp.remainder(res.add(time, step), powers[K - level])
            foot = self._flow(res.neg(time), z)
        return time, foot, level

    def rectify(self, x, z):
        """Flow-box coordinates of `z` around `x`: the time `s` and the foot `w` with
        `z = phi_s(w)`, where `w` lies on the transversal through `x` (the coordinate
        `i` with `|f_i(x)| = |f(x)|` held fixed at `x_i`).

        For `z` in the ball `B(x, |f(x)|_p)` both exist and are unique, the map
        `(s, w) -> phi_s(w)` is an isometry for the distance
        `max(|f(x)| |s - s'|, |w - w'|)`, and the flow is `s -> s + t`. The time is
        exact modulo `p^{K - k}` with `k = orbit_level(x)`; it is computed by Newton's
        method, which doubles the number of correct digits at each step. For `d = 1`
        the foot is `x` itself and `s` is the rectifying coordinate:
        `phi_t(z) = phi_{s + t}(x)`.

        Returns `(s, w)`; values for `z` outside the ball are meaningless.
        """
        batch, (x, z) = self._flatten((x, True), (z, True))
        time, foot, _ = self._rectify(x, z)
        return time.reshape(batch), self._unflatten(batch, foot)

    def orbit_time(self, x, z):
        """The time `t` with `phi_t(x) = z` (modulo `p^{K - k}`), and whether `z` lies
        on the orbit of `x` at all.
        """
        res = self.residues
        batch, (x, z) = self._flatten((x, True), (z, True))
        time, foot, level = self._rectify(x, z)
        inside = (res.valuation(res.sub(z, x)) >= level[:, None]).all(axis=1)
        on_orbit = inside & (foot == x).all(axis=1)
        time = jnp.where(level >= self.precision, 0, time)
        return time.reshape(batch), on_orbit.reshape(batch)

    def first_hitting_time(self, x, y, level: int):
        """The first integer time `n >= 0` with `phi_n(x)` in the ball
        `B(y, p^{-level})`, or `-1` if the orbit never enters it.

        In flow-box coordinates the ball is a product, so the hitting times form the
        coset `s_y + p^{level - k} Z_p` and the first one is `s_y mod p^{level - k}`.
        """
        res = self.residues
        level = int(level)
        if not 1 <= level <= self.precision:
            raise ValueError(f"`level` must lie in 1, ..., {self.precision}.")
        batch, (x, y) = self._flatten((x, True), (y, True))
        time, foot, k = self._rectify(x, y)
        close = (res.valuation(res.sub(y, x)) >= level).all(axis=1)
        inside = (res.valuation(res.sub(y, x)) >= k[:, None]).all(axis=1)
        hit = inside & (k < level) & (res.valuation(res.sub(foot, x)) >= level).all(1)
        period = jnp.asarray(res.powers)[jnp.clip(level - k, 0, self.precision)]
        first = jnp.where(hit, jnp.remainder(time, period), -1)
        return jnp.where(close, 0, first).reshape(batch)

    def return_time(self, x, level: int):
        """The first return time of `x` to `B(x, p^{-level})`, which is
        `p^{max(level - k, 0)}`.
        """
        k = self.orbit_level(x)
        return jnp.asarray(self.residues.powers)[jnp.clip(level - k, 0, None)]

    def orbit_average(self, values, x, level: int):
        """The time average `lim (1/N) sum_{n < N} F(phi_n(x))` of a function `F` that
        is constant on balls of radius `p^{-level}`, given by its `values` on
        `(Z/p^level)^d` (shape `(p^level,) * d`).

        The orbit visits `p^{level - k}` balls, each equally often, so the average is
        exact and costs `p^{level - k}` evaluations instead of a long sum.
        """
        level = int(level)
        size = self.p**level
        values = jnp.asarray(values)
        if values.shape != (size,) * self.dimension:
            raise ValueError(f"`values` must have shape {(size,) * self.dimension}.")
        x = self.residues.asarray(x)
        k = int(self.orbit_level(x))
        period = self.p ** max(level - k, 0)
        times = jnp.arange(period, dtype=jnp.int64)
        points = self.flow(times, x)
        cells = jnp.remainder(points, size)
        if self.dimension == 1:
            cells = cells[:, None]
        return values[tuple(cells.T)].mean()

    #
    # Events and equilibria
    #

    def event_times(self, event: PAdicEvent, x):
        """The times `t` in `Z_p` at which `event` holds along the orbit of `x`: one
        `EventSet` per point (a single one for a single point).

        `G(t) = g(phi_t(x)) = sum_n t^n (D^n g)(x) / n!` is a power series in `t` with
        coefficients tending to zero, known exactly modulo `p^K`. Its vanishing set
        is found by refining balls in time, digit by digit, deciding each ball from
        the Taylor expansion of `G` around its centre; Strassmann's theorem bounds
        the number of exact zeros.
        """
        sets, batch = event_times(self, event, x)
        return sets[0] if batch == () else sets

    def first_event(self, event: PAdicEvent, x):
        """The first integer time `n >= 0` at which `event` holds along the orbit of
        a single point `x`, and `phi_n(x)`; `(None, None)` if it never does.
        """
        times = self.event_times(event, x)
        if isinstance(times, list):
            raise ValueError("`first_event` takes a single point.")
        n = times.first()
        if n is None:
            return None, None
        return n, self.flow(n, x)

    def equilibria(self, max_balls: int = 100_000) -> list[Equilibrium]:
        """All zeros of the vector field in `Z_p^d`, to the working precision, found
        by refining balls in space digit by digit. Simple zeros are certified by
        Hensel's lemma.
        """
        return equilibria(self, max_balls)

    def jacobian(self, x) -> np.ndarray:
        """`Df(x)` modulo `p^K` at a single point."""
        point = np.atleast_1d(np.asarray(x))
        if point.shape != (self.dimension,):
            raise ValueError(f"`x` must be a single point of Z_p^{self.dimension}.")
        return np.array(jacobian(self, point.tolist()), dtype=np.int64)

    def linearization(self, equilibrium, level: int | None = None) -> Linearization:
        """Linearising coordinates around a simple equilibrium (`d = 1`); see
        `Linearization`.
        """
        return Linearization(self, equilibrium, level)


class TimeDependentFlow:
    """The flow of a non-autonomous polynomial equation

    `dx/dt = f(t, x)`,  `x` in `Z_p^d`,

    from time `t0` to time `t1`, for `t1 - t0` divisible by `q = p` (`q = 4` when
    `p = 2`), exactly modulo `p^K`.

    Unlike the autonomous case, time cannot range over all of `Z_p`: already
    `dx/dt = p t^{p^2 - 1}` has the solution `x(t) = x(0) + t^{p^2} / p`, which leaves
    `Z_p`. On `t0 + q Z_p` the problem is regular for any `f` with `p`-adic integer
    coefficients: with `t = t0 + q s`, the pair `(tau, x)` obeys the autonomous
    equation `dtau/ds = q`, `dx/ds = q f(tau, x)`, whose coefficients are divisible by
    `q`. That equation is solved by a `PAdicFlow`, available as `autonomous`, so
    events and orbits can be studied there too.

    **Arguments:**

    - `p`, `precision`: as for `PAdicFlow`.
    - `field`: one dictionary `{(i, j_1, ..., j_d): c}` per component, standing for
        `c t^i x_1^{j_1} ... x_d^{j_d}`. Coefficients are `p`-adic integers.
    - `**kwargs`: passed to `PAdicFlow`.
    """

    def __init__(self, p: int, precision: int, field, **kwargs):
        p = int(p)
        field = parse_field(field, p, extra_variables=1)
        d = len(field)
        q = 4 if p == 2 else p
        clock = {(0,) * (d + 1): q}
        scaled = [{e: q * c for e, c in component.items()} for component in field]
        self.p = p
        self.dimension = d
        self.step = q
        self.field = field
        self.autonomous = PAdicFlow(p, precision, [clock] + scaled, **kwargs)
        self.precision = self.autonomous.precision
        self.modulus = self.autonomous.modulus

    def __repr__(self) -> str:
        return (
            f"TimeDependentFlow(p={self.p}, precision={self.precision}, "
            f"dimension={self.dimension})"
        )

    def flow(self, t0, t1, x, method: str = "table"):
        """The solution at time `t1` of the problem started from `x` at time `t0`.
        `t1 - t0` must be divisible by `q`.
        """
        res = self.autonomous.residues
        t0, t1 = res.asarray(t0), res.asarray(t1)
        elapsed = res.sub(t1, t0)
        try:
            concrete = np.asarray(elapsed)
        except Exception:
            concrete = None
        if concrete is not None and (concrete % self.step).any():
            raise ValueError(f"`t1 - t0` must be divisible by {self.step}.")
        s = jnp.floor_divide(elapsed, self.step)
        x = res.asarray(x)
        points = x[..., None] if self.dimension == 1 else x
        batch = jnp.broadcast_shapes(elapsed.shape, points.shape[:-1])
        state = jnp.concatenate(
            [
                jnp.broadcast_to(t0, batch)[..., None],
                jnp.broadcast_to(points, batch + (self.dimension,)),
            ],
            axis=-1,
        )
        s = jnp.broadcast_to(s, batch)
        out = self.autonomous.flow(s, state, method=method)[..., 1:]
        return out[..., 0] if self.dimension == 1 else out
