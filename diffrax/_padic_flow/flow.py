import math

import jax.numpy as jnp
import numpy as np

from .arithmetic import (
    factorial_valuation,
    make_residues,
    scan_horner,
    simplex,
    simplex_horner,
    valuation,
)
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
    to_sparse,
)
from .series import PowerSeries


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

    The table method also reduces the argument in space: with `x = a + p^r y` and
    `0 <= a < p^r`, every polynomial `P` becomes `sum_e P^{[e]}(a) p^{r |e|} y^e`, in
    which the terms with `r |e| >= K` vanish. The polynomials are tabulated for every
    centre `a`, so that their degree drops from about `2K` to `ceil(K / r) - 1`.

    For `d = 1` points are arrays of shape `(...)`; for `d > 1` they have shape
    `(..., d)`. Times and points are integers, read modulo `p^K`, and broadcast
    against each other. Beyond about 62 bits (`p^K >= 2^62`) the arithmetic runs on
    `Limbs`, and results come back as arrays of Python integers.

    **Arguments:**

    - `p`: a prime.
    - `precision`: `K`, the number of `p`-adic digits, unbounded. Up to `p^K < 2^31`
        a product is one machine multiplication, up to `2^62` three (see `Residues`),
        and beyond that `L (L + 1) / 2` for `L` limbs (see `Limbs`).
    - `field`: the vector field. For `d = 1`, the coefficients `[c_0, c_1, ...]` of
        `c_0 + c_1 x + ...`. In general, one dictionary `{exponents: coefficient}` per
        component, e.g. `[{(0, 1): 3}, {(1, 0): -3}]` for `(3y, -3x)`. Coefficients
        are integers or fractions with denominators prime to `p`. Components may
        also be analytic, given as `PowerSeries`: the flow is then still exact, since
        modulo `p^{K + g}` the series is a polynomial.
    - `table_digits`: `J`. Defaults to the largest `J` with `p^J <= 32`.
    - `space_digits`: `r`, the number of digits of the centres. Defaults to the
        largest `r` whose tables hold at most `table_size` numbers; `0` disables the
        reduction.
    - `table_size`: the budget for `space_digits`, in `int64` entries.
    - `reduction`: `"barrett"`, `"barrett32"`, `"remainder"`, `"double"` or
        `"limbs"`; chosen automatically by default, see `Residues`.
    - `storage`: `"dense"` stores every exponent up to the degree in each variable,
        `(deg + 1)^d` coefficients per polynomial; `"sparse"` stores only total degree
        at most `deg`, `C(deg + d, d)` of them, and evaluates them by Horner's rule
        nested over that simplex. The Lie polynomials and the reduced tables have
        that total degree, so nothing is lost. Defaults to `"sparse"` for `d > 1`.
    """

    def __init__(
        self,
        p: int,
        precision: int,
        field,
        *,
        table_digits: int | None = None,
        space_digits: int | None = None,
        table_size: int = 2**20,
        reduction: str | None = None,
        storage: str | None = None,
    ):
        residues = make_residues(p, precision, reduction)
        p, K = residues.p, residues.precision
        given = field
        field = parse_field(given, p, digits=K)
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
        widest = min(K, getattr(residues, "width", K))
        if not 1 <= table_digits <= widest:
            raise ValueError(f"`table_digits` must lie in 1, ..., {widest}.")
        if storage is None:
            storage = "sparse" if d > 1 else "dense"
        if storage not in ("dense", "sparse"):
            raise ValueError("`storage` must be 'dense' or 'sparse'.")

        self.residues = residues
        self.p = p
        self.precision = K
        self.modulus = residues.modulus
        self.dimension = d
        self.field_valuation = rate
        self.table_digits = table_digits
        self.storage = storage

        terms = _terms(p, K, rate)
        self.terms = terms
        guard = factorial_valuation(terms - 1, p)
        self.field = field = parse_field(given, p, digits=K + guard)
        layout = to_sparse if storage == "sparse" else to_dense

        def stored(polys, deg):
            return np.asarray(residues.asarray(layout(polys, d, deg)))

        taylor = self._divided_series(None)
        self.degree = max(degree(polys) for polys in taylor)
        series = np.stack([stored(polys, self.degree) for polys in taylor])
        self._taylor = series
        reduced = [
            {e: c for e, c in component.items() if c}
            for component in reduce(field, self.modulus)
        ]
        self._field_degree = degree(reduced)
        self._field = stored(reduced, self._field_degree)

        tail = _terms(p, K, rate + table_digits)
        tail_degree = max(degree(polys) for polys in taylor[:tail])
        self.tail_terms = tail
        tail_polys = np.stack([stored(polys, tail_degree) for polys in taylor[:tail]])

        size = p**table_digits
        axes = series.ndim - 1 - residues.extra
        s = jnp.arange(size, dtype=jnp.int64).reshape((size,) + (1,) * axes)
        s = residues.asarray(s)
        table = jnp.zeros((size,) + series.shape[1:], dtype=jnp.int64)
        for coefficients in series[::-1]:
            table = residues.add(residues.mul(table, s), jnp.asarray(coefficients))

        def count(deg):
            if storage == "sparse":
                return math.comb(deg + d, d)
            return (deg + 1) ** d

        def entries(r):
            head = min(self.degree, -(-K // r) - 1)
            low = min(tail_degree, -(-K // r) - 1)
            centres = p ** (r * d)
            words = getattr(residues, "limbs", 1)
            return centres * d * words * (size * count(head) + tail * count(low))

        largest = min(K - 1, getattr(residues, "width", K))
        if space_digits is None:
            space_digits = 0
            for r in range(1, largest + 1):
                if entries(r) > table_size:
                    break
                space_digits = r
        space_digits = int(space_digits)
        if not 0 <= space_digits <= largest:
            raise ValueError(f"`space_digits` must lie in 0, ..., {largest}.")
        self.space_digits = space_digits
        if storage == "sparse":
            table, self.table_degree = self._recenter_sparse(
                table, space_digits, self.degree
            )
            tail, self._tail_degree = self._recenter_sparse(
                jnp.asarray(tail_polys), space_digits, tail_degree
            )
        else:
            table = self._recenter(table, space_digits)
            self.table_degree = table.shape[table.ndim - residues.extra - 1] - 1
            tail = self._recenter(jnp.asarray(tail_polys), space_digits)
            self._tail_degree = tail.shape[tail.ndim - residues.extra - 1] - 1
        table = np.asarray(table)
        self._table = table.reshape((-1,) + table.shape[2:])
        self._tail = np.moveaxis(np.asarray(tail), 1, 0)

    def _evaluate(self, coefficients, x, deg: int, index=None):
        coefficients = jnp.asarray(coefficients)
        if self.storage == "sparse":
            return simplex_horner(self.residues, coefficients, x, deg, index)
        return evaluate(self.residues, coefficients, x, index)

    def _recenter(self, coefficients, r: int):
        """Dense polynomials of shape `(L, *rest, deg + 1, ..., deg + 1)` re-expanded
        around every centre `a` in `(Z/p^r)^d`: `y -> P(a + p^r y)`, of shape
        `(L, p^{rd}, *rest, deg' + 1, ..., deg' + 1)` with `deg' = ceil(K/r) - 1`.
        """
        res, p, K, d = self.residues, self.p, self.precision, self.dimension
        out = jnp.asarray(coefficients)
        if r == 0:
            return out[:, None]
        centres = res.asarray(jnp.arange(p**r, dtype=jnp.int64))
        values = out.ndim - res.extra
        deg = out.shape[values - 1] - 1
        new = min(deg, -(-K // r) - 1)
        powers = [jnp.broadcast_to(res.constant(1), centres.shape)]
        for _ in range(deg):
            powers.append(res.mul(powers[-1], centres))
        for axis in range(d):
            values = out.ndim - res.extra
            position = values - d + axis
            moved = jnp.moveaxis(out, position, values - 1)
            pieces = []
            for e in range(new + 1):
                total = res.zeros(res.shape(moved)[:-1] + (p**r,))
                for f in range(e, deg + 1):
                    weight = res.mul(powers[f - e], math.comb(f, e) % self.modulus)
                    term = jnp.expand_dims(jnp.take(moved, f, axis=values - 1), -1)
                    if res.extra:
                        term = jnp.moveaxis(term, -1, -2)
                    total = res.add(total, res.mul(term, weight))
                scale = pow(p, r * e, self.modulus) if r * e < K else 0
                pieces.append(res.mul(total, scale))
            shifted = jnp.stack(pieces, axis=values)
            out = jnp.moveaxis(shifted, (values - 1, values), (1 + axis, position + 1))
        lead = out.shape[0]
        centre_axes = out.shape[1 : 1 + d]
        rest = out.shape[1 + d :]
        return out.reshape((lead, math.prod(centre_axes)) + rest)

    def _recenter_sparse(self, coefficients, r: int, deg: int):
        """`_recenter` for polynomials stored over `simplex(d, deg)`, shape
        `(L, *rest, M)`, without ever forming dense arrays. The variables are shifted
        one at a time, `x_j -> a_j + p^r y_j`, which keeps the total degree, and the
        terms of total degree `|e| >= K / r` are dropped at the end. Returns the
        polynomials over `simplex(d, deg')`, shape `(L, p^{rd}, *rest, M')`, and
        `deg'`.
        """
        res, p, K, d = self.residues, self.p, self.precision, self.dimension
        out = jnp.asarray(coefficients)[:, None]
        if r == 0:
            return out, deg
        exponents = simplex(d, deg)
        position = {e: i for i, e in enumerate(exponents)}
        side = p**r
        centres = res.asarray(jnp.arange(side, dtype=jnp.int64))
        powers = [jnp.broadcast_to(res.constant(1), centres.shape)]
        for _ in range(deg):
            powers.append(res.mul(powers[-1], centres))
        tail = out.shape[out.ndim - res.extra :]
        monomial = out.ndim - res.extra - 1
        for j in range(d):
            lead, rest = out.shape[:2], out.shape[2:monomial]
            shape = lead + (side,) + rest + (len(exponents),)
            total = res.zeros(shape)
            for s in range(deg + 1):
                pairs = []
                for e in exponents:
                    f = e[:j] + (e[j] + s,) + e[j + 1 :]
                    if f in position:
                        scale = pow(p, r * e[j], self.modulus) if r * e[j] < K else 0
                        weight = math.comb(f[j], e[j]) * scale % self.modulus
                        pairs.append((position[f], position[e], weight))
                if not pairs:
                    continue
                source, target, weights = zip(*pairs)
                weights = res.asarray(np.array(weights, dtype=object))
                weights = res.mul(powers[s][:, None], weights[None, :])
                weights = weights.reshape(
                    (1, 1, side) + (1,) * len(rest) + (len(pairs),) + tail
                )
                taken = jnp.take(out, jnp.asarray(source), axis=monomial)
                product = res.mul(jnp.expand_dims(taken, 2), weights)
                index = (Ellipsis, jnp.asarray(target)) + (slice(None),) * res.extra
                total = res.add(total, res.zeros(shape).at[index].set(product))
            out = total.reshape((lead[0], lead[1] * side) + total.shape[3:])
        new = min(deg, -(-K // r) - 1)
        keep = jnp.asarray([position[e] for e in simplex(d, new)])
        return jnp.take(out, keep, axis=out.ndim - res.extra - 1), new

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

    def _flatten(self, *arrays, encoded: bool = False):
        """Broadcast times or points (given as `(times, is_point)` pairs) to a common
        batch shape, and flatten them to values of shape `(B,)` and `(B, d)`.
        """
        res, d = self.residues, self.dimension
        tail = (res.limbs,) if res.extra else ()
        values = []
        shapes = []
        for array, is_point in arrays:
            if not encoded:
                array = res.asarray(array)
            shape = res.shape(array)
            if is_point and d > 1:
                if len(shape) == 0 or shape[-1] != d:
                    raise ValueError(f"Points of Z_p^{d} must have shape (..., {d}).")
                shapes.append(shape[:-1])
            else:
                shapes.append(shape)
            values.append((array, is_point))
        batch = jnp.broadcast_shapes(*shapes)
        out = []
        for array, is_point in values:
            if is_point:
                if d == 1:
                    array = jnp.expand_dims(array, array.ndim - res.extra)
                array = jnp.broadcast_to(array, batch + (d,) + tail)
                out.append(array.reshape((-1, d) + tail))
            else:
                out.append(jnp.broadcast_to(array, batch + tail).reshape((-1,) + tail))
        return batch, out

    def _unflatten(self, batch, x):
        """Points of value shape `(B, d)` back to `batch`, as values."""
        tail = x.shape[2:]
        shape = batch if self.dimension == 1 else batch + (self.dimension,)
        return self.residues.decode(x.reshape(shape + tail))

    def _scalars(self, batch, t):
        """Times of value shape `(B,)` back to `batch`, as values."""
        return self.residues.decode(t.reshape(batch + t.shape[1:]))

    def _select(self, condition, a, b):
        if self.residues.extra:
            return np.where(np.asarray(condition), a, b)
        return jnp.where(condition, a, b)

    #
    # The flow
    #

    def _split(self, x):
        """Centre index and reduced coordinates: `x = a + p^r y`."""
        r = self.space_digits
        if r == 0:
            return jnp.zeros(x.shape[0], dtype=jnp.int64), x
        scale = self.p**r
        digits = self.residues.low(x, r)
        weights = jnp.asarray(
            [scale ** (self.dimension - 1 - j) for j in range(self.dimension)],
            dtype=jnp.int64,
        )
        index = (digits * jnp.broadcast_to(weights, digits.shape)).sum(axis=-1)
        return index, self.residues.shift(x, r)

    def _small_flow(self, sigma, x, digits: int):
        """`phi_sigma(x)` for `sigma` divisible by `p^digits`, `digits >= J`: only the
        Lie terms that survive such a small time are used.
        """
        res = self.residues
        rate = self.field_valuation + digits
        terms = min(self.tail_terms, _terms(self.p, self.precision, rate))
        centre, y = self._split(x)
        tail = self._tail[:, :terms]
        values = self._evaluate(tail, y, self._tail_degree, centre)
        values = jnp.moveaxis(values, 1, -1 - res.extra)
        return scan_horner(res, values, sigma[:, None])

    def _flow(self, t, x, method: str = "table"):
        res = self.residues
        if method == "taylor":
            values = self._evaluate(self._taylor[None], x, self.degree)
            values = jnp.moveaxis(values, 1, -1 - res.extra)
            return scan_horner(res, values, t[:, None])
        if method != "table":
            raise ValueError("`method` must be 'table' or 'taylor'.")
        low = res.low(t, self.table_digits)
        sigma = res.sub(t, res.asarray(low))
        centre, y = self._split(x)
        row = low * self.p ** (self.space_digits * self.dimension) + centre
        middle = self._evaluate(self._table, y, self.table_degree, row)
        return self._small_flow(sigma, middle, self.table_digits)

    def flow(self, t, x, method: str = "table"):
        """`phi_t(x)` modulo `p^K`."""
        batch, (t, x) = self._flatten((t, False), (x, True))
        return self._unflatten(batch, self._flow(t, x, method))

    def _apply(self, t, x, method: str = "table"):
        batch, (t, x) = self._flatten((t, False), (x, True), encoded=True)
        return self._unflatten(batch, self._flow(t, x, method))

    def vector_field(self, x):
        """`f(x)` modulo `p^K`."""
        batch, (x,) = self._flatten((x, True))
        return self._unflatten(batch, self._vector_field(x))

    def _vector_field(self, x):
        return self._evaluate(self._field[None], x, self._field_degree)

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
        times = res.asarray(jnp.repeat(jnp.arange(count, dtype=jnp.int64), size))
        points = jnp.tile(x, (count,) + (1,) * (x.ndim - 1))
        iterates = self._flow(times, points).reshape((count,) + x.shape)
        out = []
        for n in range(count):
            total = jnp.zeros_like(iterates[0])
            for j in range(n + 1):
                weight = (-1) ** (n - j) * math.comb(n, j) % self.modulus
                total = res.add(total, res.mul(iterates[j], weight))
            out.append(total)
        stacked = jnp.stack(out)
        shape = batch if self.dimension == 1 else batch + (self.dimension,)
        return res.decode(stacked.reshape((count,) + shape + stacked.shape[3:]))

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
        x_axis = res.take(x, axis)
        time = res.zeros((x.shape[0],))
        foot = z
        for j in range(math.ceil(math.log2(K)) + 2):
            value = res.take(foot, axis)
            slope = res.take(self._vector_field(foot), axis)
            step = res.divide(res.sub(value, x_axis), slope, level)
            time = res.truncate(res.add(time, step), K - level)
            small = 2**j - 1
            if small >= self.table_digits:
                foot = self._small_flow(res.neg(step), foot, small)
            else:
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
        return self._scalars(batch, time), self._unflatten(batch, foot)

    def orbit_time(self, x, z):
        """The time `t` with `phi_t(x) = z` (modulo `p^{K - k}`), and whether `z` lies
        on the orbit of `x` at all.
        """
        res = self.residues
        batch, (x, z) = self._flatten((x, True), (z, True))
        time, foot, level = self._rectify(x, z)
        inside = (res.valuation(res.sub(z, x)) >= level[:, None]).all(axis=1)
        on_orbit = inside & res.equal(foot, x).all(axis=1)
        time = res.where(level >= self.precision, jnp.zeros_like(time), time)
        return self._scalars(batch, time), on_orbit.reshape(batch)

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
        first = self._scalars(batch, res.truncate(time, jnp.clip(level - k, 0, None)))
        first = self._select(hit.reshape(batch), first, -1)
        return self._select(close.reshape(batch), 0, first)

    def return_time(self, x, level: int):
        """The first return time of `x` to `B(x, p^{-level})`, which is
        `p^{max(level - k, 0)}`.
        """
        k = self.orbit_level(x)
        return self.residues.p_power(jnp.clip(level - k, 0, None))

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
        k = int(self.orbit_level(x))
        period = self.p ** max(level - k, 0)
        times = jnp.arange(period, dtype=jnp.int64)
        points = self.flow(times, x)
        cells = jnp.asarray(np.asarray(points % size).astype(np.int64))
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
        scaled = [
            q * component
            if isinstance(component, PowerSeries)
            else {e: q * c for e, c in component.items()}
            for component in field
        ]
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
        digits = 2 if self.p == 2 else 1
        t0, t1 = res.asarray(t0), res.asarray(t1)
        elapsed = res.sub(t1, t0)
        try:
            concrete = np.asarray(res.low(elapsed, digits))
        except Exception:
            concrete = None
        if concrete is not None and concrete.any():
            raise ValueError(f"`t1 - t0` must be divisible by {self.step}.")
        s = res.shift(elapsed, digits)
        x = res.asarray(x)
        points = jnp.expand_dims(x, x.ndim - res.extra) if self.dimension == 1 else x
        batch = jnp.broadcast_shapes(res.shape(elapsed), res.shape(points)[:-1])
        tail = (res.limbs,) if res.extra else ()
        clock = jnp.expand_dims(jnp.broadcast_to(t0, batch + tail), len(batch))
        state = jnp.concatenate(
            [clock, jnp.broadcast_to(points, batch + (self.dimension,) + tail)],
            axis=len(batch),
        )
        s = jnp.broadcast_to(s, batch + tail)
        out = self.autonomous._apply(s, state, method=method)[..., 1:]
        return out[..., 0] if self.dimension == 1 else out
