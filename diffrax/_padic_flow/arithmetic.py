import functools
import math

import jax
import jax.numpy as jnp
import numpy as np


def is_prime(p: int) -> bool:
    return p >= 2 and all(p % k for k in range(2, math.isqrt(p) + 1))


def valuation(x: int, p: int, cap: int) -> int:
    """`v_p(x)` of a Python integer, capped at `cap` (also for `x = 0`)."""
    if x == 0:
        return cap
    v = 0
    while x % p == 0 and v < cap:
        x //= p
        v += 1
    return v


def factorial_valuation(n: int, p: int) -> int:
    """`v_p(n!)`, by Legendre's formula."""
    total, power = 0, p
    while power <= n:
        total += n // power
        power *= p
    return total


def single_word_precision(p: int) -> int:
    """The largest `K` with `p^K < 2^31`: products of residues fit in one `int64`."""
    k = 0
    while p ** (k + 1) < 2**31:
        k += 1
    return k


def double_word_precision(p: int) -> int:
    """The largest `K` for which residues fit in `int64` and their products can be
    formed from two half-length digits: `p^{2 ceil(K/2)} < 2^63` and `2 p^K < 2^63`.
    """
    k = 0
    while p ** (2 * math.ceil((k + 1) / 2)) < 2**63 and 2 * p ** (k + 1) < 2**63:
        k += 1
    return k


class Residues:
    """Exact arithmetic in `Z/p^K`, the `p`-adic integers to `K` digits, on `int64`
    JAX arrays.

    Elements are integers in `[0, p^K)`. Up to `p^K < 2^31` a product of two of them
    fits in `int64`, and is reduced either with the integer remainder or with Barrett
    reduction, which replaces the division by a multiplication with a precomputed
    `float64` reciprocal and a correction of at most one modulus.

    Beyond that, up to about twice as many digits (`p^K < 2^62`), elements still fit
    in `int64` but products do not. They are then formed from the half-length digits
    `a = a_0 + B a_1` with `B = p^{ceil(K/2)}`: since `B^2 = 0` modulo `p^K`,
    `ab = a_0 b_0 + B (a_0 b_1 + a_1 b_0)`, three products that each fit. This is the
    `"double"` reduction, chosen automatically.

    **Arguments:**

    - `p`: a prime.
    - `precision`: `K`, the number of digits.
    - `reduction`: `"barrett"`, `"barrett32"`, `"remainder"` or `"double"`. Defaults to
        `"remainder"` when products fit in `int64`, and to `"double"` otherwise. Which
        single-word reduction is fastest depends on the platform: in our measurements
        the remainder won on GPU and on a Linux CPU, Barrett on a Windows CPU build.
        `"barrett32"` estimates the quotient in `float32` twice, avoiding `float64`;
        it did not help, because the cost is in the 64-bit integer products.

    Arrays of residues have the shape of the values they hold; `Limbs` handles more
    digits than fit in one `int64`.
    """

    extra = 0

    def __init__(self, p: int, precision: int, reduction: str | None = None):
        p, precision = int(p), int(precision)
        if not is_prime(p):
            raise ValueError("`p` must be a prime.")
        if precision < 1:
            raise ValueError("`precision` must be at least 1.")
        single, double = single_word_precision(p), double_word_precision(p)
        if precision > double:
            raise ValueError(
                f"For p = {p} at most {double} digits fit in int64; "
                f"got precision = {precision}."
            )
        if reduction is None:
            reduction = "double" if precision > single else "remainder"
        if reduction not in ("barrett", "barrett32", "remainder", "double"):
            raise ValueError(
                "`reduction` must be 'barrett', 'barrett32', 'remainder' or 'double'."
            )
        if reduction != "double" and precision > single:
            raise ValueError(
                f"Products of {precision}-digit residues do not fit in int64 for "
                f"p = {p} (at most {single} digits); use reduction='double'."
            )
        if jnp.asarray(0, dtype=jnp.int64).dtype != jnp.int64:
            raise RuntimeError(
                "p-adic arithmetic needs 64-bit integers: enable them with "
                "`jax.config.update('jax_enable_x64', True)`."
            )
        self.p = p
        self.precision = precision
        self.modulus = p**precision
        self.reduction = reduction
        self.powers = np.array([p**k for k in range(precision + 1)], dtype=np.int64)
        self._reciprocal = 1.0 / self.modulus
        self._reciprocal32 = np.float32(1.0 / self.modulus)
        self._half = p ** math.ceil(precision / 2)
        self._rest = p ** (precision - math.ceil(precision / 2))
        self._inverses = np.array(
            [pow(a, -1, p) if a % p else 0 for a in range(p)], dtype=np.int64
        )

    def __repr__(self) -> str:
        return f"Residues(p={self.p}, precision={self.precision})"

    def asarray(self, x):
        """`x` reduced to `[0, p^K)` as an `int64` array. Accepts any integers,
        negative ones included.
        """
        if not isinstance(x, jax.Array) and np.asarray(x).dtype == object:
            x = (np.asarray(x) % self.modulus).astype(np.int64)
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise ValueError("p-adic integers must be given as integer arrays.")
        return jnp.remainder(x.astype(jnp.int64), self.modulus)

    def decode(self, x):
        """The values held by `x`, as integers in `[0, p^K)`."""
        return x

    def constant(self, c: int):
        return jnp.asarray(int(c) % self.modulus, dtype=jnp.int64)

    def zeros(self, shape):
        return jnp.zeros(shape, dtype=jnp.int64)

    def shape(self, x):
        """The shape of the values held by `x`."""
        return jnp.shape(x)

    def low(self, x, k: int):
        """`x mod p^k` as plain `int64` values, for `p^k < 2^31`."""
        return jnp.remainder(x, self.p**k)

    def shift(self, x, k):
        """`x // p^k`, for `k` an integer or an array of them."""
        return jnp.floor_divide(x, jnp.asarray(self.powers)[k])

    def truncate(self, x, k):
        """`x mod p^k`, for `k` an integer or an array of them."""
        return jnp.remainder(x, jnp.asarray(self.powers)[k])

    def p_power(self, e):
        """`p^e` as plain values, for `0 <= e <= K`."""
        return jnp.asarray(self.powers)[e]

    def equal(self, a, b):
        return a == b

    def where(self, condition, a, b):
        return jnp.where(condition, a, b)

    def take(self, x, index):
        """`x[b, index[b]]` for `x` holding values of shape `(batch, d)`."""
        return jnp.take_along_axis(x, index[:, None], axis=1)[:, 0]

    def add(self, a, b):
        a, b = jnp.broadcast_arrays(a, b)
        return jnp.remainder(a + b, self.modulus)

    def sub(self, a, b):
        a, b = jnp.broadcast_arrays(a, b)
        return jnp.remainder(a - b, self.modulus)

    def neg(self, a):
        return jnp.remainder(-a, self.modulus)

    def mul(self, a, b):
        a, b = jnp.broadcast_arrays(a, b)
        if self.reduction == "remainder":
            return jnp.remainder(a * b, self.modulus)
        if self.reduction == "barrett32":
            m, inverse = self.modulus, self._reciprocal32
            r = a * b
            q = jnp.floor(a.astype(jnp.float32) * b.astype(jnp.float32) * inverse)
            r = r - q.astype(jnp.int64) * m
            q = jnp.floor(r.astype(jnp.float32) * inverse)
            r = r - q.astype(jnp.int64) * m
            r = jnp.where(r < 0, r + m, r)
            return jnp.where(r >= m, r - m, r)
        if self.reduction == "double":
            half, rest, m = self._half, self._rest, self.modulus
            a0, a1 = jnp.remainder(a, half), jnp.floor_divide(a, half)
            b0, b1 = jnp.remainder(b, half), jnp.floor_divide(b, half)
            low = jnp.remainder(a0 * b0, m)
            middle = jnp.remainder(a0 * b1 + a1 * b0, rest)
            return jnp.remainder(low + middle * half, m)
        m = self.modulus
        q = jnp.floor(a.astype(jnp.float64) * b.astype(jnp.float64) * self._reciprocal)
        r = a * b - q.astype(jnp.int64) * m
        r = jnp.where(r < 0, r + m, r)
        return jnp.where(r >= m, r - m, r)

    def valuation(self, x):
        """`v_p(x)`, the number of trailing zero digits; `K` for `x = 0`."""
        v = jnp.zeros(jnp.shape(x), dtype=jnp.int64)
        for k in range(1, self.precision + 1):
            v = v + (jnp.remainder(x, int(self.powers[k])) == 0).astype(jnp.int64)
        return v

    def unit_inverse(self, u):
        """`1/u` for units `u` (not divisible by `p`), by Newton's iteration
        `z <- z (2 - u z)`, which doubles the number of correct digits.
        """
        z = self.asarray(jnp.asarray(self._inverses)[self.low(u, 1)])
        two = self.constant(2)
        for _ in range(max(1, math.ceil(math.log2(self.precision)))):
            z = self.mul(z, self.sub(two, self.mul(u, z)))
        return z

    def divide(self, a, b, v):
        """`a / b` for `b` of valuation exactly `v` and `a` divisible by `p^v`. The
        quotient is only defined modulo `p^{K - v}`; it is returned in
        `[0, p^{K - v})`.
        """
        unit = self.shift(b, v)
        quotient = self.mul(self.shift(a, v), self.unit_inverse(unit))
        return self.truncate(quotient, self.precision - v)

    def power(self, a, e: int):
        result = jnp.broadcast_to(self.constant(1), jnp.shape(a))
        base = a
        while e:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result


def limb_width(p: int, precision: int) -> int:
    """The largest number of digits `b` per limb for which a truncated product of
    `ceil(K / b)` limbs, carries included, fits in `int64`.
    """
    best = 1
    for b in range(1, precision + 1):
        count = -(-precision // b)
        if count * (p**b - 1) ** 2 + 2 * count * p**b >= 2**63:
            break
        best = b
    return best


class Limbs(Residues):
    """Exact arithmetic in `Z/p^K` for any number of digits, on `int64` JAX arrays
    with one extra trailing axis.

    An element is `sum_i a_i B^i` with `B = p^b` and `L = ceil(K / b)` limbs
    `0 <= a_i < B`, stored along the last axis. Since `B^L = 0` modulo `p^K`, a
    product needs only the `L (L + 1) / 2` limb products `a_i b_j` with `i + j < L`,
    followed by one pass of carries; `b` is the largest width for which these sums fit
    in `int64`. Values enter as `int64` or Python integers through `asarray` and leave
    as Python integers through `decode`.

    **Arguments:**

    - `p`: a prime.
    - `precision`: `K`, any number of digits.
    """

    extra = 1

    def __init__(self, p: int, precision: int):
        p, precision = int(p), int(precision)
        if not is_prime(p):
            raise ValueError("`p` must be a prime.")
        if precision < 1:
            raise ValueError("`precision` must be at least 1.")
        if jnp.asarray(0, dtype=jnp.int64).dtype != jnp.int64:
            raise RuntimeError(
                "p-adic arithmetic needs 64-bit integers: enable them with "
                "`jax.config.update('jax_enable_x64', True)`."
            )
        self.p = p
        self.precision = precision
        self.modulus = p**precision
        self.reduction = "limbs"
        self.width = limb_width(p, precision)
        self.limbs = -(-precision // self.width)
        self.base = p**self.width
        self.top = p ** (precision - self.width * (self.limbs - 1))
        self.powers = np.array([p**k for k in range(self.width + 1)], dtype=np.int64)
        self._inverses = np.array(
            [pow(a, -1, p) if a % p else 0 for a in range(p)], dtype=np.int64
        )

    def __repr__(self) -> str:
        return f"Limbs(p={self.p}, precision={self.precision}, limbs={self.limbs})"

    def _carry(self, sums):
        out, carry = [], 0
        for k, value in enumerate(sums):
            value = value + carry
            base = self.top if k == self.limbs - 1 else self.base
            out.append(jnp.remainder(value, base))
            carry = jnp.floor_divide(value, base)
        return jnp.stack(out, axis=-1)

    def _encode(self, x) -> list[int]:
        x = int(x) % self.modulus
        return [x // self.base**k % self.base for k in range(self.limbs)]

    def asarray(self, x):
        if not isinstance(x, jax.Array) and np.asarray(x).dtype == object:
            array = np.asarray(x)
            values = [self._encode(v) for v in array.ravel()]
            out = np.array(values, dtype=np.int64).reshape(array.shape + (self.limbs,))
            return jnp.asarray(out)
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise ValueError("p-adic integers must be given as integer arrays.")
        rest = x.astype(jnp.int64)
        out = []
        for k in range(self.limbs):
            base = self.top if k == self.limbs - 1 else self.base
            out.append(jnp.remainder(rest, base))
            rest = jnp.floor_divide(rest, self.base)
        return jnp.stack(out, axis=-1)

    def decode(self, x):
        x = np.asarray(x).astype(object)
        total = np.zeros(x.shape[:-1], dtype=object)
        for k in reversed(range(self.limbs)):
            total = total * self.base + x[..., k]
        return total

    def constant(self, c: int):
        return jnp.asarray(self._encode(c), dtype=jnp.int64)

    def zeros(self, shape):
        return jnp.zeros(tuple(shape) + (self.limbs,), dtype=jnp.int64)

    def shape(self, x):
        return jnp.shape(x)[:-1]

    def _operands(self, a, b):
        a = self.constant(a) if isinstance(a, int) else a
        b = self.constant(b) if isinstance(b, int) else b
        return jnp.broadcast_arrays(a, b)

    def add(self, a, b):
        a, b = self._operands(a, b)
        total = a + b
        return self._carry([total[..., k] for k in range(self.limbs)])

    def sub(self, a, b):
        a, b = self._operands(a, b)
        difference = a - b
        return self._carry([difference[..., k] for k in range(self.limbs)])

    def neg(self, a):
        return self.sub(jnp.zeros_like(a), a)

    def mul(self, a, b):
        a, b = self._operands(a, b)
        sums = []
        for k in range(self.limbs):
            total = a[..., 0] * b[..., k]
            for i in range(1, k + 1):
                total = total + a[..., i] * b[..., k - i]
            sums.append(total)
        return self._carry(sums)

    def low(self, x, k: int):
        if k > self.width:
            raise ValueError(f"At most {self.width} low digits can be read at once.")
        return jnp.remainder(x[..., 0], self.p**k)

    def _per_limb(self, x, k):
        """`x` and `k` broadcast to a common shape `(..., L)`, with the limb index."""
        k = jnp.asarray(k)[..., None]
        shape = jnp.broadcast_shapes(x.shape, k.shape[:-1] + (self.limbs,))
        limb = jnp.broadcast_to(jnp.arange(self.limbs), shape)
        return jnp.broadcast_to(x, shape), jnp.broadcast_to(k, shape), limb

    def shift(self, x, k):
        x, k, limb = self._per_limb(x, k)
        index = limb + k // self.width
        part = k % self.width
        moved = jnp.take_along_axis(x, jnp.clip(index, 0, self.limbs - 1), axis=-1)
        moved = jnp.where(index < self.limbs, moved, 0)
        powers = jnp.asarray(self.powers)
        upper = jnp.concatenate([moved[..., 1:], jnp.zeros_like(moved[..., :1])], -1)
        carry = jnp.remainder(upper, powers[part]) * powers[self.width - part]
        return jnp.floor_divide(moved, powers[part]) + carry

    def truncate(self, x, k):
        x, k, limb = self._per_limb(x, k)
        digits = jnp.clip(k - self.width * limb, 0, self.width)
        return jnp.remainder(x, jnp.asarray(self.powers)[digits])

    def p_power(self, e):
        return np.vectorize(lambda k: self.p ** int(k), otypes=[object])(np.asarray(e))

    def equal(self, a, b):
        return (a == b).all(axis=-1)

    def where(self, condition, a, b):
        return jnp.where(jnp.asarray(condition)[..., None], a, b)

    def take(self, x, index):
        return jnp.take_along_axis(x, index[:, None, None], axis=1)[:, 0]

    def valuation(self, x):
        digits = jnp.zeros(x.shape, dtype=jnp.int64)
        for j in range(1, self.width + 1):
            digits = digits + (jnp.remainder(x, self.p**j) == 0).astype(jnp.int64)
        v = jnp.full(x.shape[:-1], self.precision, dtype=jnp.int64)
        for k in reversed(range(self.limbs)):
            v = jnp.where(x[..., k] != 0, k * self.width + digits[..., k], v)
        return v


def make_residues(p: int, precision: int, reduction: str | None = None) -> Residues:
    """`Residues` while residues fit in one `int64`, `Limbs` beyond that or for
    `reduction="limbs"`.
    """
    if reduction == "limbs" or (
        reduction is None and int(precision) > double_word_precision(int(p))
    ):
        return Limbs(p, precision)
    return Residues(p, precision, reduction)


def scan_horner(residues: Residues, coefficients, x):
    """`sum_i c[..., i] x^i` along the last value axis of `coefficients`, by Horner's
    rule. The loop is unrolled so that it fuses with the evaluation of the
    coefficients: on a GPU this was faster than `lax.scan` (up to four times) and
    than Estrin's scheme or two interleaved chains, for every degree and in one
    word, two words and limbs alike.
    """
    xs = jnp.moveaxis(coefficients, -1 - residues.extra, 0)
    out = jnp.broadcast_to(xs[-1], jnp.broadcast_shapes(xs.shape[1:], jnp.shape(x)))
    for k in range(xs.shape[0] - 2, -1, -1):
        out = residues.add(residues.mul(out, x), xs[k])
    return out


def gather_horner(residues: Residues, table, index, x):
    """`sum_i table[index, ..., i] x^i`, gathering row `index` one coefficient at a
    time inside Horner's loop, unrolled as in `scan_horner`. `table` has shape
    `(rows, *rest, deg + 1)`, `index` and `x` broadcast to the batch; returns
    `(batch, *rest)`.
    """
    xs = jnp.moveaxis(table, -1 - residues.extra, 0)
    shape = jnp.broadcast_shapes((index.shape[0],) + xs.shape[2:], jnp.shape(x))
    out = jnp.broadcast_to(xs[-1][index], shape)
    for k in range(xs.shape[0] - 2, -1, -1):
        out = residues.add(residues.mul(out, x), xs[k][index])
    return out


def _top(left, cap: int, weight: int, room) -> int:
    top = min(left, cap)
    if weight and room != math.inf:
        top = min(top, room // weight)
    return top


def _layout(d: int, deg: int, caps, weights, limit):
    caps = (deg,) * d if caps is None else tuple(int(c) for c in caps)
    weights = (0,) * d if weights is None else tuple(int(w) for w in weights)
    return caps, weights, math.inf if limit is None else limit


def simplex(
    d: int, deg: int, caps=None, weights=None, limit=None
) -> list[tuple[int, ...]]:
    """The exponents `e` in `d` variables with `|e| <= deg`, `e_j <= caps[j]` and
    `sum_j weights[j] e_j <= limit`, in the order in which `simplex_horner` reads
    them: each coordinate from its largest value down.
    """
    caps, weights, limit = _layout(d, deg, caps, weights, limit)

    def build(j, left, room):
        if j == d:
            return [()]
        top = _top(left, caps[j], weights[j], room)
        return [
            (k,) + rest
            for k in range(top, -1, -1)
            for rest in build(j + 1, left - k, room - k * weights[j])
        ]

    return build(0, deg, limit)


def simplex_size(d: int, deg: int, caps=None, weights=None, limit=None) -> int:
    """`len(simplex(d, deg, caps, weights, limit))`, without listing the exponents."""
    caps, weights, limit = _layout(d, deg, caps, weights, limit)

    @functools.cache
    def count(j, left, room):
        if j == d:
            return 1
        top = _top(left, caps[j], weights[j], room)
        return sum(
            count(j + 1, left - k, room - k * weights[j]) for k in range(top + 1)
        )

    return count(0, deg, limit)


def simplex_horner(
    residues: Residues,
    coefficients,
    x,
    deg: int,
    index=None,
    caps=None,
    weights=None,
    limit=None,
):
    """Polynomials stored sparsely, one coefficient per exponent of
    `simplex(d, deg, caps, weights, limit)`.

    `coefficients` has shape `(rows, *rest, M)`; `x` has value shape `(batch, d)`.
    Horner's rule is nested over the stored exponents, one multiplication per
    coefficient, instead of `(deg + 1)^d` for dense storage. With `index`, point `b`
    uses row `index[b]`; otherwise `rows` is `1` or `batch`. Returns `(batch, *rest)`.
    """
    d = x.shape[1]
    caps, weights, limit = _layout(d, deg, caps, weights, limit)
    extra = residues.extra
    columns = jnp.moveaxis(coefficients, -1 - extra, 0)
    rest = columns.ndim - 2 - extra
    points = [x[:, j].reshape((-1,) + (1,) * rest + x.shape[2:]) for j in range(d)]
    position = iter(range(columns.shape[0]))

    def leaf():
        column = columns[next(position)]
        return column[index] if index is not None else column

    def nest(j, left, room):
        if j == d:
            return leaf()
        top = _top(left, caps[j], weights[j], room)
        acc = nest(j + 1, left - top, room - top * weights[j])
        for k in range(top - 1, -1, -1):
            lower = nest(j + 1, left - k, room - k * weights[j])
            acc = residues.add(residues.mul(acc, points[j]), lower)
        return acc

    return nest(0, deg, limit)
