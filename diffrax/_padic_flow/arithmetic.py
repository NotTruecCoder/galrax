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
        `"barrett"` on CPU and `"remainder"` on GPU (the faster of the two in our
        measurements) when products fit in `int64`, and to `"double"` otherwise.
        `"barrett32"` estimates the quotient in `float32` twice, avoiding `float64`
        entirely; on a laptop GPU it was slower than `"barrett"`, because the cost
        there is in the 64-bit integer products, not in `float64`.
    """

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
            if precision > single:
                reduction = "double"
            elif jax.default_backend() == "cpu":
                reduction = "barrett"
            else:
                reduction = "remainder"
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
        x = jnp.asarray(x)
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise ValueError("p-adic integers must be given as integer arrays.")
        return jnp.remainder(x.astype(jnp.int64), self.modulus)

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
        z = jnp.asarray(self._inverses)[jnp.remainder(u, self.p)]
        for _ in range(max(1, math.ceil(math.log2(self.precision)))):
            z = self.mul(z, self.sub(2, self.mul(u, z)))
        return z

    def divide(self, a, b, v):
        """`a / b` for `b` of valuation exactly `v` and `a` divisible by `p^v`. The
        quotient is only defined modulo `p^{K - v}`; it is returned in
        `[0, p^{K - v})`.
        """
        scale = jnp.asarray(self.powers)[v]
        unit = jnp.floor_divide(b, scale)
        quotient = self.mul(jnp.floor_divide(a, scale), self.unit_inverse(unit))
        return jnp.remainder(quotient, jnp.asarray(self.powers)[self.precision - v])

    def power(self, a, e: int):
        result = jnp.ones_like(a)
        base = a
        while e:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result


def scan_horner(residues: Residues, coefficients, x):
    """`sum_i c[..., i] x^i` along the last axis of `coefficients`, by Horner's rule."""
    xs = jnp.moveaxis(coefficients, -1, 0)[::-1]

    def step(acc, c):
        return residues.add(residues.mul(acc, x), c), None

    init = jnp.zeros(jnp.broadcast_shapes(xs.shape[1:], jnp.shape(x)), jnp.int64)
    out, _ = jax.lax.scan(step, init, xs)
    return out


def gather_horner(residues: Residues, table, index, x):
    """`sum_i table[index, ..., i] x^i`, gathering row `index` one coefficient at a
    time inside Horner's loop. `table` has shape `(rows, *rest, deg + 1)`, `index`
    and `x` broadcast to the batch; returns `(batch, *rest)`.
    """
    xs = jnp.moveaxis(table, -1, 0)[::-1]

    def step(acc, c):
        return residues.add(residues.mul(acc, x), c[index]), None

    shape = jnp.broadcast_shapes((index.shape[0],) + xs.shape[2:], jnp.shape(x))
    out, _ = jax.lax.scan(step, jnp.zeros(shape, jnp.int64), xs)
    return out
