import functools

import jax.numpy as jnp
import numpy as np

from .._group_algebra import Group


def _is_prime(p: int) -> bool:
    return p >= 2 and all(p % k for k in range(2, int(p**0.5) + 1))


class PAdicIntegers:
    """The `p`-adic integers `Z_p` known to `n` digits: the ring `Z/p^n`.

    The point `x = x_0 + x_1 p + ... + x_{n-1} p^{n-1}` stands for the ball
    `x + p^n Z_p` of radius `p^{-n}`, so functions on `Z/p^n` are the functions on
    `Z_p` that are constant on such balls. They are arrays of shape `(..., p^n)`.

    Distances are `|x - y|_p = p^{-v_p(x - y)}`, where the valuation `v_p` counts the
    trailing zero digits. Integrals use the Haar measure of `Z_p`, which gives every
    point of `Z/p^n` the mass `p^{-n}`.

    **Arguments:**

    - `p`: a prime.
    - `n`: the number of digits, at least 1.
    """

    def __init__(self, p: int, n: int):
        p = int(p)
        n = int(n)
        if not _is_prime(p):
            raise ValueError("`p` must be a prime.")
        if n < 1:
            raise ValueError("`n` must be at least 1.")
        self.p = p
        self.n = n
        self.size = p**n

    def __repr__(self) -> str:
        return f"PAdicIntegers(p={self.p}, n={self.n})"

    @functools.cached_property
    def group(self) -> Group:
        """The additive group `Z/p^n`, built on first use. It stores a Cayley table of
        `p^{2n}` entries, so it is meant for a few thousand points at most.
        """
        return Group.cyclic(self.size)

    def points(self) -> np.ndarray:
        return np.arange(self.size)

    def _check_points(self, x) -> np.ndarray:
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("Points of Z/p^n must be integers.")
        return np.asarray(x.astype(np.int64) % self.size)

    def from_fraction(self, numerator: int, denominator: int = 1) -> int:
        """The point of `Z/p^n` closest to the rational number
        `numerator / denominator`, which must be a `p`-adic integer (`p` does not
        divide the denominator).
        """
        if denominator % self.p == 0:
            raise ValueError("The denominator must not be divisible by `p`.")
        return numerator * pow(denominator, -1, self.size) % self.size

    def digits(self, x) -> np.ndarray:
        """The `p`-adic digits of `x`, lowest first: shape `x.shape + (n,)`."""
        x = self._check_points(x)
        return (x[..., None] // self.p ** np.arange(self.n)) % self.p

    def from_digits(self, digits) -> np.ndarray:
        digits = np.asarray(digits)
        if digits.shape[-1:] != (self.n,):
            raise ValueError(f"Digits must have shape (..., {self.n}).")
        if digits.min(initial=0) < 0 or digits.max(initial=0) >= self.p:
            raise ValueError(f"Digits must lie in 0, ..., {self.p - 1}.")
        return np.asarray(digits @ self.p ** np.arange(self.n))

    def valuation(self, x) -> np.ndarray:
        """`v_p(x)`, the number of trailing zero digits; `n` for `x = 0`."""
        x = self._check_points(x)
        v = np.zeros(x.shape, dtype=np.int64)
        for k in range(1, self.n):
            v += x % self.p**k == 0
        v[x == 0] = self.n
        return v

    def norm(self, x) -> np.ndarray:
        """`|x|_p = p^{-v_p(x)}`, and `0` for `x = 0`."""
        x = self._check_points(x)
        return np.where(x == 0, 0.0, float(self.p) ** -self.valuation(x))

    def distance(self, x, y) -> np.ndarray:
        """`|x - y|_p`."""
        return self.norm(self._check_points(x) - self._check_points(y))

    def ball(self, center: int | np.integer, level: int | np.integer) -> np.ndarray:
        """The indicator of the ball `center + p^level Z_p`, of radius `p^{-level}`,
        as a boolean array of shape `(p^n,)`.
        """
        if not 0 <= level <= self.n:
            raise ValueError(f"`level` must lie in 0, ..., {self.n}.")
        modulus = self.p**level
        return self.points() % modulus == int(center) % modulus

    def integrate(self, f):
        """`int_{Z_p} f(x) dx` for `f` of shape `(..., p^n)`."""
        f = jnp.asarray(f)
        if f.ndim == 0 or f.shape[-1] != self.size:
            raise ValueError(f"Functions must have shape (..., {self.size}).")
        return f.sum(axis=-1) / self.size

    def inner(self, f, g):
        """`<f, g> = int_{Z_p} f(x) conj(g(x)) dx`."""
        f = jnp.asarray(f)
        g = jnp.asarray(g)
        dtype = np.result_type(f.dtype, g.dtype)
        return self.integrate(f.astype(dtype) * jnp.conj(g.astype(dtype)))
