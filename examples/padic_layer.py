"""Invertible p-adic layers: the time-one map phi^X_1 of a polynomial field X = p^c Z
on Z_p^d. The inverse is phi^X_{-1}, and the layer is an isometry, so it permutes
every (Z/p^k)^d and preserves the Haar measure.

Training needs no gradients. If X is known modulo p^k (k >= c), then for
X' = X + p^k Y

    phi^{X'}_1(x) = phi^X_1(x) + p^k Y(x)  mod p^{k + 1},

since Dphi = I and phi_s(x) = x modulo p^c. So each digit of the coefficients of X
solves one linear system modulo p, in the monomials of degree < p evaluated at the
data. If a residual is not divisible by p^k, the pairs come from no field of the
assumed shape and training stops.

    python padic_layer.py
"""

import math

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

from galrax import padic as pf  # noqa: E402
from galrax.padic.normal_form import _monomials, _solve  # noqa: E402


def monomials(d: int, degree: int) -> list[tuple]:
    return [e for n in range(degree + 1) for e in _monomials(d, n)]


def apply(p, precision, field, t, x):
    """`phi^X_t(x)` for points of shape `(n, d)`."""
    x = np.asarray(x, dtype=object)
    if not any(field):
        return x % p**precision
    flow = pf.PAdicFlow(p, precision, field)
    y = flow.flow(t, x[:, 0] if x.shape[1] == 1 else x)
    return np.asarray(np.asarray(y), dtype=object).reshape(x.shape)


def fit(p, precision, x, y, degree, c=None):
    """The field `X = p^c Z`, `Z` of degree `<= degree`, with `phi^X_1(x) = y` modulo
    `p^precision`, digit by digit. `degree < p` makes the monomials independent
    functions on `F_p^d`, so `X` is unique when the data covers `F_p^d`.
    """
    x, y = (np.asarray(v, dtype=object) for v in (x, y))
    d, m = x.shape[1], p**precision
    c = (2 if p == 2 else 1) if c is None else c
    basis = monomials(d, degree)
    rows = [
        [math.prod(int(v) ** k for v, k in zip(point, e)) % p for e in basis]
        for point in x
    ]
    field = [{} for _ in range(d)]
    for k in range(c, precision + 1):
        residual = (y - apply(p, precision, field, 1, x)) % m
        if np.any(residual % p**k):
            raise ValueError(
                f"The pairs are no time-one map of such a field (digit {k})."
            )
        if k == precision:
            return field
        for i in range(d):
            digits = _solve(rows, [int(r) // p**k % p for r in residual[:, i]], p, 1)
            if digits is None:
                raise ValueError(f"No field of degree {degree} fits digit {k}.")
            for e, v in zip(basis, digits):
                if v:
                    field[i][e] = (field[i].get(e, 0) + v * p**k) % m


if __name__ == "__main__":
    p, K, d, degree = 3, 6, 2, 2
    rng = np.random.default_rng(0)
    planted = [
        {e: int(rng.integers(p ** (K - 1))) * p for e in monomials(d, degree)}
        for _ in range(d)
    ]
    x = rng.integers(0, p**K, (200, d)).astype(object)
    y = apply(p, K, planted, 1, x)
    found = fit(p, K, x, y, degree)
    print(
        "recovered:",
        found == [{e: v % p**K for e, v in f.items() if v % p**K} for f in planted],
    )
    print("inverse:", np.array_equal(apply(p, K, found, -1, y), x % p**K))
