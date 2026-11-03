import functools
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from .._group_algebra import GroupODE
from .integers import PAdicIntegers
from .wavelets import KozyrevTransform


class UltrametricOperator:
    """An ultrametric diffusion operator on `Z_p`:

    `(L f)(x) = int_{Z_p} w(|x - y|_p) (f(y) - f(x)) dy - a f(x)`,

    with a jump rate `w >= 0` that depends only on the distance, and an absorption
    rate `a >= 0`. It generates the random walk on `Z_p` that jumps from `x` into the
    sphere `|y - x|_p = r` at a rate proportional to `w(r)`, and is killed at rate
    `a`: a model of diffusion in a hierarchical energy landscape (Avetisov, Bikulov,
    Kozyrev).

    Functions constant on the balls of radius `p^{-n}`, that is functions on `Z/p^n`,
    are mapped to functions of the same kind: from `x`, every other such ball lies at
    a single distance. So on `Z/p^n` the operator is exactly

    `(L f)(x) = p^{-n} sum_{y != x} w(|x - y|_p) (f(y) - f(x)) - a f(x) = (f * k)(x)`,

    a convolution with the radial `kernel` `k`, and no discretisation error is made.
    Its eigenfunctions are the Kozyrev wavelets: on level `m` (support of radius
    `p^{-m}`) the eigenvalue is

    `lambda_m = -w(p^{-m}) p^{-m} - (1 - 1/p) sum_{k < m} w(p^{-k}) p^{-k} - a`,

    and on constants it is `-a`.

    **Arguments:**

    - `integers`: the `PAdicIntegers` `Z/p^n`.
    - `weights`: `w(p^{-k})` for `k = 0, ..., n - 1`, the rates between points at
        distance `p^{-k}`; an array of shape `(n,)`.
    - `absorption`: `a`.
    """

    def __init__(self, integers: PAdicIntegers, weights, absorption: float = 0.0):
        if not isinstance(integers, PAdicIntegers):
            raise TypeError("`integers` must be `PAdicIntegers`.")
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (integers.n,):
            raise ValueError(f"`weights` must have shape ({integers.n},).")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("`weights` must be finite and non-negative.")
        absorption = float(absorption)
        if not np.isfinite(absorption) or absorption < 0:
            raise ValueError("`absorption` must be finite and non-negative.")
        self.integers = integers
        self.weights = weights
        self.absorption = absorption

    @classmethod
    def from_function(
        cls,
        integers: PAdicIntegers,
        rate: Callable[[float], float],
        absorption: float = 0.0,
    ) -> "UltrametricOperator":
        """The operator with jump rate `w(r) = rate(r)`."""
        radii = float(integers.p) ** -np.arange(integers.n)
        return cls(integers, [rate(float(r)) for r in radii], absorption)

    @classmethod
    def vladimirov(
        cls, integers: PAdicIntegers, alpha: float, absorbing: bool = True
    ) -> "UltrametricOperator":
        """`L = -D^alpha`, minus the Vladimirov operator of order `alpha > 0`:

        `D^alpha f(x) = (p^alpha - 1) / (1 - p^{-alpha-1})
        int_{Q_p} (f(x) - f(y)) / |x - y|_p^{alpha + 1} dy`.

        With `absorbing=True` it acts on functions supported in `Z_p` and is restricted
        back to `Z_p`: the walk is killed when it jumps out, at rate
        `(1 - 1/p) / (1 - p^{-alpha-1})`. The eigenvalues of `D^alpha` are then
        `p^{alpha (m + 1)}` on level `m` and the killing rate on constants. With
        `absorbing=False` jumps out of `Z_p` are suppressed instead, and mass is
        conserved.
        """
        alpha = float(alpha)
        if not alpha > 0:
            raise ValueError("`alpha` must be positive.")
        p = integers.p
        scale = (p**alpha - 1) / (1 - p ** (-alpha - 1))
        rate = (1 - 1 / p) / (1 - p ** (-alpha - 1))
        weights = scale * float(p) ** ((alpha + 1) * np.arange(integers.n))
        return cls(integers, weights, rate if absorbing else 0.0)

    @functools.cached_property
    def kernel(self) -> np.ndarray:
        """The kernel `k` with `L f = f * k`: `k(z) = p^{-n} w(|z|_p)` for `z != 0`,
        and `k(0)` such that `sum_z k(z) = -a`.
        """
        integers = self.integers
        points = integers.points()
        valuation = integers.valuation(points)
        kernel = np.zeros(integers.size)
        nonzero = points != 0
        kernel[nonzero] = self.weights[valuation[nonzero]] / integers.size
        kernel[0] = -kernel.sum() - self.absorption
        return kernel

    @property
    def eigenvalues(self) -> np.ndarray:
        """`[-a, lambda_0, ..., lambda_{n-1}]`: the eigenvalue on constants, then on
        the wavelets of each level.
        """
        p, n = self.integers.p, self.integers.n
        terms = self.weights * float(p) ** -np.arange(n)
        below = np.concatenate([[0.0], np.cumsum(terms)[:-1]])
        levels = -terms - (1 - 1 / p) * below - self.absorption
        return np.concatenate([[-self.absorption], levels])

    @functools.cached_property
    def transform(self) -> KozyrevTransform:
        return KozyrevTransform(self.integers)

    def __call__(self, f):
        """`L f` for `f` of shape `(..., p^n)`, in `O(n p^n)` operations through sums
        over balls.
        """
        f = jnp.asarray(f)
        integers = self.integers
        p, n, size = integers.p, integers.n, integers.size
        if f.ndim == 0 or f.shape[-1] != size:
            raise ValueError(f"Functions must have shape (..., {size}).")
        batch = f.shape[:-1]

        def spread(values):
            copies = size // values.shape[-1]
            shape = batch + (copies, values.shape[-1])
            full = jnp.broadcast_to(values[..., None, :], shape)
            return full.reshape(batch + (size,))

        jumps = jnp.zeros_like(f)
        inner = f
        for k in reversed(range(n)):
            outer = inner.reshape(batch + (p, p**k)).sum(axis=-2)
            jumps = jumps + float(self.weights[k]) * (spread(outer) - spread(inner))
            inner = outer
        rate = (1 - 1 / p) * float(np.sum(self.weights * float(p) ** -np.arange(n)))
        return jumps / size - (rate + self.absorption) * f

    def evolve(self, f, t):
        """`e^{t L} f`, the exact solution of `du/dt = L u` with `u(0) = f`, through
        the Kozyrev wavelets.
        """
        f = jnp.asarray(f)
        c = self.transform.forward(f)
        rates = jnp.asarray(self.eigenvalues[self.transform.levels + 1])
        growth = jnp.exp(jnp.asarray(t, rates.dtype) * rates).astype(c.dtype)
        out = self.transform.inverse(jnp.broadcast_to(growth, c.shape) * c)
        return out if jnp.iscomplexobj(f) else out.real

    def problem(
        self,
        y0,
        t0: float,
        t1: float,
        nonlinearity: Callable[[Any, Any, Any], Any] | None = None,
        args: Any = None,
    ) -> GroupODE:
        """The `GroupODE` `du/dt = L u + N(t, u, args)` on `Z/p^n`, to be solved with
        `diffrax.group_algebra.solve_group_ode`, best with
        `transform=KozyrevTransform(integers)`.
        """
        return GroupODE(
            self.integers.group,
            y0,
            t0,
            t1,
            right_kernel=self.kernel,
            nonlinearity=nonlinearity,
            args=args,
        )
