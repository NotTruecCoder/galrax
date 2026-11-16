from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from .fourier import convolution
from .group import Group


class GroupODE:
    """An initial value problem for a function `p: G -> C` on a finite group `G`:

    `dp/dt = lambda * p + p * mu + N(t, p, args)`,  `p(t0) = y0`,

    where `*` is the convolution `(f * h)(x) = sum_y f(y) h(y^{-1} x)`. The kernels
    `lambda` and `mu` do not depend on time. Functions on `G` are arrays of shape
    `(..., n)`, indexed by the element numbers of `group`.

    **Arguments:**

    - `group`: the group `G`.
    - `y0`: the initial condition, of shape `(..., n)`.
    - `t0`: the initial time.
    - `t1`: the final time.
    - `left_kernel`: `lambda`, of shape `(n,)`, or `None` for no left convolution.
    - `right_kernel`: `mu`, of shape `(n,)`, or `None` for no right convolution.
    - `nonlinearity`: `N(t, p, args)`, a JAX function returning an array of the same
        shape as `p`, or `None` for a linear problem. For a real problem (real `y0`
        and kernels) it must map real arrays to real arrays.
    - `args`: passed to `nonlinearity`.
    """

    def __init__(
        self,
        group: Group,
        y0,
        t0: float,
        t1: float,
        left_kernel=None,
        right_kernel=None,
        nonlinearity: Callable[[Any, Any, Any], Any] | None = None,
        args: Any = None,
    ):
        if not isinstance(group, Group):
            raise TypeError("`group` must be a `Group`.")
        n = group.n
        y0 = np.asarray(y0)
        if y0.ndim == 0 or y0.shape[-1] != n:
            raise ValueError(f"`y0` must have shape (..., {n}).")
        if not np.issubdtype(y0.dtype, np.number) or not np.isfinite(y0).all():
            raise ValueError("`y0` must be a finite numeric array.")
        t0 = float(t0)
        t1 = float(t1)
        if not (np.isfinite(t0) and np.isfinite(t1)):
            raise ValueError("`t0` and `t1` must be finite.")
        if t0 == t1:
            raise ValueError("`t0` and `t1` must differ.")
        if nonlinearity is not None and not callable(nonlinearity):
            raise TypeError("`nonlinearity` must be callable or `None`.")

        self.group = group
        self.y0 = y0
        self.t0 = t0
        self.t1 = t1
        self.left_kernel = _check_kernel(left_kernel, n, "left_kernel")
        self.right_kernel = _check_kernel(right_kernel, n, "right_kernel")
        self.nonlinearity = nonlinearity
        self.args = args

        if nonlinearity is not None:
            out = jnp.asarray(nonlinearity(t0, jnp.asarray(y0), args))
            if out.shape != y0.shape:
                raise ValueError(
                    "`nonlinearity` must return an array of the same shape as `p`."
                )
            if self.is_real and jnp.iscomplexobj(out):
                raise ValueError(
                    "`nonlinearity` must return real values for a real problem."
                )

    @property
    def is_linear(self) -> bool:
        """Whether the problem has no nonlinear term."""
        return self.nonlinearity is None

    @property
    def is_real(self) -> bool:
        """Whether `y0` and the kernels are real, so that the solution is real."""
        kernels = [k for k in (self.left_kernel, self.right_kernel) if k is not None]
        return np.isrealobj(self.y0) and all(np.isrealobj(k) for k in kernels)

    def linear(self, p):
        """The linear part `lambda * p + p * mu`."""
        p = jnp.asarray(p)
        kernels = [k for k in (self.left_kernel, self.right_kernel) if k is not None]
        p = p.astype(np.result_type(p.dtype, *kernels))
        out = jnp.zeros_like(p)
        if self.left_kernel is not None:
            out = out + convolution(self.group, self.left_kernel, p)
        if self.right_kernel is not None:
            out = out + convolution(self.group, p, self.right_kernel)
        return out

    def vector_field(self, t, p):
        """The right hand side `lambda * p + p * mu + N(t, p, args)`."""
        out = self.linear(p)
        if self.nonlinearity is not None:
            out = out + jnp.asarray(self.nonlinearity(t, p, self.args)).astype(
                out.dtype
            )
        return out

    @classmethod
    def heat_equation(
        cls, group: Group, generators, y0, t0: float, t1: float, rate: float = 1.0
    ) -> "GroupODE":
        """The heat equation on the Cayley graph of `G` with respect to the symmetric
        generating set `S = generators + generators^{-1}` (without `e`):

        `dp/dt = rate * (p * u_S - p)`,

        where `u_S` is the uniform probability distribution on `S`. Equivalently,
        `dp/dt(x) = rate * (mean_{s in S} p(x s^{-1}) - p(x))`.
        """
        generators = np.unique(np.asarray(generators, dtype=np.int64))
        if generators.ndim != 1 or len(generators) == 0:
            raise ValueError("`generators` must be a non-empty list of elements.")
        if generators.min() < 0 or generators.max() >= group.n:
            raise ValueError("`generators` must be element numbers of `group`.")
        symmetric = np.union1d(generators, group.inv[generators])
        symmetric = symmetric[symmetric != group.ed]
        if len(symmetric) == 0:
            raise ValueError("`generators` must contain an element other than `e`.")
        kernel = np.zeros(group.n)
        kernel[symmetric] = rate / len(symmetric)
        kernel[group.ed] -= rate
        return cls(group, y0, t0, t1, right_kernel=kernel)


def _check_kernel(kernel, n: int, name: str):
    if kernel is None:
        return None
    kernel = np.asarray(kernel)
    if kernel.shape != (n,):
        raise ValueError(f"`{name}` must have shape ({n},).")
    if not np.issubdtype(kernel.dtype, np.number) or not np.isfinite(kernel).all():
        raise ValueError(f"`{name}` must be a finite numeric array.")
    return kernel
