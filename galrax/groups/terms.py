import abc

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.tree_util as jtu
from diffrax import AbstractTerm

from .._types import Args, RealScalarLike, VF, Y


def to_complex(y) -> list:
    """Complex blocks from real coordinates: each block `(2, ...)` becomes
    `block[0] + i block[1]`.
    """
    return [jax.lax.complex(block[0], block[1]) for block in y]


def to_real(x) -> tuple:
    """Real coordinates from complex blocks; the inverse of `to_complex`."""
    return tuple(jnp.stack([block.real, block.imag]) for block in x)


class AbstractExponentialTerm(AbstractTerm):
    """A linear vector field `y -> L y` whose flow `e^{tau L}` is known exactly.

    It acts on real coordinates: tuples of arrays of shape `(2, ...)` holding real and
    imaginary parts. `exponential` and `apply` work with the complex blocks instead,
    so that `Lawson` can combine many flows without converting back and forth.
    """

    @abc.abstractmethod
    def exponential(self, tau):
        """The flow `e^{tau L}`, in whatever form `apply` expects."""

    @staticmethod
    @abc.abstractmethod
    def apply(exponential, x) -> list:
        """`e^{tau L} x` for complex blocks `x`, given `exponential(tau)`."""

    def exponentials(self, taus) -> list:
        """`[exponential(tau) for tau in taus]`; override to share the work."""
        return [self.exponential(tau) for tau in taus]

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: VF, control: RealScalarLike) -> Y:
        return jtu.tree_map(lambda v: v * control, vf)


class FourierLinearTerm(AbstractExponentialTerm):
    """The linear part `L p = lambda * p + p * mu` of a `GroupODE`, in the real
    coordinates of `AbstractFourierTransform.to_coordinates`. In the block of `rho`,
    with `X` the coordinates, `A = lambdahat(rho)` and `B = muhat(rho)`,

    `L X = A X + X B`,  `e^{tau L} X = e^{tau A} X e^{tau B}`.

    **Arguments:**

    - `left`: the blocks `A`, one per representation, or `None` where `lambda` is
        absent. Each block is stored as an array of shape `(2, d, d)` holding its real
        and imaginary parts, so that the term contains no complex arrays.
    - `right`: the blocks `B`, likewise.
    """

    left: tuple
    right: tuple

    @classmethod
    def from_problem(cls, problem, transform) -> "FourierLinearTerm":
        """The term of a `GroupODE` in the coordinates of an
        `AbstractFourierTransform`.
        """

        def blocks(kernel):
            if kernel is None:
                return (None,) * len(transform.degrees)
            return to_real(transform.forward(kernel))

        return cls(blocks(problem.left_kernel), blocks(problem.right_kernel))

    def _blocks(self):
        def complex_or_none(blocks):
            return [None if b is None else jax.lax.complex(b[0], b[1]) for b in blocks]

        return complex_or_none(self.left), complex_or_none(self.right)

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        out = []
        left, right = self._blocks()
        for a, b, x in zip(left, right, to_complex(y)):
            value = jnp.zeros_like(x)
            if a is not None:
                value = value + a @ x
            if b is not None:
                value = value + x @ b
            out.append(value)
        return to_real(out)

    def exponential(self, tau) -> list:
        """The pairs `(e^{tau A}, e^{tau B})`, one per representation (`None` where the
        kernel is absent).
        """
        left, right = self._blocks()
        return [
            (
                None if a is None else jsl.expm(jnp.asarray(tau, a.dtype) * a),
                None if b is None else jsl.expm(jnp.asarray(tau, b.dtype) * b),
            )
            for a, b in zip(left, right)
        ]

    def exponentials(self, taus) -> list:
        """`exponential(tau)` for each of `taus`, with one batched `expm` per block
        size: separate `expm` calls make the compiled program, and its compile time,
        grow with the number of times and representations.
        """
        taus = jnp.stack([jnp.asarray(tau) for tau in taus])

        def batched(blocks):
            out = [None] * len(blocks)
            sizes = {}
            for k, b in enumerate(blocks):
                if b is not None:
                    sizes.setdefault(b.shape, []).append(k)
            for (d, _), ks in sizes.items():
                stack = jnp.stack([blocks[k] for k in ks])
                scaled = taus[:, None, None, None].astype(stack.dtype) * stack[None]
                e = jsl.expm(scaled.reshape(-1, d, d)).reshape(scaled.shape)
                for j, k in enumerate(ks):
                    out[k] = e[:, j]
            return out

        left, right = self._blocks()
        left, right = batched(left), batched(right)
        return [
            [
                (None if a is None else a[i], None if b is None else b[i])
                for a, b in zip(left, right)
            ]
            for i in range(len(taus))
        ]

    @staticmethod
    def apply(exponential, x) -> list:
        out = []
        for (left, right), block in zip(exponential, x):
            if left is not None:
                block = left @ block
            if right is not None:
                block = block @ right
            out.append(block)
        return out


class DiagonalLinearTerm(AbstractExponentialTerm):
    """A linear part that is diagonal in the coordinates: `L x = r x` entrywise, with
    complex rates `r`, so that `e^{tau L} x = e^{tau r} x`.

    **Arguments:**

    - `rates`: one array per block of coordinates, of shape `(2, ...)` holding the real
        and imaginary parts of `r`. Each must broadcast against its block.
    """

    rates: tuple

    def _rates(self) -> list:
        return to_complex(self.rates)

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        x = to_complex(y)
        return to_real(self.apply(self._rates(), x))

    def exponential(self, tau) -> list:
        return [jnp.exp(jnp.asarray(tau, r.dtype) * r) for r in self._rates()]

    @staticmethod
    def apply(exponential, x) -> list:
        return [jnp.broadcast_to(e, b.shape) * b for e, b in zip(exponential, x)]
