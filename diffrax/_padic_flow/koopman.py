import jax
import jax.numpy as jnp
import numpy as np

from .._ultrametric import UltrametricOperator
from .flow import PAdicFlow


class KoopmanOperator:
    """The Koopman operator `U u = u o phi_tau` of a `PAdicFlow` on functions on
    `(Z/p^n)^d`, arrays of shape `(..., p^{nd})` indexed by the points in C order.

    `phi_tau` permutes the balls of radius `p^{-n}`. Ordering each cycle of the
    permutation by flow time turns `U` into a cyclic shift, so it is diagonalised by
    the FFT of each cycle: this is the Fourier transform in rectifying coordinates.

    **Arguments:**

    - `flow`: the `PAdicFlow`.
    - `level`: `n`, at most the precision of `flow`.
    - `time`: `tau`, an integer read modulo `p^K`.
    """

    def __init__(self, flow: PAdicFlow, level: int, time: int = 1):
        if not isinstance(flow, PAdicFlow):
            raise TypeError("`flow` must be a `PAdicFlow`.")
        level = int(level)
        if not 1 <= level <= flow.precision:
            raise ValueError(f"`level` must lie in 1, ..., {flow.precision}.")
        self.flow = flow
        self.level = level
        self.time = int(time) % flow.modulus
        d = flow.dimension
        side = flow.p**level
        self.size = side**d
        points = np.indices((side,) * d).reshape(d, -1).T

        def index(images):
            cells = np.asarray(jnp.remainder(images, side)).reshape(-1, d)
            return np.ravel_multi_index(tuple(cells.T), (side,) * d)

        images = flow.flow(self.time, points if d > 1 else points[:, 0])
        self.permutation = index(images)

        labels = np.arange(self.size)
        jump = self.permutation.copy()
        for _ in range(max(1, int(np.ceil(np.log2(self.size))))):
            labels = np.minimum(labels, labels[jump])
            jump = jump[jump]
        representatives, lengths = np.unique(labels, return_counts=True)
        self.cycle_lengths = lengths

        owner = np.repeat(np.arange(len(representatives)), lengths)
        offset = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        steps = np.arange(self.size) - np.repeat(offset, lengths)
        start = points[representatives[owner]]
        times = (steps.astype(object) * self.time % flow.modulus).astype(np.int64)
        visited = index(flow.flow(times, start if d > 1 else start[:, 0]))

        self.groups = []
        for length in np.unique(lengths):
            chosen = np.flatnonzero(lengths[owner] == length)
            order = visited[chosen].reshape(-1, length)
            self.groups.append((int(length), order))
        if not np.array_equal(np.sort(visited), np.arange(self.size)):
            raise RuntimeError("Cycles of the flow do not cover every point.")

    def __call__(self, u):
        """`U u = u o phi_tau`."""
        u = jnp.asarray(u)
        self._check(u)
        return u[..., self.permutation]

    def _check(self, u):
        if u.ndim == 0 or u.shape[-1] != self.size:
            raise ValueError(f"Functions must have shape (..., {self.size}).")

    def evolve_jumps(self, u, t, rate: float = 1.0):
        """`exp(t rate (U - I)) u`: the expected value of `u` after a Poisson number of
        flow steps of length `tau`, at rate `rate`, up to time `t`. On a cycle of
        length `L` the eigenvalues of `U` are the `L`-th roots of unity `w`, and the
        factor is `exp(t rate (w - 1))`.
        """
        u = jnp.asarray(u)
        self._check(u)
        real = not jnp.iscomplexobj(u)
        u = u.astype(jnp.complex128)
        scale = jnp.asarray(t, dtype=jnp.float64) * rate
        out = jnp.zeros_like(u)
        batch = u.shape[:-1]
        for length, order in self.groups:
            angle = 2 * np.pi * np.arange(length) / length
            decay = jnp.exp(scale * (np.cos(angle) - 1))
            factor = jax.lax.complex(
                decay * jnp.cos(scale * np.sin(angle)),
                decay * jnp.sin(scale * np.sin(angle)),
            )
            block = jnp.fft.fft(u[..., order], axis=-1)
            block = jnp.fft.ifft(block * jnp.broadcast_to(factor, block.shape), axis=-1)
            out = out.at[..., order.reshape(-1)].set(block.reshape(batch + (-1,)))
        return out.real if real else out


def advection_diffusion(
    koopman: KoopmanOperator, operator: UltrametricOperator, u0, t, rate: float = 1.0
):
    """The exact solution at time `t` of

    `du/dt = L u + rate (U - I) u`,

    ultrametric diffusion `L` together with jumps along the `p`-adic flow. The flow is
    an isometry of `Z_p` that preserves the Haar measure, so `U` commutes with every
    radial convolution `L`, and the solution is `e^{tL} e^{t rate (U - I)} u0`.
    """
    integers = operator.integers
    flow = koopman.flow
    if flow.dimension != 1:
        raise ValueError("Ultrametric diffusion is defined on Z_p, so `d` must be 1.")
    if integers.p != flow.p or integers.n != koopman.level:
        raise ValueError("`koopman` and `operator` must live on the same Z/p^n.")
    return operator.evolve(koopman.evolve_jumps(u0, t, rate), t)
