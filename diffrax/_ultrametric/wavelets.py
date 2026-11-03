import jax
import jax.numpy as jnp
import numpy as np

from .._group_algebra import AbstractSpectralTransform, DiagonalLinearTerm, Group
from .integers import PAdicIntegers


def _complex_dtype():
    return jnp.asarray(1j).dtype


class KozyrevTransform(AbstractSpectralTransform):
    """Expansion of functions on `Z/p^n` in the Kozyrev wavelet basis of `L^2(Z_p)`.

    The basis consists of the constant `1` and, for every level `m = 0, ..., n - 1`,
    every ball `B = a + p^m Z_p` with `a` in `Z/p^m` and every `j = 1, ..., p - 1`, the
    wavelet

    `psi_{m,a,j}(x) = p^{m/2} exp(2 pi i j (x - a) / p^{m+1}) 1_B(x)`.

    On `B` it is a character of the digit `x_m`, so it has mean zero on `B` and is
    constant on the `p` balls of radius `p^{-m-1}` inside `B`. The basis is
    orthonormal for `<f, g> = int f conj(g) dx`, and every radial convolution, in
    particular every `UltrametricOperator`, is diagonal in it with an eigenvalue that
    depends only on the level `m`.

    Coefficients are arrays of shape `(..., p^n)`: index `0` is the constant, and level
    `m` occupies indices `p^m, ..., p^{m+1} - 1`, ordered by `a` and then `j`. Both
    directions cost `O(p^n)` operations: one FFT of size `p` per ball.

    As an `AbstractSpectralTransform` it solves `GroupODE`s on `Z/p^n` with radial
    kernels; its coordinates are `sqrt(p^n)` times the coefficients, which makes it an
    isometry of the values.

    **Arguments:**

    - `integers`: the `PAdicIntegers` the functions live on.
    """

    def __init__(self, integers: PAdicIntegers):
        if not isinstance(integers, PAdicIntegers):
            raise TypeError("`integers` must be `PAdicIntegers`.")
        self.integers = integers
        p, n = integers.p, integers.n
        levels, centers, frequencies = [-1], [0], [0]
        for m in range(n):
            a, j = np.meshgrid(np.arange(p**m), np.arange(1, p), indexing="ij")
            levels.extend([m] * a.size)
            centers.extend(a.reshape(-1).tolist())
            frequencies.extend(j.reshape(-1).tolist())
        self.levels = np.array(levels)
        self.centers = np.array(centers)
        self.frequencies = np.array(frequencies)

    @property
    def group(self) -> Group:
        return self.integers.group

    def forward(self, f):
        """The coefficients `<f, psi>` of `f`, of shape `(..., p^n)`."""
        f = self._check(f, "Functions")
        p, n, size = self.integers.p, self.integers.n, self.integers.size
        batch = f.shape[:-1]
        sums = f
        levels = []
        for m in reversed(range(n)):
            grid = sums.reshape(batch + (p, p**m))
            spectrum = jnp.fft.fft(grid, axis=-2)[..., 1:, :]
            block = jnp.swapaxes(spectrum, -1, -2).reshape(batch + (-1,))
            levels.append(block * (p ** (m / 2) / size))
            sums = grid.sum(axis=-2)
        return jnp.concatenate([sums / size] + levels[::-1], axis=-1)

    def inverse(self, c):
        """The function `sum_k c_k psi_k`, of shape `(..., p^n)`."""
        c = self._check(c, "Coefficients")
        p, n = self.integers.p, self.integers.n
        batch = c.shape[:-1]
        values = c[..., :1]
        for m in range(n):
            block = c[..., p**m : p ** (m + 1)].reshape(batch + (p**m, p - 1))
            padded = jnp.concatenate([jnp.zeros_like(block[..., :1]), block], axis=-1)
            waves = jnp.swapaxes(jnp.fft.ifft(padded, axis=-1), -1, -2)
            values = values[..., None, :] + waves * (p * p ** (m / 2))
            values = values.reshape(batch + (p ** (m + 1),))
        return values

    def basis(self) -> np.ndarray:
        """All wavelets as an array of shape `(p^n, p^n)`: row `k` is `psi_k`."""
        return np.asarray(self.inverse(np.eye(self.integers.size)))

    def to_coordinates(self, f) -> tuple:
        c = self.forward(f) * float(np.sqrt(self.integers.size))
        return (jnp.stack([c.real, c.imag]),)

    def from_coordinates(self, y):
        (block,) = y
        c = jax.lax.complex(block[0], block[1]) / float(np.sqrt(self.integers.size))
        return self.inverse(c)

    def level_eigenvalues(self, kernel) -> np.ndarray:
        """The eigenvalues of `f -> f * kernel` for a radial `kernel` (a function of
        `|x|_p` only): an array of shape `(n + 1,)` with the eigenvalue on constants
        first, then on the wavelets of levels `0, ..., n - 1`.

        The wavelet `psi_{m,0,1}` gives `lambda_m = sum_t kernel(p^m t) w^{t mod p}`
        with `w = exp(2 pi i / p)`, summed over `t` in `Z/p^{n-m}`.
        """
        kernel = self._check_radial(kernel)
        p, n = self.integers.p, self.integers.n
        out = [kernel.sum()]
        for m in range(n):
            t = np.arange(p ** (n - m))
            out.append(np.sum(kernel[p**m * t] * np.exp(2j * np.pi * (t % p) / p)))
        out = np.array(out)
        return out.real if np.isrealobj(kernel) else out

    def spectrum(self, kernel) -> np.ndarray:
        """The eigenvalue of `f -> f * kernel` on each basis function, of shape
        `(p^n,)`, for a radial `kernel`.
        """
        return self.level_eigenvalues(kernel)[self.levels + 1]

    def linear_term(self, problem) -> DiagonalLinearTerm:
        kernels = [
            k for k in (problem.left_kernel, problem.right_kernel) if k is not None
        ]
        total = sum(kernels, np.zeros(self.integers.size))
        rates = self.spectrum(total).astype(np.complex128)
        return DiagonalLinearTerm((jnp.asarray(np.stack([rates.real, rates.imag])),))

    def _check(self, x, name: str):
        x = jnp.asarray(x)
        if x.ndim == 0 or x.shape[-1] != self.integers.size:
            raise ValueError(f"{name} must have shape (..., {self.integers.size}).")
        return x.astype(_complex_dtype())

    def _check_radial(self, kernel) -> np.ndarray:
        kernel = np.asarray(kernel)
        if kernel.shape != (self.integers.size,):
            raise ValueError(f"A kernel must have shape ({self.integers.size},).")
        valuation = self.integers.valuation(self.integers.points())
        scale = max(1.0, float(np.abs(kernel).max()))
        for v in range(self.integers.n + 1):
            shell = kernel[valuation == v]
            if not np.allclose(shell, shell[0], rtol=0, atol=1e-10 * scale):
                raise ValueError(
                    "The kernel is not radial: Kozyrev wavelets only diagonalise "
                    "kernels that depend on |x|_p alone."
                )
        return kernel
