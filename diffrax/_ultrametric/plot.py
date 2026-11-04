import math

import numpy as np

from .integers import PAdicIntegers
from .wavelets import KozyrevTransform


def _digits(x, p: int, n: int) -> np.ndarray:
    """The first `n` digits of integers `x`, of shape `x.shape + (n,)`."""
    x = np.asarray(x)
    if x.dtype != object:
        x = x.astype(np.int64)
    out = np.empty(x.shape + (n,), dtype=np.int64)
    for k in range(n):
        out[..., k] = np.asarray(x // p**k % p, dtype=np.int64)
    return out


def monna(x, p: int, n: int) -> np.ndarray:
    """The Monna map `sum_k a_k p^k -> sum_k a_k p^{-k-1}` on the first `n` digits.

    It maps `Z_p` onto `[0, 1]`, carries the Haar measure to the length, and sends the
    ball `a + p^k Z_p` to an interval of length `p^{-k}`; on `Z/p^n` it is the
    digit-reversal permutation divided by `p^n`.
    """
    digits = _digits(x, p, n)
    weights = float(p) ** -np.arange(1, n + 1)
    return np.asarray(digits @ weights)


def fractal_ratio(p: int, d: int = 1) -> float:
    """The largest ratio for which `fractal_embedding` keeps disjoint balls apart."""
    s = math.sin(math.pi / p**d)
    return s / (1 + s)


def fractal_embedding(
    x, p: int, n: int, dimension: int = 1, ratio: float | None = None
) -> np.ndarray:
    """Points of `Z_p^d` in the complex plane: `sum_k w^{a_k} ratio^k` with
    `w = exp(2 pi i / p^d)` and `a_k` the `k`-th digit of `x`, read jointly over the
    `d` coordinates (`x` of shape `(..., d)` for `d > 1`).

    Balls become nested clusters, `p^d` copies of the whole shrunk by `ratio` at each
    level: a Sierpinski triangle for `p = 3`. Points in different balls of radius
    `p^{-k}` stay apart by a distance of order `ratio^k`.
    """
    d = int(dimension)
    digits = _digits(x, p, n)
    if d > 1:
        joint = sum(digits[..., j, :] * p**j for j in range(d))
    else:
        joint = digits
    if ratio is None:
        ratio = fractal_ratio(p, d)
    roots = np.exp(2j * np.pi * np.arange(p**d) / p**d)
    return np.asarray((roots[joint] * ratio ** np.arange(n)).sum(axis=-1))


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting needs matplotlib: `pip install matplotlib`."
        ) from error
    return plt


def _axes(ax, ncols: int = 1, size=(5.0, 4.0)):
    plt = _pyplot()
    if ax is not None:
        return ax.figure, np.atleast_1d(ax)
    fig, axes = plt.subplots(
        1, ncols, figsize=(size[0] * ncols, size[1]), squeeze=False
    )
    return fig, axes[0]


def plot_kozyrev_wavelets(integers: PAdicIntegers, embedding: str = "monna", ax=None):
    """One Kozyrev wavelet `psi_{m,0,1}` per level `m`.

    With `embedding="monna"` the real and imaginary parts are drawn as step functions
    on `[0, 1]`: each wavelet lives on an interval of length `p^{-m}` and is constant
    on its `p` subintervals. With `embedding="fractal"` each wavelet is drawn on the
    fractal picture of `Z/p^n`, coloured by its phase, with the points outside its
    ball greyed out. Returns the figure.
    """
    p, n = integers.p, integers.n
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    rows = [int(p**m) for m in range(n)]
    points = np.arange(integers.size)
    if embedding == "monna":
        fig, axes = _axes(ax, size=(7.0, 1.6 * n))
        order = np.argsort(monna(points, p, n))
        position = np.append(np.sort(monna(points, p, n)), 1.0)
        axis = axes[0]
        for k, row in enumerate(rows):
            wave = basis[row][order] / np.abs(basis[row]).max()
            wave = np.append(wave, wave[-1])
            offset = -2.5 * k
            axis.step(position, wave.real + offset, where="post", lw=1.2, color="C0")
            axis.step(position, wave.imag + offset, where="post", lw=1.0, color="C1")
            axis.text(-0.02, offset, f"m = {k}", ha="right", va="center")
        axis.set_xlim(0, 1)
        axis.set_yticks([])
        axis.set_xlabel("Monna coordinate of x")
        axis.set_title(f"Kozyrev wavelets on Z_{p}: real (blue) and imaginary (orange)")
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    fig, axes = _axes(ax, ncols=len(rows), size=(3.0, 3.0))
    z = fractal_embedding(points, p, n)
    for k, (row, axis) in enumerate(zip(rows, axes)):
        wave = basis[row]
        inside = np.abs(wave) > 0
        axis.scatter(z.real[~inside], z.imag[~inside], s=4, color="0.85")
        axis.scatter(
            z.real[inside],
            z.imag[inside],
            s=8,
            c=np.angle(wave[inside]),
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axis.set_title(f"m = {k}")
        axis.set_aspect("equal")
        axis.axis("off")
    fig.suptitle(f"Kozyrev wavelets on Z_{p}, coloured by phase", y=1.04)
    return fig
