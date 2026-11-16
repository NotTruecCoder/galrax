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


def _base(p: int, d: int) -> tuple[int, int]:
    """The number of points per level of the fractal picture, and how many `p`-adic
    levels make one picture level: two for `p^d = 2`, whose two points would lie on a
    line.
    """
    return (4, 2) if p**d == 2 else (p**d, 1)


def fractal_ratio(p: int, d: int = 1) -> float:
    """The largest ratio for which `fractal_embedding` keeps disjoint balls apart."""
    base, _ = _base(p, d)
    s = math.sin(math.pi / base)
    return s / (1 + s)


def fractal_embedding(
    x, p: int, n: int, dimension: int = 1, ratio: float | None = None
) -> np.ndarray:
    """Points of `Z_p^d` in the complex plane: `sum_k w^{a_k} ratio^k` with
    `w = exp(2 pi i / p^d)` and `a_k` the `k`-th digit of `x`, read jointly over the
    `d` coordinates (`x` of shape `(..., d)` for `d > 1`). For `p^d = 2` two levels
    are read as one digit in base `4`, so that the picture is a square fractal rather
    than a segment.

    Balls become nested clusters, `p^d` copies of the whole shrunk by `ratio` at each
    level: a Sierpinski triangle for `p = 3`, a pentagonal one for `p = 5`. Points in
    different balls of radius `p^{-k}` stay apart by a distance of order `ratio^k`.
    """
    d = int(dimension)
    digits = _digits(x, p, n)
    if d > 1:
        joint = sum(digits[..., j, :] * p**j for j in range(d))
    else:
        joint = digits
    base, group = _base(p, d)
    if group == 2:
        if n % 2:
            joint = np.concatenate([joint, np.zeros_like(joint[..., :1])], axis=-1)
        joint = joint[..., 0::2] + 2 * joint[..., 1::2]
    if ratio is None:
        ratio = fractal_ratio(p, d)
    roots = np.exp(2j * np.pi * np.arange(base) / base)
    levels = joint.shape[-1]
    return np.asarray((roots[joint] * ratio ** np.arange(levels)).sum(axis=-1))


def marker_size(count: int, area: float = 25000.0) -> float:
    """A marker size for `count` points that keeps the picture readable."""
    return float(np.clip(area / max(count, 1), 0.5, 80.0))


def monna_grid(axis, p: int, levels: int = 2, x: bool = True, y: bool = True):
    """Lines at the boundaries of the balls of radius `p^{-1}, ..., p^{-levels}` on
    Monna axes, thinner for smaller balls.
    """
    for level in range(1, levels + 1):
        style = {"color": "0.4" if level == 1 else "0.8", "lw": 1.0 / level}
        for j in range(1, p**level):
            if j % p == 0 and level > 1:
                continue
            if x:
                axis.axvline(j / p**level, zorder=0.5, **style)
            if y:
                axis.axhline(j / p**level, zorder=0.5, **style)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Plotting needs matplotlib: `pip install matplotlib`."
        ) from error
    return plt


def _axes(ax, ncols: int = 1, size=(5.0, 4.0), nrows: int = 1):
    plt = _pyplot()
    if ax is not None:
        return ax.figure, np.atleast_1d(ax)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(size[0] * ncols, size[1] * nrows),
        squeeze=False,
        layout="constrained",
    )
    return fig, axes.reshape(-1)


def plot_kozyrev_wavelets(integers: PAdicIntegers, embedding: str = "monna", ax=None):
    """One Kozyrev wavelet `psi_{m,0,1}` per level `m`.

    With `embedding="monna"` the real part is drawn as a filled step function and the
    imaginary part as a line on `[0, 1]`: each wavelet lives on an interval of length
    `p^{-m}` (shaded) and is constant on its `p` subintervals. With
    `embedding="fractal"` each wavelet is drawn on the fractal picture of `Z/p^n`,
    coloured by its phase, with the points outside its ball greyed out. Returns the
    figure.
    """
    p, n = integers.p, integers.n
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    rows = [int(p**m) for m in range(n)]
    points = np.arange(integers.size)
    if embedding == "monna":
        fig, axes = _axes(ax, size=(7.5, 1.5 * n + 0.6))
        order = np.argsort(monna(points, p, n))
        position = np.append(np.sort(monna(points, p, n)), 1.0)
        axis = axes[0]
        for k, row in enumerate(rows):
            wave = basis[row][order] / np.abs(basis[row]).max()
            wave = np.append(wave, wave[-1])
            offset = -2.6 * k
            axis.axvspan(0, p**-k, ymin=0, ymax=1, color="0.94", zorder=0)
            axis.fill_between(
                position,
                offset,
                wave.real + offset,
                step="post",
                color="C0",
                alpha=0.35,
                lw=0,
            )
            axis.step(position, wave.real + offset, where="post", lw=1.4, color="C0")
            axis.step(position, wave.imag + offset, where="post", lw=1.2, color="C3")
            axis.axhline(offset, color="0.6", lw=0.6)
            axis.text(-0.015, offset, f"m = {k}", ha="right", va="center")
        axis.set_xlim(0, 1)
        axis.set_yticks([])
        axis.set_xlabel("Monna coordinate of x")
        axis.set_title(
            f"Kozyrev wavelets on Z_{p}: real part (blue, filled), imaginary part (red)"
        )
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    fig, axes = _axes(ax, ncols=len(rows), size=(3.2, 3.4))
    z = fractal_embedding(points, p, n)
    size = marker_size(len(points), 6000.0)
    for k, (row, axis) in enumerate(zip(rows, axes)):
        wave = basis[row]
        inside = np.abs(wave) > 0
        axis.scatter(z.real[~inside], z.imag[~inside], s=size, color="0.85", lw=0)
        axis.scatter(
            z.real[inside],
            z.imag[inside],
            s=1.6 * size,
            c=np.angle(wave[inside]),
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            lw=0,
        )
        axis.set_title(f"m = {k}: support of radius {p}^-{k}")
        axis.set_aspect("equal")
        axis.axis("off")
    fig.suptitle(f"Kozyrev wavelets on Z_{p}, coloured by phase")
    return fig
