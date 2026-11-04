import numpy as np

from .._ultrametric.plot import (
    _axes,
    fractal_embedding,
    marker_size,
    monna,
    monna_grid,
)
from .koopman import advection_diffusion


def _leaves(flow, points, level: int):
    """The leaf of the flow-box foliation through each point of `(Z/p^level)^2`, as a
    number in `[0, 1)`, and the scale `k = v_p(f(x))` of its orbit.

    A point `z` with `k < level` lies in the ball of radius `p^{-k}` around its low
    digits `c = z mod p^k`, on which `v_p(f) = k`. Around `c` it has flow-box
    coordinates `z = phi_s(w)` with `w` on the transversal through `c`; the leaf is the
    Monna coordinate of the free coordinate of `w`, read from digit `k` on. It is the
    same along each orbit, and close for orbits that pass close to each other.
    """
    p = flow.p
    speed = np.minimum(np.asarray(flow.orbit_level(points)), level)
    moving = speed < level
    scale = p ** speed[:, None]
    base = points % scale
    leaf = np.zeros(len(points))
    if moving.any():
        _, foot = flow.rectify(base[moving], points[moving])
        foot = np.asarray(foot).astype(np.int64)
        field = np.asarray(flow.vector_field(base[moving])).astype(np.int64)
        valuations = np.zeros(field.shape, dtype=np.int64)
        for k in range(1, flow.precision + 1):
            valuations += field % p**k == 0
        free = 1 - np.argmin(valuations, axis=1)
        rows = np.arange(len(foot))
        shift = foot[rows, free] // p ** speed[moving]
        digits = np.maximum(level - speed[moving], 1)
        leaf[moving] = [
            monna(int(v), p, int(n)) for v, n in zip(shift.tolist(), digits.tolist())
        ]
    return leaf, speed


def _orbit(flow, start, level: int):
    """The orbit of `start` under `phi_1` on `(Z/p^level)^d`, in time order: the
    `p^{level - k}` points `phi_t(start)`, `0 <= t < p^{level - k}`.
    """
    p = flow.p
    k = min(int(flow.orbit_level(start)), level)
    times = np.arange(p ** (level - k))
    points = np.asarray(flow.flow(times, start)).astype(np.int64) % p**level
    return points, k


def _starts(flow, level: int, count: int):
    """Starting points of up to `count` orbits that are not equilibria modulo
    `p^level`: one per ball of radius `1/p` and orbit scale first, then any.
    """
    p, d = flow.p, flow.dimension
    points = np.indices((p**level,) * d).reshape(d, -1).T
    points = points[:, 0] if d == 1 else points
    speed = np.minimum(np.asarray(flow.orbit_level(points)), level)
    order = np.random.default_rng(1).permutation(len(points))
    moving = [i for i in order if speed[i] < level]
    seen, first, later = set(), [], []
    for i in moving:
        key = (tuple(np.atleast_1d(points[i] % p)), int(speed[i]))
        (later if key in seen else first).append(i)
        seen.add(key)
    return [points[i] for i in (first + later)[:count]]


def plot_trajectory(flow, x, level: int, embedding: str = "monna", ax=None):
    """The orbit `t -> phi_t(x)` over the integer times `0 <= t < p^level`, read
    modulo `p^level`.

    With `embedding="monna"` (`d = 1`) it is the graph of `t -> phi_t(x)` with both
    axes in Monna coordinates, over a grid of the balls of radius `1/p` and `1/p^2`.
    Since `t -> phi_t(x)` multiplies distances by exactly `|f(x)|_p`, the graph is a
    self-similar curve filling the shaded band of height `p^{-k}`, `k = v_p(f(x))`,
    above the whole time axis. With `embedding="fractal"` the orbits are drawn on the
    fractal picture of `(Z/p^level)^d`, coloured by the Monna coordinate of the time,
    next to a close-up of the first one.
    `x` may hold several starting points. Returns the figure.
    """
    p, d = flow.p, flow.dimension
    times = np.arange(p**level)
    starts = np.asarray(x)
    starts = starts.reshape((-1,) if d == 1 else (-1, d))
    if embedding == "monna":
        if d != 1:
            raise ValueError("The Monna graph needs d = 1; use embedding='fractal'.")
        fig, axes = _axes(ax, size=(6.0, 6.6))
        axis = axes[0]
        monna_grid(axis, p, min(2, level))
        horizontal = monna(times, p, level)
        size = marker_size(len(times), 5000.0)
        for k, start in enumerate(starts):
            colour = f"C{k % 10}"
            points = np.asarray(flow.flow(times, int(start)))
            radius = min(int(flow.orbit_level(int(start))), level)
            low = float(monna(int(start) % p**radius, p, level))
            axis.axhspan(low, low + float(p) ** -radius, color=colour, alpha=0.12)
            axis.scatter(
                horizontal,
                monna(points, p, level),
                s=size,
                color=colour,
                lw=0,
                label=f"x = {int(start)}: band of height |f(x)| = {p}^-{radius}",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("Monna coordinate of t")
        axis.set_ylabel("Monna coordinate of phi_t(x)")
        axis.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, markerscale=3
        )
        axis.set_title(f"t -> phi_t(x) on Z_{p}, modulo {p}^{level}")
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    fig, axes = _axes(ax, ncols=2, size=(4.8, 5.0))
    every = np.indices((p**level,) * d).reshape(d, -1).T
    every = every[:, 0] if d == 1 else every
    background = fractal_embedding(every, p, level, d)
    colour = monna(times, p, level)
    maps = ["viridis", "plasma", "cividis", "cool"]
    whole, close = axes[0], axes[-1]
    size = marker_size(len(every), 9000.0)
    whole.scatter(background.real, background.imag, s=size, color="0.82", lw=0)
    first = None
    for k, start in enumerate(starts):
        points = np.asarray(flow.flow(times, start if d > 1 else int(start)))
        z = fractal_embedding(points, p, level, d)
        whole.scatter(z.real, z.imag, s=2.5 * size, c=colour, cmap=maps[k % 4], lw=0)
        first = z if first is None else first
    plane = "^2" if d == 2 else ""
    whole.set_title(f"Orbits on the fractal picture of Z_{p}{plane}")
    if close is not whole:
        span = max(np.ptp(first.real), np.ptp(first.imag), 1e-9)
        centre = complex(np.mean(first.real), np.mean(first.imag))
        near = (np.abs(background.real - centre.real) < 0.7 * span) & (
            np.abs(background.imag - centre.imag) < 0.7 * span
        )
        close.scatter(background.real[near], background.imag[near], s=10, color="0.85")
        image = close.scatter(
            first.real, first.imag, s=30, c=colour, cmap=maps[0], lw=0
        )
        close.set_xlim(centre.real - 0.7 * span, centre.real + 0.7 * span)
        close.set_ylim(centre.imag - 0.7 * span, centre.imag + 0.7 * span)
        close.set_title("Close-up of the first orbit")
        fig.colorbar(image, ax=close, shrink=0.7, label="Monna coordinate of t")
    for axis in axes:
        axis.set_aspect("equal")
        axis.axis("off")
    return fig


def plot_phase_portrait(
    flow,
    level: int,
    embedding: str = "monna",
    colour: str = "orbits",
    orbits=None,
    count: int = 8,
    ax=None,
):
    """The phase portrait of a flow on `Z_p^2`, read modulo `p^level`.

    - `colour="orbits"`: a few whole orbits of `phi_1`, one colour each, drawn over
        the scale `k = v_p(f(x))` of the orbits in light shades; equilibria are
        black. `orbits` gives their starting points (by default `count` of them,
        spread over the balls of radius `1/p`).
    - `colour="leaves"`: every point coloured by the leaf of the flow-box foliation
        through it (see `_leaves`), so that orbits are the sets of one colour.
    - `colour="speed"`: every point coloured by `k = v_p(f(x))`.

    With `embedding="monna"` the plane is the unit square in Monna coordinates, so
    balls are squares, outlined at radii `1/p` and `1/p^2`; with
    `embedding="fractal"` it is the fractal picture of `Z_p^2`. Returns the figure.
    """
    import matplotlib

    p = flow.p
    if flow.dimension != 2:
        raise ValueError("The phase portrait needs d = 2.")
    if colour not in ("orbits", "leaves", "speed"):
        raise ValueError("`colour` must be 'orbits', 'leaves' or 'speed'.")
    if embedding not in ("monna", "fractal"):
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    side = p**level
    points = np.indices((side, side)).reshape(2, -1).T
    if colour == "leaves":
        leaf, speed = _leaves(flow, points, level)
    else:
        speed = np.minimum(np.asarray(flow.orbit_level(points)), level)
    fixed = speed >= level
    if colour == "leaves":
        colours = matplotlib.colormaps["hsv"](0.9 * leaf)
    elif colour == "speed":
        colours = matplotlib.colormaps["viridis"](speed / max(level, 1))
    else:
        colours = matplotlib.colormaps["Blues"](0.12 + 0.45 * speed / max(level, 1))
    colours[fixed] = (0.0, 0.0, 0.0, 1.0)
    fig, axes = _axes(ax, size=(6.2, 6.8))
    axis = axes[0]
    if embedding == "monna":
        column = np.rint(monna(points[:, 0], p, level) * side).astype(np.int64)
        row = np.rint(monna(points[:, 1], p, level) * side).astype(np.int64)
        image = np.zeros((side, side, 4))
        image[row, column] = colours
        axis.imshow(image, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
        monna_grid(axis, p, min(2, level))

        def place(values):
            half = 0.5 / side
            return (
                monna(values[:, 0], p, level) + half,
                monna(values[:, 1], p, level) + half,
            )

        axis.set_xlabel("Monna coordinate of x")
        axis.set_ylabel("Monna coordinate of y")
        cell = (300.0 / side) ** 2
    else:
        z = fractal_embedding(points, p, level, 2)
        size = marker_size(len(points), 12000.0)
        axis.scatter(z.real, z.imag, s=size, c=colours, lw=0)

        def place(values):
            w = fractal_embedding(values, p, level, 2)
            return w.real, w.imag

        axis.set_aspect("equal")
        axis.axis("off")
        cell = 3.0 * size
    if colour == "orbits":
        starts = _starts(flow, level, count) if orbits is None else list(orbits)
        for k, start in enumerate(starts):
            orbit, radius = _orbit(flow, np.asarray(start), level)
            u, v = place(orbit)
            label = (
                f"({int(start[0])}, {int(start[1])}): {len(orbit)} points, k = {radius}"
            )
            axis.scatter(
                u,
                v,
                s=float(np.clip(cell, 14, 140)),
                color=f"C{k % 10}",
                edgecolors="white",
                linewidths=0.3,
                label=label,
                zorder=3,
            )
        if fixed.any() and embedding == "fractal":
            u, v = place(points[fixed])
            axis.scatter(u, v, marker="x", s=25, color="black", lw=1.0, zorder=4)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1 if embedding == "monna" else -0.02),
            ncol=2,
            fontsize=8,
            title="orbits through",
            title_fontsize=8,
        )
    names = {
        "orbits": "orbits of phi_1",
        "leaves": "flow-box leaves",
        "speed": "v_p(f)",
    }
    axis.set_title(f"Phase portrait on Z_{p}^2 modulo {p}^{level}: {names[colour]}")
    if colour == "speed":
        mappable = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(0, level), cmap="viridis"
        )
        fig.colorbar(mappable, ax=axis, shrink=0.7, label="k = v_p(f(x))")
    return fig


def plot_advection_diffusion(
    koopman, operator, u0, times, rate: float = 1.0, embedding: str = "monna", ax=None
):
    """The solution of `du/dt = L u + rate (U - I) u` at the given times.

    With `embedding="monna"` it is a heat map, time upwards and `x` in Monna
    coordinates across, so that diffusion blurs whole intervals (balls) while the jumps
    along the flow permute them. With `embedding="fractal"` there is one fractal
    picture per time. Returns the figure.
    """
    import matplotlib

    integers = operator.integers
    p, n = integers.p, integers.n
    times = np.asarray(times, dtype=float)
    values = np.stack(
        [
            np.real(np.asarray(advection_diffusion(koopman, operator, u0, t, rate)))
            for t in times
        ]
    )
    points = np.arange(integers.size)
    if embedding == "monna":
        fig, axes = _axes(ax, size=(7.5, 4.2))
        axis = axes[0]
        order = np.argsort(monna(points, p, n))
        floor = max(values.max() * 1e-4, np.finfo(float).tiny)
        image = axis.imshow(
            np.maximum(values[:, order], floor),
            origin="lower",
            aspect="auto",
            extent=(0, 1, times[0], times[-1]),
            cmap="inferno",
            norm=matplotlib.colors.LogNorm(vmin=floor, vmax=values.max()),
        )
        for j in range(1, p):
            axis.axvline(j / p, color="white", lw=0.8, alpha=0.7)
        for j in range(1, p**2):
            if j % p:
                axis.axvline(j / p**2, color="white", lw=0.3, alpha=0.5)
        fig.colorbar(image, ax=axis, label="u(t, x)")
        axis.set_xlabel("Monna coordinate of x")
        axis.set_ylabel("t")
        axis.set_title(f"Ultrametric diffusion with jumps along the flow on Z_{p}")
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    fig, axes = _axes(ax, ncols=len(times), size=(3.2, 3.4))
    z = fractal_embedding(points, p, n)
    floor = max(values.max() * 1e-4, np.finfo(float).tiny)
    norm = matplotlib.colors.LogNorm(vmin=floor, vmax=values.max())
    size = marker_size(len(points), 7000.0)
    for t, row, axis in zip(times, values, axes):
        image = axis.scatter(
            z.real,
            z.imag,
            s=size,
            c=np.maximum(row, floor),
            cmap="inferno",
            norm=norm,
            lw=0,
        )
        axis.set_title(f"t = {t:g}")
        axis.set_aspect("equal")
        axis.axis("off")
    fig.colorbar(image, ax=list(axes), shrink=0.7, label="u(t, x)")
    return fig
