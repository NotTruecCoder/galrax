import numpy as np

from .._ultrametric.plot import _axes, fractal_embedding, monna
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


def plot_trajectory(flow, x, level: int, embedding: str = "monna", ax=None):
    """The orbit `t -> phi_t(x)` over the integer times `0 <= t < p^level`, read
    modulo `p^level`.

    With `embedding="monna"` (`d = 1`) it is the graph of `t -> phi_t(x)` with both
    axes in Monna coordinates: since `t -> phi_t(x)` multiplies distances by exactly
    `|f(x)|_p`, the graph is a self-similar curve filling a square of side `p^{-k}`,
    `k = v_p(f(x))`, above the whole time axis. With `embedding="fractal"` the orbit is
    drawn on the fractal picture of `(Z/p^level)^d`, coloured by the Monna coordinate
    of the time. `x` may hold several starting points. Returns the figure.
    """
    p, d = flow.p, flow.dimension
    times = np.arange(p**level)
    starts = np.asarray(x)
    starts = starts.reshape((-1,) if d == 1 else (-1, d))
    fig, axes = _axes(ax)
    axis = axes[0]
    if embedding == "monna":
        if d != 1:
            raise ValueError("The Monna graph needs d = 1; use embedding='fractal'.")
        horizontal = monna(times, p, level)
        for k, start in enumerate(starts):
            points = np.asarray(flow.flow(times, int(start)))
            radius = int(flow.orbit_level(int(start)))
            axis.scatter(
                horizontal,
                monna(points, p, level),
                s=2,
                color=f"C{k % 10}",
                label=f"x = {int(start)}, k = {min(radius, level)}",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal")
        axis.set_xlabel("Monna coordinate of t")
        axis.set_ylabel("Monna coordinate of phi_t(x)")
        axis.legend(loc="upper right", fontsize=8, markerscale=4)
        axis.set_title(f"Orbits of a flow on Z_{p}")
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    every = np.indices((p**level,) * d).reshape(d, -1).T
    every = every[:, 0] if d == 1 else every
    background = fractal_embedding(every, p, level, d)
    axis.scatter(background.real, background.imag, s=2, color="0.88")
    colour = monna(times, p, level)
    for start in starts:
        points = np.asarray(flow.flow(times, start if d > 1 else int(start)))
        z = fractal_embedding(points, p, level, d)
        axis.scatter(z.real, z.imag, s=6, c=colour, cmap="viridis")
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("Orbit coloured by the Monna coordinate of t")
    return fig


def plot_phase_portrait(
    flow, level: int, embedding: str = "monna", colour: str = "orbit", ax=None
):
    """The phase portrait of a flow on `Z_p^2`, read modulo `p^level`.

    With `colour="orbit"` every point of `(Z/p^level)^2` is coloured by the leaf of
    the flow-box foliation through it (see `_leaves`): orbits are the sets of one
    colour, and nearby orbits get nearby colours; equilibria are black. With
    `colour="speed"` the colour is `k = v_p(f(x))`, the scale of the orbit. With
    `embedding="monna"` the plane is the unit square in Monna coordinates, so balls
    are squares; with `embedding="fractal"` it is the fractal picture of `Z_p^2`.
    Returns the figure.
    """
    import matplotlib

    p = flow.p
    if flow.dimension != 2:
        raise ValueError("The phase portrait needs d = 2.")
    side = p**level
    points = np.indices((side, side)).reshape(2, -1).T
    leaf, speed = _leaves(flow, points, level)
    if colour == "orbit":
        colours = matplotlib.colormaps["hsv"](0.9 * leaf)
        colours[speed >= level] = (0.0, 0.0, 0.0, 1.0)
    elif colour == "speed":
        colours = matplotlib.colormaps["magma"](speed / max(level, 1))
    else:
        raise ValueError("`colour` must be 'orbit' or 'speed'.")
    fig, axes = _axes(ax, size=(5.0, 5.0))
    axis = axes[0]
    if embedding == "monna":
        column = np.rint(monna(points[:, 0], p, level) * side).astype(np.int64)
        row = np.rint(monna(points[:, 1], p, level) * side).astype(np.int64)
        image = np.zeros((side, side, 4))
        image[row, column] = colours
        axis.imshow(image, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
        axis.set_xlabel("Monna coordinate of x")
        axis.set_ylabel("Monna coordinate of y")
    elif embedding == "fractal":
        z = fractal_embedding(points, p, level, 2)
        axis.scatter(z.real, z.imag, s=5, c=colours)
        axis.set_aspect("equal")
        axis.axis("off")
    else:
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    what = "flow-box leaves" if colour == "orbit" else "v_p(f(x))"
    axis.set_title(f"Phase portrait on Z_{p}^2 mod {p}^{level}: {what}")
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
        fig, axes = _axes(ax, size=(7.0, 4.0))
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
        fig.colorbar(image, ax=axis, label="u(t, x)")
        axis.set_xlabel("Monna coordinate of x")
        axis.set_ylabel("t")
        axis.set_title("Ultrametric diffusion with jumps along the flow")
        return fig
    if embedding != "fractal":
        raise ValueError("`embedding` must be 'monna' or 'fractal'.")
    fig, axes = _axes(ax, ncols=len(times), size=(3.0, 3.0))
    z = fractal_embedding(points, p, n)
    floor = max(values.max() * 1e-4, np.finfo(float).tiny)
    norm = matplotlib.colors.LogNorm(vmin=floor, vmax=values.max())
    for t, row, axis in zip(times, values, axes):
        axis.scatter(
            z.real, z.imag, s=6, c=np.maximum(row, floor), cmap="inferno", norm=norm
        )
        axis.set_title(f"t = {t:g}")
        axis.set_aspect("equal")
        axis.axis("off")
    return fig
