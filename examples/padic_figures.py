"""Figures of p-adic flows and Kozyrev wavelets for p = 2, 3, 5, 7, written to
`imgs/padic/`.
"""

import pathlib

import jax
import matplotlib


matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from diffrax import padic_flow as pf, ultrametric as um  # noqa: E402


out = pathlib.Path(__file__).resolve().parents[1] / "imgs" / "padic"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.png"):
    old.unlink()


def save(fig, name):
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def starts(flow, count=3):
    """Starting points with different orbit scales k = v_p(f(x))."""
    p = flow.p
    candidates = np.arange(p**3)
    levels = np.asarray(flow.orbit_level(candidates))
    chosen = {}
    for x, k in zip(candidates.tolist(), levels.tolist()):
        if k < flow.precision and k not in chosen:
            chosen[k] = x
    return [chosen[k] for k in sorted(chosen)][:count]


line_levels = {2: 8, 3: 6, 5: 4, 7: 3}
plane_levels = {2: 6, 3: 4, 5: 3, 7: 2}
overview, panels = plt.subplots(1, 4, figsize=(16, 4.4), layout="constrained")
for p, panel in zip((2, 3, 5, 7), panels):
    c = 4 if p == 2 else p
    # dx/dt = c - c x + 2c x^3 on Z_p, with c = p (c = 4 for p = 2).
    line = pf.PAdicFlow(p, line_levels[p] + 2, [c, -c, 0, 2 * c])
    points = starts(line)
    save(pf.plot_trajectory(line, points, line_levels[p]), f"p{p}_trajectory.png")
    save(
        pf.plot_trajectory(line, points[:1], line_levels[p], embedding="fractal"),
        f"p{p}_trajectory_fractal.png",
    )
    pf.plot_trajectory(line, points[:1], line_levels[p], embedding="fractal", ax=panel)
    panel.set_title(f"p = {p}: the orbit of {points[0]}")
    # dx/dt = c y, dy/dt = -c x + c x^2 on Z_p^2, with equilibria (0, 0) and (1, 0).
    plane = pf.PAdicFlow(p, plane_levels[p] + 2, [{(0, 1): c}, {(1, 0): -c, (2, 0): c}])
    level = plane_levels[p]
    save(pf.plot_phase_portrait(plane, level), f"p{p}_phase.png")
    save(
        pf.plot_phase_portrait(plane, level, embedding="fractal"),
        f"p{p}_phase_fractal.png",
    )
    save(pf.plot_phase_portrait(plane, level, colour="speed"), f"p{p}_speed.png")
overview.suptitle("Orbits of dx/dt = c - cx + 2cx^3 on the fractal pictures of Z_p")
save(overview, "fractal_orbits.png")

for p, n in ((3, 4), (5, 3)):
    integers = um.PAdicIntegers(p, n)
    save(um.plot_kozyrev_wavelets(integers), f"p{p}_kozyrev.png")
    save(
        um.plot_kozyrev_wavelets(integers, embedding="fractal"),
        f"p{p}_kozyrev_fractal.png",
    )

for p, n in ((3, 6), (5, 4)):
    # Vladimirov diffusion with jumps along the flow of dx/dt = p - p x + 2p x^3.
    line = pf.PAdicFlow(p, n + 2, [p, -p, 0, 2 * p])
    integers = um.PAdicIntegers(p, n)
    operator = um.UltrametricOperator.vladimirov(integers, 0.5)
    koopman = pf.KoopmanOperator(line, n)
    start = starts(line)[-1]
    u0 = np.zeros(integers.size)
    u0[integers.ball(start, n - 2)] = 1.0
    times = np.linspace(0.0, 3.0, 150)
    save(
        pf.plot_advection_diffusion(koopman, operator, u0, times, rate=2.0),
        f"p{p}_advection.png",
    )
    save(
        pf.plot_advection_diffusion(
            koopman, operator, u0, [0.0, 0.3, 1.0, 3.0], rate=2.0, embedding="fractal"
        ),
        f"p{p}_advection_fractal.png",
    )
print("written to", out)
