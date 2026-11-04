"""Figures of p-adic flows and Kozyrev wavelets, written to `imgs/padic/`."""

import pathlib

import jax
import matplotlib


matplotlib.use("Agg")
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from diffrax import padic_flow as pf, ultrametric as um  # noqa: E402


out = pathlib.Path(__file__).resolve().parents[1] / "imgs" / "padic"
out.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    matplotlib.pyplot.close(fig)


# dx/dt = 3 - 3x + 6x^3 on Z_3: the orbit of 0 is a ball of radius 1/3, that of 2 a
# ball of radius 1/9.
line = pf.PAdicFlow(3, 8, [3, -3, 0, 6])
save(pf.plot_trajectory(line, [0, 2], level=6), "trajectory_monna.png")
save(
    pf.plot_trajectory(line, [2], level=6, embedding="fractal"),
    "trajectory_fractal.png",
)

# dx/dt = 3y, dy/dt = -3x + 3x^2 on Z_3^2, with equilibria (0, 0) and (1, 0).
plane = pf.PAdicFlow(3, 8, [{(0, 1): 3}, {(1, 0): -3, (2, 0): 3}])
save(pf.plot_phase_portrait(plane, level=4), "phase_monna.png")
save(pf.plot_phase_portrait(plane, level=4, embedding="fractal"), "phase_fractal.png")
save(pf.plot_phase_portrait(plane, level=4, colour="speed"), "phase_speed.png")

# Vladimirov diffusion with jumps along the flow of the first example.
integers = um.PAdicIntegers(3, 6)
operator = um.UltrametricOperator.vladimirov(integers, 0.5)
koopman = pf.KoopmanOperator(line, 6)
u0 = np.zeros(integers.size)
u0[integers.ball(2, 4)] = 1.0
times = np.linspace(0.0, 3.0, 150)
save(
    pf.plot_advection_diffusion(koopman, operator, u0, times, rate=2.0),
    "advection_monna.png",
)
save(
    pf.plot_advection_diffusion(
        koopman, operator, u0, [0.0, 0.3, 1.0, 3.0], rate=2.0, embedding="fractal"
    ),
    "advection_fractal.png",
)

wavelets = um.PAdicIntegers(3, 4)
save(um.plot_kozyrev_wavelets(wavelets), "kozyrev_monna.png")
save(um.plot_kozyrev_wavelets(wavelets, embedding="fractal"), "kozyrev_fractal.png")
print("written to", out)
