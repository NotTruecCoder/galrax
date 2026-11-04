import jax
import numpy as np
import pytest
from diffrax import padic_flow as pf, ultrametric as um


matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_monna_map():
    p, n = 3, 5
    x = np.arange(p**n)
    values = um.monna(x, p, n)
    assert np.allclose(np.sort(values), np.arange(p**n) / p**n)
    ball = x[x % 9 == 2]
    low = um.monna(2, p, n)
    assert np.all(
        (values[x % 9 == 2] >= low - 1e-12) & (values[x % 9 == 2] < low + 1 / 9)
    )
    assert len(ball) == p ** (n - 2)
    big = np.array([3**40 + 1], dtype=object)
    assert np.isclose(um.monna(big, p, 3)[0], 1 / 3)


@pytest.mark.parametrize("p,d", [(2, 1), (3, 1), (5, 1), (3, 2)])
def test_fractal_embedding_separates_balls(p, d):
    n = 4
    points = np.indices((p**n,) * d).reshape(d, -1).T
    points = points[:, 0] if d == 1 else points
    z = um.fractal_embedding(points, p, n, d)
    top = points % p if d == 1 else (points % p) @ (p ** np.arange(d))
    inside = max(
        np.abs(z[top == a][:, None] - z[top == a][None, :]).max() for a in range(p**d)
    )
    across = min(
        np.abs(z[top == a][:, None] - z[top == b][None, :]).min()
        for a in range(p**d)
        for b in range(a + 1, p**d)
    )
    assert across > 0
    assert len(np.unique(np.round(z, 12))) == p ** (n * d)
    assert inside > 0


def test_plots_draw():
    jax.config.update("jax_enable_x64", True)
    line = pf.PAdicFlow(3, 6, [3, -3, 0, 6])
    plane = pf.PAdicFlow(3, 4, [{(0, 1): 3}, {(1, 0): -3, (2, 0): 3}])
    figures = [
        pf.plot_trajectory(line, [0, 2], level=3),
        pf.plot_trajectory(line, 2, level=3, embedding="fractal"),
        pf.plot_trajectory(plane, [[1, 2]], level=2, embedding="fractal"),
        pf.plot_phase_portrait(plane, level=2),
        pf.plot_phase_portrait(plane, level=2, embedding="fractal"),
        pf.plot_phase_portrait(plane, level=2, colour="speed"),
        um.plot_kozyrev_wavelets(um.PAdicIntegers(3, 3)),
        um.plot_kozyrev_wavelets(um.PAdicIntegers(2, 3), embedding="fractal"),
    ]
    integers = um.PAdicIntegers(3, 3)
    operator = um.UltrametricOperator.vladimirov(integers, 1.0)
    koopman = pf.KoopmanOperator(line, 3)
    u0 = np.zeros(integers.size)
    u0[2] = 1.0
    figures.append(pf.plot_advection_diffusion(koopman, operator, u0, [0.0, 0.5, 1.0]))
    figures.append(
        pf.plot_advection_diffusion(
            koopman, operator, u0, [0.0, 1.0], embedding="fractal"
        )
    )
    for fig in figures:
        assert fig.axes
        matplotlib.pyplot.close(fig)
    with pytest.raises(ValueError):
        pf.plot_trajectory(plane, [[1, 2]], level=2)
    with pytest.raises(ValueError):
        pf.plot_phase_portrait(line, level=2)
    with pytest.raises(ValueError):
        pf.plot_phase_portrait(plane, level=2, colour="random")
