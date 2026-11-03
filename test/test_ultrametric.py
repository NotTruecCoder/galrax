import itertools

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax.group_algebra import (
    convolution,
    exact_linear_solution,
    FourierTransform,
    GroupODE,
    Lawson,
    solve_group_ode,
)
from diffrax.ultrametric import KozyrevTransform, PAdicIntegers, UltrametricOperator


SIZES = [(2, 1), (2, 4), (3, 3), (5, 2), (7, 1)]
SMALL = [(2, 3), (3, 2), (5, 2)]


def _ids(sizes):
    return [f"Z{p}^{n}" for p, n in sizes]


def _random(integers, *batch, seed=0):
    return np.random.default_rng(seed).standard_normal(batch + (integers.size,))


def _direct(operator, f):
    integers = operator.integers
    x = integers.points()
    distance = integers.distance(x[:, None], x[None, :])
    valuation = integers.valuation(x[:, None] - x[None, :])
    rate = np.where(
        distance > 0, operator.weights[np.minimum(valuation, integers.n - 1)], 0
    )
    jumps = (rate * (f[None, :] - f[:, None])).sum(axis=1) / integers.size
    return jumps - operator.absorption * f


def _operators(integers):
    rng = np.random.default_rng(1)
    return [
        UltrametricOperator.vladimirov(integers, 0.8),
        UltrametricOperator.vladimirov(integers, 1.5, absorbing=False),
        UltrametricOperator(integers, rng.uniform(0, 3, integers.n), absorption=0.4),
        UltrametricOperator.from_function(integers, lambda r: np.exp(-r) / r),
    ]


#
# p-adic integers
#


def test_rejects_bad_parameters():
    for p in (0, 1, 4, 9):
        with pytest.raises(ValueError):
            PAdicIntegers(p, 2)
    with pytest.raises(ValueError):
        PAdicIntegers(3, 0)


def test_digits_and_valuation():
    integers = PAdicIntegers(3, 4)
    x = integers.points()
    digits = integers.digits(x)
    assert np.array_equal(integers.from_digits(digits), x)
    assert np.array_equal(integers.digits(5), [2, 1, 0, 0])
    assert np.array_equal(integers.digits(-1), [2, 2, 2, 2])
    trailing = np.argmax(digits != 0, axis=1)
    expected = np.where(x == 0, integers.n, trailing)
    assert np.array_equal(integers.valuation(x), expected)
    assert integers.norm(0) == 0
    assert integers.norm(18) == pytest.approx(1 / 9)
    with pytest.raises(ValueError):
        integers.from_digits([[0, 1, 3, 0]])
    with pytest.raises(ValueError):
        integers.valuation(np.array([0.5]))


@pytest.mark.parametrize("p,n", SMALL, ids=_ids(SMALL))
def test_strong_triangle_inequality(p, n):
    integers = PAdicIntegers(p, n)
    x, y, z = np.meshgrid(*[integers.points()] * 3, indexing="ij")
    d_xz = integers.distance(x, z)
    bound = np.maximum(integers.distance(x, y), integers.distance(y, z))
    assert (d_xz <= bound).all()
    unequal = integers.distance(x, y) != integers.distance(y, z)
    assert np.array_equal(d_xz[unequal], bound[unequal])


def test_fractions():
    integers = PAdicIntegers(2, 8)
    third = integers.from_fraction(1, 3)
    assert 3 * third % integers.size == 1
    assert integers.from_fraction(-1) == integers.size - 1
    assert integers.from_fraction(7, 5) * 5 % integers.size == 7
    with pytest.raises(ValueError):
        integers.from_fraction(1, 2)


def test_balls_and_haar_measure():
    integers = PAdicIntegers(3, 3)
    for level in range(integers.n + 1):
        for center in range(integers.p**level):
            ball = integers.ball(center, level)
            assert ball.sum() == integers.p ** (integers.n - level)
            assert float(integers.integrate(ball.astype(float))) == pytest.approx(
                3.0**-level
            )
            inside = integers.points()[ball]
            assert (integers.distance(inside, center) <= 3.0**-level).all()
    with pytest.raises(ValueError):
        integers.ball(0, 4)
    assert float(integers.integrate(np.ones(integers.size))) == pytest.approx(1.0)


#
# Kozyrev wavelets
#


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_wavelets_are_orthonormal(p, n):
    integers = PAdicIntegers(p, n)
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    gram = basis @ basis.conj().T / integers.size
    assert np.allclose(gram, np.eye(integers.size), atol=1e-12)
    assert np.allclose(basis[0], 1)


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_wavelet_shapes(p, n):
    integers = PAdicIntegers(p, n)
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    for m in range(n):
        assert (transform.levels == m).sum() == p**m * (p - 1)
    x = integers.points()
    for k in range(1, integers.size):
        m, a, j = transform.levels[k], transform.centers[k], transform.frequencies[k]
        ball = integers.ball(a, m)
        assert np.allclose(basis[k][~ball], 0)
        assert abs(basis[k][ball].sum()) < 1e-10
        digit = integers.digits(x[ball])[:, m]
        expected = p ** (m / 2) * np.exp(2j * np.pi * j * digit / p)
        assert np.allclose(basis[k][ball], expected)


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_transform_round_trip(p, n):
    integers = PAdicIntegers(p, n)
    transform = KozyrevTransform(integers)
    f = _random(integers, 2, 3) + 1j * _random(integers, 2, 3, seed=1)
    c = transform.forward(f)
    assert c.shape == f.shape
    expected = f @ transform.basis().conj().T / integers.size
    assert np.allclose(c, expected, atol=1e-12)
    assert np.allclose(transform.inverse(c), f, atol=1e-12)
    (y,) = transform.to_coordinates(f)
    assert y.shape == (2,) + f.shape
    assert np.allclose(np.sum(y**2), np.sum(np.abs(f) ** 2))
    assert np.allclose(transform.from_coordinates((y,)), f, atol=1e-12)


def test_transform_rejects_bad_shapes():
    transform = KozyrevTransform(PAdicIntegers(3, 2))
    with pytest.raises(ValueError):
        transform.forward(np.zeros(8))
    with pytest.raises(ValueError):
        transform.inverse(1.0)
    with pytest.raises(TypeError):
        KozyrevTransform(9)


def test_spectrum_needs_a_radial_kernel():
    integers = PAdicIntegers(3, 2)
    transform = KozyrevTransform(integers)
    kernel = np.zeros(integers.size)
    kernel[1] = 1.0
    with pytest.raises(ValueError, match="radial"):
        transform.spectrum(kernel)
    problem = GroupODE(integers.group, np.ones(9), 0.0, 1.0, right_kernel=kernel)
    with pytest.raises(ValueError, match="radial"):
        exact_linear_solution(problem, [1.0], transform=transform)


#
# Operators
#


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_operator_matches_its_definition(p, n):
    integers = PAdicIntegers(p, n)
    f = _random(integers)
    for operator in _operators(integers):
        expected = _direct(operator, f)
        assert np.allclose(operator(f), expected, atol=1e-10)
        assert np.allclose(convolution(integers.group, f, operator.kernel), expected)
        assert operator.kernel.sum() == pytest.approx(-operator.absorption)
        batch = _random(integers, 2, 3)
        assert np.allclose(operator(batch)[1, 2], operator(batch[1, 2]))


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_wavelets_are_eigenfunctions(p, n):
    integers = PAdicIntegers(p, n)
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    for operator in _operators(integers):
        eigenvalues = operator.eigenvalues[transform.levels + 1]
        assert np.allclose(operator(basis), eigenvalues[:, None] * basis, atol=1e-9)
        assert np.allclose(
            transform.level_eigenvalues(operator.kernel), operator.eigenvalues
        )
        assert np.allclose(transform.spectrum(operator.kernel), eigenvalues)


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_characters_see_the_same_spectrum(p, n):
    integers = PAdicIntegers(p, n)
    operator = _operators(integers)[2]
    symbol = np.fft.fft(operator.kernel)
    v = integers.valuation(integers.points())
    level = np.where(v == n, -1, n - 1 - v)
    assert np.allclose(symbol, operator.eigenvalues[level + 1])


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_vladimirov_spectrum(p, n, alpha):
    integers = PAdicIntegers(p, n)
    killed = UltrametricOperator.vladimirov(integers, alpha)
    rate = (1 - 1 / p) / (1 - p ** (-alpha - 1))
    expected = [rate] + [p ** (alpha * (m + 1)) for m in range(n)]
    assert np.allclose(-killed.eigenvalues, expected)
    free = UltrametricOperator.vladimirov(integers, alpha, absorbing=False)
    assert np.allclose(-free.eigenvalues, np.array(expected) - rate)


def test_operator_validation():
    integers = PAdicIntegers(2, 3)
    with pytest.raises(ValueError):
        UltrametricOperator(integers, [1.0, 1.0])
    with pytest.raises(ValueError):
        UltrametricOperator(integers, [1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        UltrametricOperator(integers, [1.0, 1.0, 1.0], absorption=-1.0)
    with pytest.raises(ValueError):
        UltrametricOperator.vladimirov(integers, 0.0)
    with pytest.raises(TypeError):
        UltrametricOperator(None, [1.0])
    with pytest.raises(ValueError):
        UltrametricOperator.vladimirov(integers, 1.0)(np.ones(7))


#
# Dynamics
#


@pytest.mark.parametrize("p,n", SIZES, ids=_ids(SIZES))
def test_evolution_is_a_markov_semigroup(p, n):
    integers = PAdicIntegers(p, n)
    free = UltrametricOperator.vladimirov(integers, 1.0, absorbing=False)
    killed = UltrametricOperator.vladimirov(integers, 1.0)
    f = _random(integers)
    for operator in (free, killed):
        twice = operator.evolve(operator.evolve(f, 0.3), 0.4)
        assert np.allclose(twice, operator.evolve(f, 0.7))
        assert np.allclose(operator.evolve(f, 0.0), f)
        delta = np.eye(integers.size)[0]
        assert (operator.evolve(delta, 0.5) > -1e-12).all()
    assert float(integers.integrate(free.evolve(f, 2.0))) == pytest.approx(
        float(integers.integrate(f))
    )
    assert np.allclose(free.evolve(f, 50.0), integers.integrate(f))
    survival = integers.integrate(killed.evolve(np.ones(integers.size), 1.5))
    assert float(survival) == pytest.approx(np.exp(killed.eigenvalues[0] * 1.5))


@pytest.mark.parametrize("p,n", SMALL, ids=_ids(SMALL))
def test_exact_solutions_agree(p, n):
    integers = PAdicIntegers(p, n)
    operator = _operators(integers)[2]
    problem = operator.problem(_random(integers), 0.0, 2.0)
    ts = np.linspace(0.0, 2.0, 5)
    wavelets = exact_linear_solution(problem, ts, transform=KozyrevTransform(integers))
    fourier = exact_linear_solution(
        problem, ts, transform=FourierTransform(integers.group)
    )
    direct = np.stack([operator.evolve(problem.y0, t) for t in ts])
    assert np.allclose(wavelets, fourier, atol=1e-10)
    assert np.allclose(wavelets, direct, atol=1e-12)


def test_lawson_is_exact_and_ignores_stiffness():
    integers = PAdicIntegers(2, 6)
    operator = UltrametricOperator.vladimirov(integers, 2.0)
    assert -operator.eigenvalues.min() > 4000
    problem = operator.problem(_random(integers), 0.0, 1.0)
    transform = KozyrevTransform(integers)
    ts = np.linspace(0.0, 1.0, 3)
    solution = solve_group_ode(problem, transform=transform, ts=ts)
    expected = exact_linear_solution(problem, ts, transform=transform)
    assert np.allclose(solution.ys, expected, atol=1e-10)
    assert int(solution.stats["num_steps"]) <= 10


def _reaction(t, u, args):
    return u - u**3


@pytest.mark.parametrize("p,n", SMALL, ids=_ids(SMALL))
def test_nonlinear_solvers_agree(p, n):
    integers = PAdicIntegers(p, n)
    operator = UltrametricOperator.vladimirov(integers, 0.7, absorbing=False)
    problem = operator.problem(_random(integers), 0.0, 1.0, nonlinearity=_reaction)
    ts = np.linspace(0.0, 1.0, 4)
    tol = dict(rtol=1e-10, atol=1e-12)
    wavelets = solve_group_ode(
        problem,
        Lawson(diffrax.Dopri5()),
        transform=KozyrevTransform(integers),
        ts=ts,
        **tol,
    )
    physical = solve_group_ode(problem, diffrax.Tsit5(), space="physical", ts=ts, **tol)
    plain = solve_group_ode(
        problem, diffrax.Tsit5(), transform=KozyrevTransform(integers), ts=ts, **tol
    )
    for other in (physical, plain):
        assert np.allclose(wavelets.ys, other.ys, atol=1e-8)
    assert wavelets.ys.dtype == np.float64
    if integers.size <= 8:
        fourier = solve_group_ode(problem, ts=ts, **tol)
        assert np.allclose(wavelets.ys, fourier.ys, atol=1e-8)


def test_reaction_diffusion_conserves_symmetry():
    integers = PAdicIntegers(3, 3)
    operator = UltrametricOperator.vladimirov(integers, 1.0, absorbing=False)
    y0 = np.where(integers.ball(1, 1), 1.0, -0.5)
    problem = operator.problem(y0, 0.0, 3.0, nonlinearity=_reaction)
    solution = solve_group_ode(problem, transform=KozyrevTransform(integers), ts=[3.0])
    u = np.asarray(solution.ys[0])
    for center in range(3):
        values = u[integers.ball(center, 1)]
        assert np.allclose(values, values[0], atol=1e-8)
    assert (np.abs(u) <= 1 + 1e-8).all()


def test_complex_and_batched_initial_conditions():
    integers = PAdicIntegers(3, 2)
    operator = _operators(integers)[0]
    transform = KozyrevTransform(integers)
    y0 = _random(integers, 2) + 1j * _random(integers, 2, seed=3)
    problem = operator.problem(y0, 0.0, 1.0)
    solution = solve_group_ode(problem, transform=transform, ts=[0.5, 1.0])
    assert solution.ys.shape == (2, 2, integers.size)
    expected = jnp.stack([operator.evolve(y0, t) for t in (0.5, 1.0)])
    assert np.allclose(solution.ys, expected, atol=1e-8)


def test_jit_and_grad():
    integers = PAdicIntegers(2, 4)
    operator = UltrametricOperator.vladimirov(integers, 1.0)
    f = jnp.asarray(_random(integers))

    def energy(f, t):
        return jnp.sum(operator.evolve(f, t) ** 2)

    value = jax.jit(energy)(f, 0.3)
    assert float(value) == pytest.approx(float(energy(f, 0.3)))
    gradient = jax.grad(energy)(f, 0.3)
    expected = 2 * operator.evolve(operator.evolve(f, 0.3), 0.3)
    assert np.allclose(gradient, expected)
    dt = jax.grad(energy, argnums=1)(f, 0.3)
    expected = 2 * jnp.sum(operator.evolve(f, 0.3) * operator(operator.evolve(f, 0.3)))
    assert float(dt) == pytest.approx(float(expected))
    assert np.allclose(jax.jit(lambda f: operator(f))(f), operator(f))


def test_all_pairs_of_levels_are_orthogonal_under_the_operator():
    integers = PAdicIntegers(2, 3)
    transform = KozyrevTransform(integers)
    basis = transform.basis()
    operator = _operators(integers)[2]
    matrix = np.asarray(operator(basis)) @ basis.conj().T / integers.size
    for i, j in itertools.product(range(integers.size), repeat=2):
        if i != j:
            assert abs(matrix[i, j]) < 1e-10
