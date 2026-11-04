import functools
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from diffrax.padic_flow import (
    advection_diffusion,
    Equilibrium,
    KoopmanOperator,
    PAdicEvent,
    PAdicFlow,
    PAdicMap,
    Residues,
    TimeDependentFlow,
)
from diffrax.ultrametric import UltrametricOperator


def _random(modulus, *shape, seed=0):
    return np.random.default_rng(seed).integers(0, modulus, shape)


@functools.cache
def _flow(p, precision, field, **kwargs):
    if isinstance(field[0], tuple):
        field = [dict(component) for component in field]
    return PAdicFlow(p, precision, list(field), **kwargs)


def _exact_series(terms, modulus):
    total = sum(terms, Fraction(0))
    return total.numerator * pow(total.denominator, -1, modulus) % modulus


FIELDS_1D = [
    (3, 12, (0, 0, 3)),
    (3, 10, (3, -3, 0, 6)),
    (5, 8, (5, 0, -5)),
    (7, 7, (0, 7, 14)),
    (2, 20, (4, 0, 4)),
    (3, 9, (9, 0, 0, 3)),
]
IDS_1D = [f"p{p}K{k}-{c}" for p, k, c in FIELDS_1D]


#
# Arithmetic
#


@pytest.mark.parametrize(
    "p,k,reduction",
    [(2, 61, "double"), (3, 38, "double"), (5, 26, "double"), (7, 21, "double")],
)
def test_double_word_reduction_is_exact(p, k, reduction):
    res = Residues(p, k)
    assert res.reduction == reduction
    m = res.modulus
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.integers(0, m, 5000, dtype=np.int64), [0, 1, m - 1]])
    b = np.concatenate(
        [rng.integers(0, m, 5000, dtype=np.int64), [m - 1, m - 1, m - 1]]
    )
    expected = [x * y % m for x, y in zip(a.tolist(), b.tolist())]
    assert np.asarray(res.mul(a, b)).tolist() == expected
    u = a[a % p != 0][:50]
    assert (np.asarray(res.mul(u, res.unit_inverse(u))) == 1).all()


@pytest.mark.parametrize("p,k", [(2, 30), (3, 19), (5, 13), (7, 11), (31, 6)])
def test_barrett_reduction_is_exact(p, k):
    m = p**k
    a = np.concatenate([_random(m, 20000), [0, 1, m - 1, m - 2, m // 2]])
    b = np.concatenate([_random(m, 20000, seed=1), [m - 1, m - 1, m - 1, 2, m // 2]])
    expected = ((a.astype(object) * b.astype(object)) % m).astype(np.int64)
    for reduction in ("barrett", "barrett32", "remainder"):
        res = Residues(p, k, reduction)
        assert np.array_equal(np.asarray(res.mul(a, b)), expected), reduction


def test_residue_operations():
    res = Residues(3, 10)
    m = res.modulus
    u = np.array([1, 2, 4, 5, m - 1, 12345 * 3 + 1])
    assert np.array_equal(np.asarray(res.mul(u, res.unit_inverse(u))), np.ones(6))
    x = np.array([0, 1, 3, 18, 81 * 2, m - 3])
    assert np.array_equal(np.asarray(res.valuation(x)), [10, 0, 1, 2, 4, 1])
    a, b = np.array([18 * 7]), np.array([9 * 5])
    q = int(res.divide(a, b, np.array([2]))[0])
    assert (q * 5 - 14) % 3**8 == 0
    assert np.array_equal(np.asarray(res.asarray(np.array([-1]))), [m - 1])
    assert int(res.power(np.array(2), 10)) == 1024


def test_residue_validation():
    with pytest.raises(ValueError):
        Residues(4, 3)
    with pytest.raises(ValueError, match="at most 38 digits"):
        Residues(3, 39)
    with pytest.raises(ValueError, match="reduction='double'"):
        Residues(3, 20, reduction="barrett")
    assert Residues(3, 20).reduction == "double"
    with pytest.raises(ValueError):
        Residues(3, 0)
    with pytest.raises(ValueError):
        Residues(3, 5, reduction="fast")
    with pytest.raises(ValueError):
        Residues(3, 5).asarray(np.array([0.5]))


#
# The flow
#


def test_closed_forms():
    p, K = 3, 12
    m = p**K
    x = _random(m, 40)
    t = _random(m, 40, seed=1)
    riccati = _flow(p, K, (0, 0, p))
    expected = [
        xi * pow((1 - p * ti * xi) % m, -1, m) % m
        for xi, ti in zip(x.tolist(), t.tolist())
    ]
    assert np.array_equal(np.asarray(riccati.flow(t, x)), expected)

    linear = _flow(p, K, (0, p))
    expected = [
        _exact_series(
            [Fraction(int(p * ti)) ** n / math.factorial(n) for n in range(60)], m
        )
        * int(xi)
        % m
        for xi, ti in zip(x.tolist(), t.tolist())
    ]
    assert np.array_equal(np.asarray(linear.flow(t, x)), expected)

    translation = _flow(p, K, (p,))
    assert np.array_equal(np.asarray(translation.flow(t, x)), (x + p * t) % m)


def test_linear_system():
    p, K = 5, 8
    m = p**K
    field = (((0, 1), p), ((1, 0), 2 * p)), (((1, 0), -p), ((0, 1), p))
    flow = _flow(p, K, field)
    matrix = [[Fraction(2 * p), Fraction(p)], [Fraction(-p), Fraction(p)]]
    xy = _random(m, 10, 2)
    t = _random(m, 10, seed=2)
    got = np.asarray(flow.flow(t, xy))
    for (x0, y0), ti, row in zip(xy.tolist(), t.tolist(), got):
        power = [[Fraction(int(i == j)) for j in range(2)] for i in range(2)]
        total = [[Fraction(0)] * 2 for _ in range(2)]
        for n in range(40):
            for i in range(2):
                for j in range(2):
                    total[i][j] += power[i][j] * Fraction(ti) ** n / math.factorial(n)
            power = [
                [sum(power[i][k] * matrix[k][j] for k in range(2)) for j in range(2)]
                for i in range(2)
            ]
        expected = [
            _exact_series([total[i][0] * x0, total[i][1] * y0], m) for i in range(2)
        ]
        assert row.tolist() == expected


@pytest.mark.parametrize("p,K,field", FIELDS_1D, ids=IDS_1D)
def test_methods_agree_and_form_a_group(p, K, field):
    flow = _flow(p, K, field)
    m = p**K
    x = _random(m, 64)
    s = _random(m, 64, seed=1)
    t = _random(m, 64, seed=2)
    table = flow.flow(t, x)
    assert np.array_equal(table, flow.flow(t, x, method="taylor"))
    assert np.array_equal(flow.flow(s, table), flow.flow(s + t, x))
    assert np.array_equal(flow.flow(0, x), x)
    once = flow.flow(1, x)
    assert np.array_equal(flow.flow(3, x), flow.flow(1, flow.flow(1, once)))
    assert np.array_equal(flow.flow(-1, once), x)
    assert np.array_equal(jax.jit(flow.flow)(t, x), table)


@pytest.mark.parametrize("p,K,field", FIELDS_1D, ids=IDS_1D)
def test_flow_is_an_isometry_and_unipotent(p, K, field):
    flow = _flow(p, K, field)
    res = flow.residues
    x = _random(p**K, 200)
    y = _random(p**K, 200, seed=1)
    t = _random(p**K, 200, seed=2)
    distance = res.valuation(res.sub(x, y))
    moved = res.valuation(res.sub(flow.flow(t, x), flow.flow(t, y)))
    assert np.array_equal(distance, moved)
    shift = res.valuation(res.sub(flow.flow(t, x), x))
    assert (shift >= flow.field_valuation).all()


def test_table_digits_do_not_change_the_result():
    p, K = 3, 10
    x = _random(p**K, 50)
    t = _random(p**K, 50, seed=1)
    reference = _flow(p, K, (3, 3, 3)).flow(t, x, method="taylor")
    for digits in (1, 2, 4, 10):
        flow = PAdicFlow(p, K, [3, 3, 3], table_digits=digits)
        assert np.array_equal(flow.flow(t, x), reference)
    for reduction in ("remainder", "barrett32"):
        flow = PAdicFlow(p, K, [3, 3, 3], reduction=reduction)
        assert np.array_equal(flow.flow(t, x), reference)


@pytest.mark.parametrize(
    "p,K,field",
    [
        (3, 19, [3, -3, 0, 6]),
        (5, 13, [5, 0, -5]),
        (3, 30, [3, -3, 0, 6]),
        (3, 9, [{(0, 1): 3, (2, 0): 3}, {(0, 0): 3, (1, 1): -6}]),
    ],
    ids=["p3K19", "p5K13", "p3K30", "plane"],
)
def test_space_reduction_is_exact(p, K, field):
    reference = PAdicFlow(p, K, field, space_digits=0)
    d = reference.dimension
    x = _random(p**K, 300, *((d,) if d > 1 else ()))
    t = _random(p**K, 300, seed=1)
    expected = reference.flow(t, x, method="taylor")
    z = reference.flow(t, x)
    assert np.array_equal(z, expected)
    s0, w0 = reference.rectify(x, z)
    for r in (1, 2, 4, None):
        flow = PAdicFlow(p, K, field, space_digits=r)
        assert flow.table_degree <= reference.table_degree
        assert np.array_equal(flow.flow(t, x), expected)
        s, w = flow.rectify(x, z)
        assert np.array_equal(s, s0) and np.array_equal(w, w0)
    with pytest.raises(ValueError):
        PAdicFlow(p, K, field, space_digits=K)


def test_shapes_and_broadcasting():
    flow = _flow(3, 8, (3, 0, 3))
    x = _random(3**8, 4, 5)
    assert flow.flow(7, x).shape == (4, 5)
    assert flow.flow(np.arange(6)[:, None, None], x).shape == (6, 4, 5)
    plane = _flow(5, 6, ((((0, 1), 5),), (((1, 0), 5),)))
    assert plane.flow(np.arange(3), np.zeros(2, dtype=np.int64)).shape == (3, 2)
    with pytest.raises(ValueError):
        plane.flow(1, np.zeros(3, dtype=np.int64))


def test_fractions_in_the_field():
    p, K = 5, 8
    m = p**K
    flow = PAdicFlow(p, K, [0, Fraction(5, 3)])
    reference = PAdicFlow(p, K, [0, 5 * pow(3, -1, m) % m])
    x = _random(m, 20)
    assert np.array_equal(flow.flow(11, x), reference.flow(11, x))


def test_field_validation():
    with pytest.raises(ValueError, match="divisible by 3"):
        PAdicFlow(3, 8, [1, 3])
    with pytest.raises(ValueError, match="divisible by 4"):
        PAdicFlow(2, 10, [2])
    with pytest.raises(ValueError, match="denominators"):
        PAdicFlow(3, 8, [Fraction(3, 3 * 3)])
    with pytest.raises(ValueError):
        PAdicFlow(3, 8, [{(0, 1): 3}])
    with pytest.raises(ValueError):
        PAdicFlow(3, 8, [3], table_digits=9)
    zero = PAdicFlow(3, 6, [0])
    x = _random(3**6, 5)
    assert np.array_equal(zero.flow(17, x), x)


#
# Mahler interpolation
#


@pytest.mark.parametrize("p,K,field", FIELDS_1D[:4], ids=IDS_1D[:4])
def test_mahler_interpolation(p, K, field):
    flow = _flow(p, K, field)
    m = p**K
    x = _random(m, 6)
    deltas = np.asarray(flow.mahler_coefficients(x)).astype(object)
    for n in range(K):
        assert all(int(v) % p**n == 0 for v in deltas[n])
    for t in _random(m, 5, seed=3).tolist():
        expected = sum(math.comb(t, n) * deltas[n] for n in range(K)) % m
        assert np.array_equal(np.asarray(flow.flow(t, x)), expected.astype(np.int64))


#
# Orbits
#


@pytest.mark.parametrize("p,K,field", FIELDS_1D, ids=IDS_1D)
def test_orbits_are_balls(p, K, field):
    flow = _flow(p, K, field)
    n = min(K, 6 if p <= 3 else 4)
    side = p**n
    points = np.arange(side)
    image = np.asarray(flow.flow(1, points)) % side
    level = np.minimum(np.asarray(flow.orbit_level(points)), n)
    for x in range(side):
        orbit, y = {x}, int(image[x])
        while y != x:
            orbit.add(y)
            y = int(image[y])
        ball = set(range(x % p ** level[x], side, p ** level[x]))
        assert orbit == ball


def test_orbit_levels_see_equilibria():
    p, K = 3, 10
    flow = _flow(p, K, (0, -3, 3))
    assert int(flow.orbit_level(0)) == K
    assert int(flow.orbit_level(1)) == K
    assert int(flow.orbit_level(2)) == 1
    assert int(flow.orbit_level(4)) == 2
    assert np.array_equal(
        flow.flow(_random(p**K, 5), np.array([0, 1])[:, None]), [[0] * 5, [1] * 5]
    )


@pytest.mark.parametrize("p,K,field", FIELDS_1D, ids=IDS_1D)
def test_rectifying_coordinate(p, K, field):
    flow = _flow(p, K, field)
    m = p**K
    x = _random(m, 100)
    t = _random(m, 100, seed=1)
    k = np.asarray(flow.orbit_level(x))
    z = flow.flow(t, x)
    s, foot = flow.rectify(x, z)
    moving = k < K
    period = p ** (K - k)
    assert np.array_equal((np.asarray(s) - t) % period * moving, np.zeros(100))
    assert np.array_equal(np.asarray(foot)[moving], x[moving])
    time, on_orbit = flow.orbit_time(x, z)
    assert np.asarray(on_orbit).all()
    assert np.array_equal(np.asarray(flow.flow(time, x)), np.asarray(z))
    later = flow.flow(5, z)
    shifted, _ = flow.rectify(x, later)
    assert np.array_equal((np.asarray(shifted) - np.asarray(s) - 5) % period, 0 * k)


def test_flow_box_in_the_plane():
    p, K = 3, 9
    m = p**K
    field = (((0, 1), 3), ((2, 0), 3)), (((0, 0), 3), ((1, 1), -6))
    flow = _flow(p, K, field)
    res = flow.residues
    x = _random(m, 50, 2)
    k = np.asarray(flow.orbit_level(x))
    offsets = _random(m, 50, 2, seed=1) * (p ** np.minimum(k, K))[:, None]
    z = res.add(x, offsets)
    s, foot = flow.rectify(x, z)
    assert np.array_equal(np.asarray(flow.flow(s, foot)), np.asarray(z))
    moving = k < K
    f = np.asarray(flow.vector_field(x))
    axis = np.argmin(np.asarray(res.valuation(f)), axis=1)
    rows = np.arange(50)[moving]
    assert np.array_equal(np.asarray(foot)[rows, axis[moving]], x[rows, axis[moving]])

    z2 = res.add(x, _random(m, 50, 2, seed=2) * (p ** np.minimum(k, K))[:, None])
    s2, foot2 = flow.rectify(x, z2)
    scale = p ** -k.astype(float)
    time_distance = float(p) ** -np.asarray(res.valuation(res.sub(s, s2))).astype(float)
    foot_distance = float(p) ** -np.asarray(res.valuation(res.sub(foot, foot2))).min(1)
    space_distance = float(p) ** -np.asarray(res.valuation(res.sub(z, z2))).min(1)
    expected = np.maximum(scale * time_distance, foot_distance)
    exact = np.asarray(res.valuation(res.sub(s, s2))) < K - k
    assert np.allclose(space_distance[moving & exact], expected[moving & exact])


def _step_table(flow, level):
    side = flow.p**level
    d = flow.dimension
    points = np.indices((side,) * d).reshape(d, -1).T
    image = np.asarray(flow.flow(1, points if d > 1 else points[:, 0])) % side
    return image.reshape((side,) * d + ((d,) if d > 1 else ()))


def _brute_hitting(step, p, x, y, level):
    point = np.asarray(x) % p**level
    y = np.asarray(y) % p**level
    for n in range(p ** (level * point.size) + 1):
        if (point == y).all():
            return n
        point = step[tuple(np.atleast_1d(point))]
    return -1


@pytest.mark.parametrize("p,K,field", [FIELDS_1D[1], FIELDS_1D[2]], ids=IDS_1D[1:3])
def test_hitting_and_return_times(p, K, field):
    flow = _flow(p, K, field)
    level = 4
    x = _random(p**K, 12)
    y = _random(p**K, 12, seed=1)
    near = (x + p * _random(p**K, 12, seed=2)) % p**K
    step = _step_table(flow, level)
    for target in (y, near):
        fast = np.asarray(flow.first_hitting_time(x, target, level))
        for xi, yi, n in zip(x, target, fast):
            assert n == _brute_hitting(step, p, xi, yi, level)
    backs = np.asarray(flow.return_time(x, level))
    for xi, back in zip(x, backs):
        point, n = step[xi % p**level], 1
        while point != xi % p**level:
            point, n = step[point], n + 1
        assert back == n


def test_hitting_times_in_the_plane():
    p, K = 3, 8
    field = (((0, 1), 3), ((2, 0), 3)), (((0, 0), 3),)
    flow = _flow(p, K, field)
    x = _random(p**K, 6, 2)
    targets = flow.flow(_random(p**K, 6, seed=1), x)
    targets = np.asarray(targets) + np.array(
        [[0, 0], [p**6, 0], [0, 1], [0, 0], [9, 0], [0, 0]]
    )
    fast = np.asarray(flow.first_hitting_time(x, targets, 3))
    step = _step_table(flow, 3)
    for xi, yi, n in zip(x, targets, fast):
        assert n == _brute_hitting(step, p, xi, yi, 3)


def test_orbit_average():
    p, K = 3, 10
    flow = _flow(p, K, (3, -3, 0, 6))
    level = 5
    values = np.random.default_rng(0).standard_normal(p**level)
    step = _step_table(flow, level)
    for x in _random(p**K, 5).tolist():
        point, total = x % p**level, 0.0
        steps = 3 * p**level
        for _ in range(steps):
            total += values[point]
            point = step[point]
        assert float(flow.orbit_average(values, x, level)) == pytest.approx(
            total / steps
        )


#
# Koopman operators
#


@pytest.mark.parametrize("time", [1, 4])
def test_koopman_cycles(time):
    flow = _flow(3, 8, (3, -3, 0, 6))
    koopman = KoopmanOperator(flow, 4, time=time)
    u = np.random.default_rng(0).standard_normal(koopman.size)
    image = np.asarray(flow.flow(time, np.arange(koopman.size))) % 3**4
    assert np.array_equal(koopman(u), u[image])
    assert koopman.cycle_lengths.sum() == koopman.size
    for length, order in koopman.groups:
        assert np.array_equal(koopman.permutation[order], np.roll(order, -1, axis=1))


def test_jump_evolution_matches_the_matrix_exponential():
    flow = _flow(5, 6, (5, 0, 5))
    koopman = KoopmanOperator(flow, 2)
    n = koopman.size
    shift = np.zeros((n, n))
    shift[np.arange(n), koopman.permutation] = 1
    u = np.random.default_rng(0).standard_normal((2, n))
    expected = u @ scipy.linalg.expm(0.7 * 1.3 * (shift - np.eye(n))).T
    assert np.allclose(koopman.evolve_jumps(u, 0.7, rate=1.3), expected)


def test_koopman_in_the_plane():
    flow = _flow(3, 6, ((((0, 1), 3),), (((1, 0), -3), ((0, 0), 3))))
    koopman = KoopmanOperator(flow, 2)
    assert koopman.size == 81
    n = koopman.size
    shift = np.zeros((n, n))
    shift[np.arange(n), koopman.permutation] = 1
    u = np.random.default_rng(1).standard_normal(n)
    expected = scipy.linalg.expm(0.5 * (shift - np.eye(n))) @ u
    assert np.allclose(koopman.evolve_jumps(u, 0.5), expected)


def test_advection_diffusion():
    flow = _flow(3, 8, (3, -3, 0, 6))
    koopman = KoopmanOperator(flow, 3)
    operator = UltrametricOperator.vladimirov(
        __import__("diffrax").ultrametric.PAdicIntegers(3, 3), 0.8
    )
    n = koopman.size
    shift = np.zeros((n, n))
    shift[np.arange(n), koopman.permutation] = 1
    kernel = operator.kernel
    diffusion = kernel[(np.arange(n)[:, None] - np.arange(n)[None, :]) % n]
    assert np.allclose(shift @ diffusion, diffusion @ shift)
    u0 = np.random.default_rng(0).standard_normal(n)
    generator = diffusion + 2.0 * (shift - np.eye(n))
    expected = scipy.linalg.expm(0.6 * generator) @ u0
    got = advection_diffusion(koopman, operator, u0, 0.6, rate=2.0)
    assert np.allclose(got, expected)
    with pytest.raises(ValueError):
        advection_diffusion(KoopmanOperator(flow, 2), operator, u0, 0.6)


def test_jit_of_queries():
    flow = _flow(3, 10, (3, -3, 0, 6))
    x = jnp.asarray(_random(3**10, 30))
    z = flow.flow(123, x)
    s, _ = jax.jit(flow.rectify)(x, z)
    assert np.array_equal(np.asarray(flow.flow(s, x)), np.asarray(z))
    hits = jax.jit(lambda x, y: flow.first_hitting_time(x, y, 3))(x, z)
    assert (np.asarray(hits) >= 0).all()


#
# More digits
#


@pytest.mark.parametrize("p,K", [(3, 30), (3, 38), (5, 26), (2, 50)])
def test_double_precision_flows(p, K):
    field = (0, 0, 4) if p == 2 else (0, 0, p)
    flow = PAdicFlow(p, K, list(field))
    q = field[2]
    m = p**K
    rng = np.random.default_rng(0)
    x = rng.integers(0, m, 30, dtype=np.int64)
    t = rng.integers(0, m, 30, dtype=np.int64)
    expected = [
        xi * pow((1 - q * ti * xi) % m, -1, m) % m
        for xi, ti in zip(x.tolist(), t.tolist())
    ]
    assert np.asarray(flow.flow(t, x)).tolist() == expected
    assert np.array_equal(flow.flow(t, x, method="taylor"), flow.flow(t, x))
    single = PAdicFlow(p, 12, list(field))
    assert np.array_equal(np.asarray(flow.flow(t, x)) % p**12, single.flow(t, x))


#
# Time-dependent equations
#


def test_time_dependent_closed_forms():
    p, K = 3, 12
    m = p**K
    t0 = np.array([5, 7, 11, 0])
    t1 = t0 + 3 * np.array([4, 100, 7, 33])
    x0 = np.array([1, 2, 3, 40])
    quadrature = TimeDependentFlow(p, K, [{(1, 0): 1}])
    expected = [
        (xi + (b * b - a * a) * pow(2, -1, m)) % m
        for xi, a, b in zip(x0.tolist(), t0.tolist(), t1.tolist())
    ]
    assert np.asarray(quadrature.flow(t0, t1, x0)).tolist() == expected

    growth = TimeDependentFlow(p, K, [{(1, 1): 1}])
    got = np.asarray(growth.flow(t0, t1, x0)).tolist()
    for value, xi, a, b in zip(got, x0.tolist(), t0.tolist(), t1.tolist()):
        z = Fraction(b * b - a * a, 2)
        factor = _exact_series([z**n / math.factorial(n) for n in range(60)], m)
        assert value == xi * factor % m


def test_time_dependent_composition_and_validation():
    p, K = 5, 8
    field = [{(1, 0, 1): 1, (0, 2, 0): 2}, {(0, 1, 0): -1, (2, 0, 0): 3}]
    flow = TimeDependentFlow(p, K, field)
    x = _random(p**K, 10, 2)
    t0 = _random(p**K, 10, seed=1)
    t1 = t0 + 5 * _random(p**K, 10, seed=2)
    t2 = t1 + 5 * _random(p**K, 10, seed=3)
    direct = flow.flow(t0, t2, x)
    assert np.array_equal(flow.flow(t1, t2, flow.flow(t0, t1, x)), direct)
    assert np.array_equal(flow.flow(t2, t0, direct), x)
    assert np.array_equal(jax.jit(flow.flow)(t0, t2, x), direct)
    with pytest.raises(ValueError, match="divisible by 5"):
        flow.flow(0, 1, x)
    binary = TimeDependentFlow(2, 16, [{(1, 1): 1}])
    with pytest.raises(ValueError, match="divisible by 4"):
        binary.flow(0, 2, 3)
    assert binary.flow(0, 8, 3).shape == ()


#
# Events
#


def _holds(flow, polys, x, depth, times):
    points = np.asarray(flow.flow(np.asarray(list(times)), x))
    points = points.reshape(len(times), -1).tolist()
    out = []
    for point in points:
        ok = True
        for poly in polys:
            total = 0
            for e, c in poly.items():
                term = int(c)
                for xi, k in zip(point, np.atleast_1d(e)):
                    term *= xi ** int(k)
                total += term
            ok = ok and total % flow.p**depth == 0
        out.append(ok)
    return out


@pytest.mark.parametrize("p,K,field", FIELDS_1D[:4], ids=IDS_1D[:4])
def test_event_sets_match_brute_force(p, K, field):
    flow = _flow(p, K, field)
    depth = 3
    rng = np.random.default_rng(1)
    for _ in range(3):
        g = {(k,): int(c) for k, c in enumerate(rng.integers(-4, 5, 3))}
        g[(0,)] = g.get((0,), 0) + int(rng.integers(0, p**K))
        x = int(rng.integers(0, p**K))
        event = PAdicEvent(g, depth=depth)
        events = flow.event_times(event, x)
        times = range(p ** (depth + 1))
        expected = _holds(flow, [g], x, depth, times)
        assert [t in events for t in times] == expected
        first = next((t for t, hit in zip(times, expected) if hit), None)
        assert events.first() == first
        if first is not None:
            n, state = flow.first_event(event, x)
            assert n == first and int(state) == int(flow.flow(first, x))


def test_events_agree_with_hitting_times():
    flow = _flow(3, 10, (3, -3, 0, 6))
    x = _random(3**10, 8)
    y = _random(3**10, 8, seed=1)
    fast = np.asarray(flow.first_hitting_time(x, y, 4))
    for xi, yi, n in zip(x.tolist(), y.tolist(), fast):
        event = PAdicEvent({(1,): 1, (0,): -yi}, depth=4)
        first = flow.event_times(event, xi).first()
        assert (first if first is not None else -1) == n

    plane = _flow(3, 8, ((((0, 1), 3), ((2, 0), 3)), (((0, 0), 3),)))
    x = _random(3**8, 4, 2)
    targets = np.asarray(plane.flow(_random(3**8, 4, seed=2), x))
    for xi, yi in zip(x, targets):
        event = PAdicEvent(
            [{(1, 0): 1, (0, 0): -int(yi[0])}, {(0, 1): 1, (0, 0): -int(yi[1])}],
            depth=3,
        )
        first = plane.event_times(event, xi).first()
        assert first == int(plane.first_hitting_time(xi, yi, 3))


def test_strassmann_bound_and_exact_zero():
    flow = _flow(3, 12, (0, 0, 3))
    x, target_time = 7, 12345
    z = int(flow.flow(target_time, x))
    events = flow.event_times(PAdicEvent({(1,): 1, (0,): -z}), x)
    assert events.zero_bound == 1
    assert len(events.balls) == 1
    assert target_time in events
    assert events.measure() == pytest.approx(3.0 ** -events.balls[0].level)
    never = PAdicEvent({(1,): 3, (0,): 1})
    assert not flow.event_times(never, x)
    assert flow.event_times(never, x).first() is None
    assert flow.first_event(never, x) == (None, None)


#
# Equilibria and linearisation
#


def test_equilibria():
    flow = _flow(3, 10, (0, -3, 0, 3))
    found = flow.equilibria()
    assert sorted(e.point[0] for e in found) == [0, 1, 3**9 - 1]
    assert all(e.certified and e.level == 9 for e in found)
    double = PAdicFlow(3, 10, [12, -12, 3])
    (root,) = double.equilibria()
    assert not root.certified and root.point[0] % 3**root.level == 2
    plane = _flow(5, 6, ((((2, 0), 5), ((0, 1), -5)), (((0, 1), 5), ((0, 0), -5))))
    found = plane.equilibria()
    assert len(found) == 2 and all(e.certified for e in found)
    signs = sorted(e.point[0] % 5**e.level == 1 for e in found)
    assert signs == [False, True]
    for e in found:
        assert e.point[1] == 1 and (e.point[0] ** 2 - 1) % 5**e.level == 0
    assert plane.jacobian([1, 1]).tolist() == [[10, 5**6 - 5], [0, 5]]


@pytest.mark.parametrize(
    "p,K,field", [(3, 12, (0, -3, 0, 3)), (5, 9, (0, 5, 5)), (7, 8, (0, 14, 0, 7))]
)
def test_linearization_matches_the_flow(p, K, field):
    flow = PAdicFlow(p, K, list(field))
    checked = 0
    for equilibrium in flow.equilibria():
        if not equilibrium.certified:
            continue
        lin = flow.linearization(equilibrium)
        assert (lin.equilibrium - equilibrium.point[0]) % p**equilibrium.level == 0
        for k in (1, 2, 7):
            x = (lin.equilibrium + k * p**lin.level) % p**K
            for t in (1, 5, 12345):
                assert lin.flow_map(t, x) == int(flow.flow(t, x))
            z = int(flow.flow(77, x))
            level = int(flow.orbit_level(x))
            assert lin.orbit_time(x, z) == 77 % p ** (K - level)
            assert lin.inverse(lin.forward(x)) == x
            checked += 1
        with pytest.raises(ValueError, match="outside the domain"):
            lin.forward((lin.equilibrium + 1) % p**K)
    assert checked > 0


def test_linearization_validation():
    flow = PAdicFlow(3, 8, [0, 0, 3])
    with pytest.raises(ValueError, match="degenerate"):
        flow.linearization(0)
    plane = _flow(5, 6, ((((2, 0), 5), ((0, 1), -5)), (((0, 1), 5), ((0, 0), -5))))
    with pytest.raises(ValueError, match="d = 1"):
        plane.linearization(Equilibrium((1, 1), 5, True))


#
# Maps
#


def _brute_cycles(table):
    seen = np.zeros(len(table), dtype=bool)
    cycles = []
    for start in range(len(table)):
        index, order, x = {}, [], start
        while x not in index and not seen[x]:
            index[x] = len(order)
            order.append(x)
            x = int(table[x])
        if x in index:
            cycle = order[index[x] :]
            k = cycle.index(min(cycle))
            cycles.append(tuple(cycle[k:] + cycle[:k]))
        seen[order] = True
    return sorted(cycles)


def _canonical(cycles):
    out = []
    for cycle in cycles:
        cycle = list(cycle)
        k = cycle.index(min(cycle))
        out.append(tuple(cycle[k:] + cycle[:k]))
    return sorted(out)


@pytest.mark.parametrize("p,level", [(2, 7), (3, 5), (5, 3), (7, 3)])
def test_cycle_tree_matches_brute_force(p, level):
    rng = np.random.default_rng(p)
    for _ in range(25):
        coefficients = rng.integers(-6, 7, int(rng.integers(2, 6))).tolist()
        mapping = PAdicMap(p, coefficients)
        assert _canonical(mapping.cycles(level)) == _brute_cycles(mapping.table(level))


@pytest.mark.parametrize("p", [2, 3, 5])
def test_minimality_matches_transitivity(p):
    rng = np.random.default_rng(10 + p)
    level = {2: 8, 3: 5, 5: 3}[p]
    seen = set()
    candidates = [[1, 1], [1, 1, p], [p - 1, 1 + p, p]]
    candidates += [rng.integers(-4, 5, 3).tolist() for _ in range(40)]
    for coefficients in candidates:
        mapping = PAdicMap(p, coefficients)
        brute = _brute_cycles(mapping.table(level))
        transitive = len(brute) == 1 and len(brute[0]) == p**level
        assert mapping.is_minimal() == transitive
        seen.add(transitive)
    assert seen == {True, False}


def test_black_box_t_function():
    mapping = PAdicMap(2, lambda x: x + (x * x | 5))
    assert mapping.is_minimal(12)
    assert len(mapping.cycles(10)) == 1
    assert mapping.is_bijective(10)
    other = PAdicMap(2, lambda x: x ^ (x << 1))
    assert _canonical(other.cycles(6)) == _brute_cycles(other.table(6))


def test_minimal_decomposition_kinds():
    attracting = PAdicMap(3, [0, 0, 1])
    assert "attracting" in {c.kind for c in attracting.minimal_decomposition(4)}
    rotation = PAdicMap(5, [0, 2])
    decomposition = rotation.minimal_decomposition(3)
    assert any(c.kind == "periodic" and c.points == (0,) for c in decomposition)
    assert any(c.kind == "minimal" for c in decomposition)
    odometer = PAdicMap(3, [1, 1])
    (component,) = odometer.minimal_decomposition(5)
    assert component.kind == "minimal"
    assert len(component.points) == 3**component.level
    assert PAdicMap(5, [0, 1, 5]).is_bijective()
    assert not PAdicMap(5, [0, 0, 1]).is_bijective()
