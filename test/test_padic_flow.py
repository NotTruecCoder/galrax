import functools
import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from diffrax._padic_flow.arithmetic import (
    simplex,
    simplex_horner,
    simplex_size,
    valuation,
)
from diffrax._padic_flow.normal_form import _monomials, homological_matrix, smith
from diffrax.padic_flow import (
    advection_diffusion,
    Equilibrium,
    KoopmanOperator,
    Limbs,
    NormalForm,
    PAdicEvent,
    PAdicFlow,
    PAdicMap,
    PowerSeries,
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


@pytest.mark.parametrize("p,K", [(3, 100), (2, 150), (101, 12), (3, 19)])
def test_limb_arithmetic(p, K):
    res = Limbs(p, K)
    m = p**K
    rng = np.random.default_rng(0)
    a = [int(v) for v in rng.integers(0, 2**62, 300)]
    a = [v * v * v % m for v in a] + [0, 1, m - 1, p ** (K - 1)]
    b = [(v * 7 + 3) * v % m for v in a[::-1]]
    A, B = (
        res.asarray(np.array(a, dtype=object)),
        res.asarray(np.array(b, dtype=object)),
    )
    assert res.decode(A).tolist() == a
    assert res.decode(res.mul(A, B)).tolist() == [x * y % m for x, y in zip(a, b)]
    assert res.decode(res.add(A, B)).tolist() == [(x + y) % m for x, y in zip(a, b)]
    assert res.decode(res.sub(A, B)).tolist() == [(x - y) % m for x, y in zip(a, b)]
    assert res.decode(res.neg(A)).tolist() == [-x % m for x in a]
    k = rng.integers(0, K + 1, len(a))
    assert res.decode(res.shift(A, k)).tolist() == [
        x // p ** int(j) for x, j in zip(a, k)
    ]
    assert res.decode(res.truncate(A, k)).tolist() == [
        x % p ** int(j) for x, j in zip(a, k)
    ]
    valuations = [next((j for j in range(K) if x % p ** (j + 1)), K) for x in a]
    assert np.asarray(res.valuation(A)).tolist() == valuations
    units = [x if x % p else x + 1 for x in a]
    inverses = res.decode(res.unit_inverse(res.asarray(np.array(units, dtype=object))))
    assert all(x * y % m == 1 for x, y in zip(units, inverses))
    small = np.array([-5, 0, 7, 2**62], dtype=np.int64)
    assert res.decode(res.asarray(small)).tolist() == [int(x) % m for x in small]


@pytest.mark.parametrize("p,K", [(3, 60), (2, 100), (5, 40), (101, 12)])
def test_many_digit_flows(p, K):
    field = (0, 0, 4) if p == 2 else (0, 0, p)
    flow = PAdicFlow(p, K, list(field))
    assert isinstance(flow.residues, Limbs)
    q, m = field[2], p**K
    rng = np.random.default_rng(1)
    x = [int(v) ** 3 % m for v in rng.integers(0, 2**62, 20)]
    t = [int(v) ** 3 % m for v in rng.integers(0, 2**62, 20)]
    X, T = np.array(x, dtype=object), np.array(t, dtype=object)
    expected = [xi * pow((1 - q * ti * xi) % m, -1, m) % m for xi, ti in zip(x, t)]
    z = flow.flow(T, X)
    assert z.tolist() == expected
    assert flow.flow(T, X, method="taylor").tolist() == expected
    single = PAdicFlow(p, 6, list(field))
    assert np.array_equal((z % p**6).astype(np.int64), single.flow(T % p**6, X % p**6))
    time, on_orbit = flow.orbit_time(X, z)
    level = np.asarray(flow.orbit_level(X))
    assert on_orbit.all()
    assert all((s - ti) % p ** (K - int(k)) == 0 for s, ti, k in zip(time, t, level))


@pytest.mark.parametrize(
    "p,K,field",
    [
        (3, 9, ((((0, 1), 3), ((2, 0), 3)), (((0, 0), 3), ((1, 1), -6)))),
        (3, 6, ((((0, 1, 0), 3),), (((0, 0, 1), 3),), (((2, 0, 0), 3),))),
    ],
    ids=["plane", "space"],
)
def test_sparse_storage_matches_dense(p, K, field):
    sparse = _flow(p, K, field, storage="sparse")
    dense = _flow(p, K, field, storage="dense")
    d = sparse.dimension
    x = _random(p**K, 200, d)
    t = _random(p**K, 200, seed=1)
    z = dense.flow(t, x)
    assert np.array_equal(sparse.flow(t, x), z)
    assert np.array_equal(sparse.flow(t, x, method="taylor"), z)
    assert np.array_equal(dense.flow(t, x, method="taylor"), z)
    for got, want in zip(sparse.rectify(x, z), dense.rectify(x, z)):
        assert np.array_equal(got, want)
    assert sparse._table[0].size < dense._table[0].size
    with pytest.raises(ValueError):
        PAdicFlow(p, K, [dict(c) for c in field], storage="compressed")


def test_sparse_flow_in_four_dimensions():
    field = [{(0, 1, 0, 0): 3}, {(0, 0, 1, 0): 3}, {(0, 0, 0, 1): 3}, {(2, 0, 0, 0): 3}]
    flow = PAdicFlow(3, 5, field)
    assert flow.storage == "sparse" and sum(flow.space_digits) >= 1
    x = _random(3**5, 300, 4)
    t = _random(3**5, 300, seed=1)
    z = flow.flow(t, x)
    assert np.array_equal(flow.flow(t, x, method="taylor"), z)
    assert np.array_equal(flow.flow(-t, z), x)


def test_centres_per_coordinate():
    plane = [{(0, 1): 3, (2, 0): 3}, {(0, 0): 3, (1, 1): -6}]
    reference = PAdicFlow(3, 9, plane, space_digits=0)
    x = _random(3**9, 300, 2)
    t = _random(3**9, 300, seed=1)
    expected = reference.flow(t, x)
    for kwargs in [
        {"space_digits": (2, 1)},
        {"space_digits": (0, 3)},
        {"space_digits": (3, 2)},
    ]:
        flow = PAdicFlow(3, 9, plane, **kwargs)
        assert np.array_equal(flow.flow(t, x), expected), kwargs
    with pytest.raises(ValueError):
        PAdicFlow(3, 9, plane, space_digits=(1, 2, 3))
    with pytest.raises(ValueError):
        PAdicFlow(3, 9, plane, storage="dense", space_digits=(1, 2))


def test_linear_coordinates_get_no_centres():
    growth = TimeDependentFlow(3, 12, [{(1, 1): 1}])
    autonomous = growth.autonomous
    clock, state = autonomous.space_digits
    assert clock >= 1 and state == 0
    reference = TimeDependentFlow(3, 12, [{(1, 1): 1}], space_digits=0)
    t0 = _random(3**12, 100)
    t1 = t0 + 3 * _random(3**11, 100, seed=1)
    x = _random(3**12, 100, seed=2)
    assert np.array_equal(growth.flow(t0, t1, x), reference.flow(t0, t1, x))


@pytest.mark.parametrize(
    "d,deg,caps,weights,limit",
    [
        (2, 6, None, None, None),
        (3, 5, (5, 1, 2), None, None),
        (2, 8, (8, 3), (2, 3), 11),
    ],
)
def test_weighted_simplex_layout(d, deg, caps, weights, limit):
    exponents = simplex(d, deg, caps, weights, limit)
    assert len(exponents) == simplex_size(d, deg, caps, weights, limit)
    assert len(set(exponents)) == len(exponents)
    caps = caps or (deg,) * d
    weights = weights or (0,) * d
    for e in exponents:
        assert sum(e) <= deg and all(k <= c for k, c in zip(e, caps))
        assert limit is None or sum(w * k for w, k in zip(weights, e)) <= limit
    res = Residues(5, 9)
    m = 5**9
    x = _random(m, 40, d)
    coefficients = _random(m, 1, 2, len(exponents), seed=3)
    got = np.asarray(
        simplex_horner(
            res, coefficients, res.asarray(x), deg, None, caps, weights, limit
        )
    )
    for b in range(40):
        for c in range(2):
            want = 0
            for i, e in enumerate(exponents):
                term = int(coefficients[0, c, i])
                for j, k in enumerate(e):
                    term = term * pow(int(x[b, j]), k, m) % m
                want += term
            assert got[b, c] == want % m


def test_analytic_field_has_the_exact_flow():
    p, K = 3, 12
    m = p**K
    flow = PAdicFlow(p, K, [3 * PowerSeries.geometric(3)])
    x = _random(m, 300)
    t = _random(m, 300, seed=1)
    z = np.asarray(flow.flow(t, x)).astype(object)
    half = pow(2, -1, m)
    invariant = (z - 3 * z * z * half) % m
    expected = (x.astype(object) - 3 * x.astype(object) ** 2 * half + 3 * t) % m
    assert np.array_equal(invariant, expected)
    assert np.array_equal(flow.flow(t, x, method="taylor"), flow.flow(t, x))


def test_analytic_fields_agree_across_precisions():
    field = 3 * PowerSeries.exp(3) - 3 + PowerSeries.log1p(9) * 2
    coarse, fine = PAdicFlow(3, 7, [field]), PAdicFlow(3, 15, [field])
    x = _random(3**15, 300)
    t = _random(3**15, 300, seed=1)
    z = fine.flow(t, x)
    assert np.array_equal(np.asarray(z) % 3**7, coarse.flow(t, x))
    assert np.array_equal(fine.flow(t[::-1], z), fine.flow(t + t[::-1], x))
    plane = [3 * PowerSeries.exp(3, 1, 2), 3 * PowerSeries.variable(0, 2)]
    small, large = PAdicFlow(3, 5, plane), PAdicFlow(3, 9, plane)
    y = _random(3**9, 200, 2)
    assert np.array_equal(
        np.asarray(large.flow(t[:200], y)) % 3**5, small.flow(t[:200], y)
    )
    clock = TimeDependentFlow(3, 9, [PowerSeries.exp(3, 0, 2)])
    late = clock.flow(0, 3 * np.arange(10), np.zeros(10, dtype=np.int64))
    early = TimeDependentFlow(3, 5, [PowerSeries.exp(3, 0, 2)])
    assert np.array_equal(
        np.asarray(late) % 3**5,
        early.flow(0, 3 * np.arange(10), np.zeros(10, dtype=np.int64)),
    )
    with pytest.raises(ValueError, match="converges"):
        PAdicFlow(3, 6, [3 * PowerSeries.exp(1)])
    with pytest.raises(ValueError, match="divisible"):
        PAdicFlow(3, 6, [PowerSeries.geometric(3)])


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
    form = plane.linearization(Equilibrium((1, 1), 5, True))
    assert isinstance(form, NormalForm) and form.equilibrium == (1, 1)


X, Y = PowerSeries.variable(0, 2), PowerSeries.variable(1, 2)
NORMAL_FORMS = [
    ("line", 3, 8, [0, 3, 0, 3], (0,), True),
    (
        "real",
        3,
        8,
        [{(1, 0): 3, (0, 1): 21, (2, 0): 3}, {(1, 0): 3, (0, 1): 3, (1, 1): 3}],
        (0, 0),
        True,
    ),
    (
        "complex",
        3,
        8,
        [{(1, 0): 3, (0, 1): 6, (0, 2): 3}, {(1, 0): 3, (0, 1): 3, (2, 0): 3}],
        (0, 0),
        True,
    ),
    (
        "jordan",
        3,
        8,
        [{(1, 0): 3, (0, 1): 3, (0, 2): 3}, {(0, 1): 3, (2, 0): 3}],
        (0, 0),
        None,
    ),
    (
        "resonant",
        3,
        8,
        [{(1, 0): 3, (0, 2): 3}, {(0, 1): 6, (2, 0): 3, (1, 1): 3}],
        (0, 0),
        False,
    ),
    ("zero", 3, 8, [{(1, 1): 3}, {(0, 1): 3, (2, 0): 3}], (0, 0), False),
    (
        "space",
        3,
        6,
        [
            {(1, 0, 0): 3, (0, 1, 0): 6, (0, 0, 2): 3},
            {(1, 0, 0): 3, (0, 1, 0): 3},
            {(0, 0, 1): 6, (2, 0, 0): 3},
        ],
        (0, 0, 0),
        False,
    ),
    ("two", 2, 10, [{(1, 0): 4, (0, 2): 4}, {(0, 1): 12, (3, 0): 4}], (0, 0), False),
    (
        "analytic",
        3,
        8,
        [3 * X + 6 * Y + 3 * (PowerSeries.exp(3, 1, 2) - 1 - 3 * Y), 3 * X + 3 * Y],
        (0, 0),
        True,
    ),
    ("shifted", 5, 6, [{(2, 0): 5, (0, 1): -5}, {(0, 1): 5, (0, 0): -5}], (1, 1), None),
]


def _conjugates(flow, form, count=10, seed=0):
    p, K, d = flow.p, flow.precision, flow.dimension
    r = form.radius
    rng = np.random.default_rng(seed)
    centre = np.array(form.equilibrium, dtype=object)
    x = (
        centre + p**r * rng.integers(0, p ** (K - r), (count, d)).astype(object)
    ) % p**K
    t = rng.integers(0, p**K, count)
    points = x[:, 0] if d == 1 else x
    got = np.asarray(form.flow_map(t, points)).astype(object)
    want = np.asarray(flow.flow(t, points)).astype(object)
    back = np.asarray(form.inverse(form.forward(points))).astype(object)
    return np.array_equal(got, want) and np.array_equal(back, points)


@pytest.mark.parametrize(
    "name,p,K,field,point,linear", NORMAL_FORMS, ids=[case[0] for case in NORMAL_FORMS]
)
def test_normal_form_conjugates_the_flow(name, p, K, field, point, linear):
    flow = PAdicFlow(p, K, field)
    form = flow.normal_form(point)
    assert _conjugates(flow, form)
    certificate = form.certificate
    assert certificate.verify() and certificate.degree == form.degree
    assert certificate.margin + certificate.slope * form.degree >= K
    if linear is not None:
        assert form.linearizable == linear
    needed = 2 if p == 2 else 1
    for component in form.normal_form:
        assert all(c % p**needed == 0 for c in component.values())
    for component in form.transformation:
        assert all(c % p == 0 for e, c in component.items() if sum(e) >= 2)
    keys = [set(component) for component in form.resonant_terms]
    if name == "resonant":
        assert keys == [set(), {(2, 0)}]
    if name == "two":
        assert keys == [set(), {(3, 0)}]
    if name == "zero":
        assert keys[0] and all(e[1] == 0 for e in keys[0])
        assert keys[1] and all(e[1] == 1 for e in keys[1])
    if name == "space":
        assert not keys[0] and not keys[1] and keys[2]
        assert all(e[2] == 0 and sum(e) == 2 for e in keys[2])
        assert 2 in form.divisors and max(form.divisors[2]) >= K


def test_normal_form_thresholds_and_linear_part():
    jordan = PAdicFlow(3, 8, NORMAL_FORMS[3][3])
    kept = jordan.normal_form((0, 0))
    removed = jordan.normal_form((0, 0), resonance_digits=16)
    assert not kept.linearizable and removed.linearizable
    assert removed.radius >= kept.radius
    assert _conjugates(jordan, removed)
    real = PAdicFlow(3, 8, NORMAL_FORMS[1][3])
    cautious = real.normal_form((0, 0), resonance_digits=0)
    assert not cautious.linearizable and _conjugates(real, cautious)
    complex_pair = PAdicFlow(3, 8, NORMAL_FORMS[2][3])
    form = complex_pair.normal_form((0, 0))
    z = _random(3**8, 6, 2)
    for t in (1, 7, 3**8 - 2):
        matrix = np.array(form.exponential(t), dtype=object)
        linear = (z.astype(object) @ matrix.T) % 3**8
        moved = np.asarray(form._normal_flow.flow(t, z)).astype(object)
        assert np.array_equal(linear, moved)


def test_normal_form_certificate_detects_errors():
    flow = PAdicFlow(3, 8, NORMAL_FORMS[4][3])
    form = flow.normal_form((0, 0))
    assert form.certificate.verify()
    e, c = next(iter(form.transformation[0].items()))
    form.transformation[0][e] = (c + 3) % 3**8
    assert not form.certificate.verify()
    form.transformation[0][e] = c
    form.normal_form[1][(2, 0)] += 3**7
    assert not form.certificate.verify()
    form.normal_form[1][(2, 0)] -= 3**7
    assert form.certificate.verify()
    form.certificate.slope /= 2
    assert not form.certificate.verify()


def test_normal_form_sufficient_radius():
    # Theorem: r >= 1 + max(sigma - c, 0) always works; here c = 1, sigma = 4.
    flow = PAdicFlow(3, 8, NORMAL_FORMS[3][3])
    form = flow.normal_form((0, 0), radius=4)
    assert form.radius == 4 and form.certificate.verify()
    assert _conjugates(flow, form)


def test_normal_form_removes_cheap_large_divisors():
    # A Jordan block: some divisors exceed the threshold, but their numerators are
    # divisible enough that removing them costs nothing.
    field = [
        {(1, 0): 3, (0, 1): 3, (2, 1): -3, (1, 1): -3, (3, 0): 6},
        {(0, 1): 3, (1, 1): -3, (3, 0): 27, (0, 3): 6},
    ]
    flow = PAdicFlow(3, 8, field)
    form = flow.normal_form((0, 0))
    work = form.precision + form.resonance_digits + 2
    large = [s for exponents in form.divisors.values() for s in exponents]
    assert any(form.resonance_digits < s < work for s in large)
    assert form.linearizable and form.certificate.verify()
    assert _conjugates(flow, form)


def _matching_exponents(t, weights):
    """Increments of the least weights of the matchings of each size in the path
    with edge weights `t, w_1, t, w_2, ..., w_k, t`.
    """
    edges = [t]
    for w in weights:
        edges += [w, t]
    best = {(0, False): 0}
    for x in edges:
        new = {}
        for (size, taken), value in best.items():
            options = [((size, False), value)]
            if not taken:
                options.append(((size + 1, True), value + x))
            for key, total in options:
                new[key] = min(new.get(key, total), total)
        best = new
    least = [
        min(v for (size, _), v in best.items() if size == j)
        for j in range(len(weights) + 2)
    ]
    return [b - a for a, b in zip(least, least[1:])]


def test_smith_exponents_of_planar_jordan_blocks():
    # p odd and p not dividing n + 1: L_n splits over binary forms of degrees n + 1
    # and n - 1, on which it is a weighted chain whose exponents are matchings.
    def v(x, p):
        return valuation(x, p, 100)

    for p, lam, mu in [(3, 3, 3), (3, 9, 3), (3, 3, 1), (5, 5, 5), (5, 25, 1)]:
        a = v(mu, p)
        for n in range(2, 13):
            if (n + 1) % p == 0:
                continue
            L = homological_matrix([[lam, mu], [0, lam]], _monomials(2, n))
            got = sorted(smith(L % p**60, p, 60)[3])
            nu = v((n - 1) * lam, p)
            if nu <= a:
                want = [nu] * (2 * n + 2)
            else:
                want = sorted(
                    a + s
                    for k in (n + 1, n - 1)
                    for s in _matching_exponents(
                        nu - a, [v(j, p) for j in range(k, 0, -1)]
                    )
                )
            assert got == want


def test_normal_form_validation():
    flow = PAdicFlow(3, 8, NORMAL_FORMS[1][3])
    with pytest.raises(ValueError, match="equilibrium"):
        flow.normal_form((1, 2))
    with pytest.raises(ValueError, match="point of"):
        flow.normal_form((0, 0, 0))
    form = flow.normal_form((0, 0))
    with pytest.raises(ValueError, match="outside the domain"):
        form.forward(np.array([[1, 0]]))
    assert "linearizable" in repr(form)


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
