import numpy as np
import pytest
from diffrax.padic_flow import matrix_power_index, PAdicMap


GENERATOR = PAdicMap(2, [1, 3, 2])  # 2x^2 + 3x + 1, full period modulo 2^k
PLANE = PAdicMap(
    3, [{(1, 0): 1, (0, 1): 1, (0, 0): 1}, {(1, 0): 1, (0, 0): 3, (2, 0): 3}]
)


def _naive(mapping, x, count, level):
    out = [x]
    for _ in range(count - 1):
        out.append(mapping(out[-1], level))
    return out


def _chebyshev(n):
    low, high = [1], [0, 1]
    for _ in range(n - 1):
        low, high = high, [-c for c in low] + [0] * (len(high) + 1 - len(low))
        high = [a + 2 * b for a, b in zip(high, [0] + low)]
    return high


def test_iterate_matches_the_loop():
    n = np.arange(10_000)
    assert GENERATOR.iterate(5, n, 20).tolist() == _naive(GENERATOR, 5, 10_000, 20)
    points = [tuple(int(v) for v in y) for y in PLANE.iterate((1, 2), n, 4)]
    assert points == _naive(PLANE, (1, 2), 10_000, 4)


def test_batched_equals_scalar_and_composes():
    rng = np.random.default_rng(0)
    n = np.array([int(v) for v in rng.integers(-(10**6), 10**6, 50)], dtype=object)
    batch = GENERATOR.iterate(7, n, 24).tolist()
    assert batch == [GENERATOR.iterate(7, int(k), 24) for k in n]
    tails = PAdicMap(3, [0, 1, 1])  # 1 -> 2 -> 0 modulo 3, then periodic
    assert tails.iterate(1, np.arange(60), 5).tolist() == _naive(tails, 1, 60, 5)
    a, b = 10**30 + 7, 3**50
    for mapping, x, level in ((GENERATOR, 7, 24), (PLANE, (1, 2), 4)):
        y = mapping.iterate(x, a, level)
        assert np.array_equal(
            mapping.iterate(y, b, level), mapping.iterate(x, a + b, level)
        )
        assert np.array_equal(
            mapping.iterate(mapping(x, level), -1, level), np.array(x)
        )


def test_index_is_n_modulo_the_period():
    for mapping, x, level in (
        (GENERATOR, 7, 16),
        (PAdicMap(5, [3, 2]), 1, 4),
        (PLANE, (1, 2), 4),
    ):
        period = mapping.period(x, level)
        for n in (0, 1, 12345, 10**20 + 3):
            y = mapping.iterate(x, n, level)
            assert mapping.index(x, y, level) == n % period


@pytest.mark.parametrize(
    "p,coefficients", [(2, [1, 3, 2]), (3, [1, 1, 6]), (3, [2, 1, 3]), (5, [1, 1, 5])]
)
def test_transitive_polynomials_have_period_p_to_the_k(p, coefficients):
    mapping = PAdicMap(p, coefficients)
    assert mapping.is_minimal()
    for x in range(p):
        assert mapping.period(x, 5) == p**5


def test_chebyshev_periods_follow_li_lu_tan_chen():
    # arXiv:2209.14658, Table II: periods of x <- T_n(x) modulo p^k, k = 1, ..., 5.
    t29, t19 = PAdicMap(7, _chebyshev(29)), PAdicMap(13, _chebyshev(19))
    assert [t29.period(2, k) for k in range(1, 6)] == [2, 2, 14, 98, 686]
    assert [t29.period(3, k) for k in range(1, 6)] == [1, 2, 14, 98, 686]
    assert [t29.period(5, k) for k in range(1, 6)] == [2, 2, 2, 14, 98]
    assert [t19.period(3, k) for k in range(1, 5)] == [3, 3, 12, 12]


def _first(matrix, start, target, m, count):
    v, (a, b), (c, d) = list(start), *matrix
    for n in range(count):
        if (v[0] - target) % m == 0:
            return n
        v = [(a * v[0] + b * v[1]) % m, (c * v[0] + d * v[1]) % m]
    return None


@pytest.mark.parametrize("p,k", [(5, 3), (7, 2), (3, 3)])
def test_chebyshev_lucas_dickson_indices(p, k):
    rng = np.random.default_rng(p)
    m = p**k
    count = 2 * p**2 * m  # past every period of a 2 x 2 matrix modulo p^k
    for x in (1, -1, 1 + p * 3, -1 + p * 2, int(rng.integers(m)), int(rng.integers(m))):
        P, Q, a = int(rng.integers(m)), int(rng.integers(1, p)), int(rng.integers(1, p))
        cases = [
            ([[0, 1], [-1, 2 * x]], (1, x)),  # Chebyshev T_n(x)
            ([[0, 1], [-Q, P]], (2, P)),  # Lucas V_n(P, Q)
            ([[0, 1], [-a, x]], (2, x)),  # Dickson D_n(x, a)
        ]
        for matrix, start in cases:
            n = int(rng.integers(0, 10**4))
            v, (r0, r1), (r2, r3) = list(start), *matrix
            for _ in range(n):
                v = [(r0 * v[0] + r1 * v[1]) % m, (r2 * v[0] + r3 * v[1]) % m]
            for target in (v[0], int(rng.integers(m))):
                expected = _first(matrix, start, target, m, count)
                assert matrix_power_index(p, k, matrix, start, target) == expected
