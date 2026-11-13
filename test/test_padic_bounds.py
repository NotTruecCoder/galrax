import itertools
import math
from fractions import Fraction

import jax
import numpy as np
import pytest
from diffrax._padic_flow.arithmetic import valuation
from diffrax._padic_flow.normal_form import (
    _monomials,
    _rank,
    _residual_splitting,
    homological_matrix,
    smith,
)
from diffrax.padic_flow import (
    EllipticCurve,
    height_bound,
    linear_degree,
    Linearization,
    logarithmic_exponents,
    PAdicFlow,
    smith_exponent_bound,
    TimeDependentFlow,
)


MATRICES = [
    [[3, 0], [0, 6]],  # resonant: 2 lambda_1 = lambda_2
    [[3, 3], [0, 3]],  # Jordan block
    [[3, 6], [3, 3]],  # eigenvalues 3(1 +- sqrt 2), not in Q_3
    [[3, 21], [3, 3]],  # eigenvalues 3(1 +- sqrt 7) in Q_3
    [[0, 0], [0, 3]],  # zero eigenvalue
    [[3, 6, 0], [3, 3, 0], [0, 0, 6]],
    [[Fraction(3, 2), 3], [0, 6]],
]


def _minors(L, j, p, cap):
    rows, cols = L.shape
    best = cap
    for r in itertools.combinations(range(rows), j):
        for c in itertools.combinations(range(cols), j):
            sub = [[int(L[a, b]) for b in c] for a in r]
            det = round(np.linalg.det(np.array(sub, dtype=float)))
            if det:
                best = min(best, valuation(abs(det), p, cap))
    return best


@pytest.mark.parametrize("matrix", MATRICES)
def test_smith_exponent_bound(matrix):
    p = 3
    d = len(matrix)
    q = math.lcm(*(Fraction(c).denominator for row in matrix for c in row))
    integral = [[int(Fraction(c) * q) for c in row] for row in matrix]
    for n in range(2, 5):
        L = homological_matrix(integral, _monomials(d, n))
        rho, bound = smith_exponent_bound(matrix, n, p)
        assert rho == np.linalg.matrix_rank(np.array(L, dtype=float))
        if bound is None:
            assert rho == 0
            continue
        W = bound + 1
        s = smith(L % p**W, p, W)[3]
        finite = [v for v in s if v < W]
        # Past the bound, the zero factors are exactly the kernel over Q.
        assert len(finite) == rho
        assert max(finite) <= bound


def test_smith_exponents_are_differences_of_minors():
    p, cap = 3, 30
    L = homological_matrix([[3, 3], [0, 3]], _monomials(2, 2))
    s = sorted(smith(L % p**cap, p, cap)[3])
    previous = 0
    for j in range(1, _rank(L) + 1):
        current = _minors(np.array(L, dtype=object), j, p, cap)
        assert s[j - 1] == current - previous
        previous = current


def test_one_dimensional_exponents():
    # s_1(n) = v(lambda) + v(n - 1) for d = 1, lambda != 0.
    p = 3
    for lam in (3, 6, 9, 15):
        for n in range(2, 12):
            (s,) = smith(homological_matrix([[lam]], [(n,)]) % p**20, p, 20)[3]
            assert s == valuation(lam, p, 20) + valuation(n - 1, p, 20)


def test_height_bound_and_linear_segment():
    p = 3
    alphas = [Fraction(10), Fraction(28, 19)]
    for n in range(2, 8):
        bound = height_bound(alphas, n, p)
        exact = logarithmic_exponents(alphas, n, p)
        assert max(exact.values()) <= bound
        assert p**bound <= 2 * 28 ** (n + 1) < p ** (bound + 1)
    sigma = 12
    top = linear_degree(alphas, p, sigma)
    assert height_bound(alphas, top, p) <= sigma < height_bound(alphas, top + 1, p)
    with pytest.raises(ValueError, match="gram_eigenvalue"):
        height_bound(alphas, 2, p, curve=EllipticCurve([0, 0, 1, -1, 0]))
    with pytest.raises(ValueError, match="odd p"):
        height_bound(alphas, 2, 2)


def test_residual_splitting_is_exact():
    p, digits = 3, 12
    for matrix in ([[3, 6, 0], [3, 3, 0], [0, 0, 6]], [[3, 0], [0, 6]]):
        found = _residual_splitting(matrix, p, digits)
        assert found is not None
        T, inverse, groups = found
        m = p**digits
        product = np.array(inverse, dtype=object) @ np.array(matrix, dtype=object)
        conjugate = product @ np.array(T, dtype=object) % m
        block = {j: g for g, group in enumerate(groups) for j in group}
        d = len(matrix)
        assert all(
            conjugate[i, j] == 0
            for i in range(d)
            for j in range(d)
            if block[i] != block[j]
        )


def test_rectify_rejects_points_outside_the_flow_box():
    flow = PAdicFlow(3, 8, [3, -3, 0, 6])
    k = int(flow.orbit_level(1))
    with pytest.raises(ValueError, match="outside the ball"):
        flow.rectify(1, 1 + 3 ** (k - 1))
    with pytest.raises(Exception, match="outside the ball"):
        jax.block_until_ready(jax.jit(flow.rectify)(1, 1 + 3 ** (k - 1)))
    s, w = flow.rectify(1, flow.flow(5, 1))
    assert int(s) == 5 and int(w) == 1


def test_time_dependent_flow_checks_under_jit():
    flow = TimeDependentFlow(3, 6, [{(1, 1): 1}])
    with pytest.raises(Exception, match="divisible by 3"):
        jax.block_until_ready(jax.jit(flow.flow)(0, 1, 5))


def test_linearization_orbit_time_rejects_other_orbits():
    flow = PAdicFlow(3, 8, [0, 3, 3])
    lin = Linearization(flow, 0)
    x = 9
    z = lin.flow_map(7, x)
    assert lin.orbit_time(x, z) % 3 ** (8 - 3) == 7
    with pytest.raises(ValueError, match="not on the orbit"):
        lin.orbit_time(x, 27)
    with pytest.raises(ValueError, match="not on the orbit"):
        lin.orbit_time(x, 18)
    assert lin.orbit_time(0, 0) == 0
    with pytest.raises(ValueError, match="equilibrium"):
        lin.orbit_time(0, 9)
