"""d-dimensional formal groups as commuting flows, and Chabauty zeros in residue
discs of a genus 2 curve."""

import math
from fractions import Fraction

import numpy as np
import pytest
from galrax.padic import (
    chabauty_zeros,
    EllipticCurve,
    FormalGroup,
    FormalGroupFlow,
    lie_bracket,
    tiny_integrals,
    weierstrass_omega,
)


p, K = 5, 6
m = p ** (K + 1)
CURVE_37, CURVE_389 = (0, 0, 1, -1, 0), (0, 1, 1, -2, 0)


def _r(x):
    x = Fraction(x)
    return x.numerator * pow(x.denominator, -1, m) % m


def _kernel(curve, P, depth=2):
    E = EllipticCurve(curve)
    n = next(n for n in range(1, 1000) if E.depth(E.multiply(n, P), p) >= depth)
    return E, E.multiply(n, P)


def _twisted(omega1, omega2):
    """Omega of log(w) = (L_1(w_1 + w_2^2), L_2(w_2)), a non-product group."""

    def omega(i, j, e):
        a, b = e
        if i == 1:
            return omega2(b) if (j, a) == (1, 0) else 0
        if j == 0 and b % 2 == 0:
            n = a + b // 2
            return omega1(n) * math.comb(n, a)
        if j == 1 and b % 2 == 1:
            n = a + (b - 1) // 2
            return 2 * omega1(n) * math.comb(n, a)
        return 0

    return omega


def test_products_match_the_one_dimensional_groups():
    multiplicative = lambda n: (-1) ** n  # noqa: E731
    group = FormalGroup.product(p, K, [multiplicative, multiplicative])
    alphas = [(1 + p**2 * 3, 1 + p**3), (1 + p**2 * 7, Fraction(1, 1 + p**2))]
    z = np.array([[_r(a - 1) for a in pair] for pair in alphas], dtype=object)
    one = FormalGroupFlow.multiplicative(p, K).logarithm(z.reshape(-1))
    assert np.array_equal(group.logarithm(z), one.reshape(z.shape))

    E, Q = _kernel(CURVE_37, (Fraction(0), Fraction(0)))
    F, R = _kernel(CURVE_389, (Fraction(0), Fraction(0)))
    pairs = FormalGroup.product(
        p, K, [weierstrass_omega(CURVE_37), weierstrass_omega(CURVE_389)]
    )
    z = np.array([[_r(E.parameter(Q)), _r(F.parameter(R))]], dtype=object)
    expected = [
        FormalGroupFlow.weierstrass(p, K, curve).logarithm(z[:, i])[0]
        for i, curve in enumerate((CURVE_37, CURVE_389))
    ]
    assert pairs.logarithm(z)[0].tolist() == expected
    assert np.array_equal(
        pairs.logarithm(pairs.exponential(pairs.logarithm(z))), pairs.logarithm(z)
    )


def test_fields_commute():
    group = FormalGroup(
        p, K, _twisted(weierstrass_omega(CURVE_37), lambda n: (-1) ** n), 2
    )
    X, Y = group.fields
    assert lie_bracket(p, K, X, Y) == [{}, {}]
    assert lie_bracket(p, K, X, X) == [{}, {}]


def test_logarithm_is_additive_in_exact_arithmetic():
    E, Q = _kernel(CURVE_37, (Fraction(0), Fraction(0)))
    points = [E.multiply(k, Q) for k in (1, 2, 5)]
    pairs = FormalGroup.product(p, K, [weierstrass_omega(CURVE_37)] * 2)
    for P1, P2 in ((points[0], points[1]), (points[1], points[2])):
        S = E.add(P1, P2)
        z = np.array(
            [
                [_r(E.parameter(P1)), _r(E.parameter(P2))],
                [_r(E.parameter(S)), _r(E.parameter(S))],
            ],
            dtype=object,
        )
        logs = pairs.logarithm(z)
        assert logs[1][0] == (logs[0][0] + logs[0][1]) % m

    # The twisted group: w + w' = phi^-1(phi(w) * phi(w')) for phi(w) = (w_1 + w_2^2,
    # w_2) and the multiplicative law on 1 + phi(w).
    twisted = FormalGroup(p, K, _twisted(lambda n: (-1) ** n, lambda n: (-1) ** n), 2)

    def law(w, v):
        z = [
            (1 + w[0] + w[1] ** 2) * (1 + v[0] + v[1] ** 2) - 1,
            (1 + w[1]) * (1 + v[1]) - 1,
        ]
        return [z[0] - z[1] ** 2, z[1]]

    w, v = [Fraction(p**2 * 3), Fraction(p**2, 7)], [Fraction(p**3), Fraction(-(p**2))]
    logs = twisted.logarithm(
        np.array([[_r(c) for c in u] for u in (w, v, law(w, v))], dtype=object)
    )
    assert logs[2].tolist() == ((logs[0] + logs[1]) % m).tolist()


def test_chabauty_counts_the_rational_points_of_a_genus_two_curve():
    # y^2 = x(x - 1)(x - 2)(x - 5)(x - 6): Jacobian of rank 1 and the ten rational
    # points below (McCallum, Poonen, "The method of Chabauty and Coleman"). At p = 7
    # they fill the eight residue discs.
    f, q, digits = [0, 60, -112, 65, -14, 1], 7, 8
    ell = tiny_integrals(
        f, q, digits, (3, 6), (10, -120)
    )  # log of [(10, -120) - (3, 6)]
    killing = (ell[1], -ell[0])
    known = {(3, 6): [(3, 6), (10, -120)], (3, -6): [(3, -6), (10, 120)]}
    known.update({(x, 0): [(x, 0)] for x in (0, 1, 2, 5, 6)})
    total = 0
    for base, points in known.items():
        zeros = chabauty_zeros(f, q, digits, base, killing)
        assert len(zeros.balls) == zeros.zero_bound == len(points)
        for x, y in points:
            s = Fraction(x - base[0], q) if base[1] % q else Fraction(y - base[1], q)
            assert int(s) in zeros
        total += zeros.zero_bound
    # The disc at infinity, through x = 1/X, y = Y/X^3: dx/y = -X dX/Y, x dx/y = -dX/Y.
    at_infinity = chabauty_zeros(
        [0] + f[:0:-1], q, digits, (0, 0), (-killing[1], -killing[0])
    )
    assert at_infinity.zero_bound == len(at_infinity.balls) == 1
    assert total + 1 == 10
    with pytest.raises(ValueError, match="same residue disc"):
        tiny_integrals(f, q, digits, (3, 6), (10, 120))
