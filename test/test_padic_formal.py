"""Formal groups as flows, and logarithmic eigenvalues (paper, section "Formal groups
and logarithmic eigenvalues")."""

import math
from fractions import Fraction

import numpy as np
import pytest
from diffrax._padic_flow.arithmetic import valuation
from diffrax._padic_flow.formal import weierstrass_differential
from diffrax._padic_flow.normal_form import _monomials, homological_matrix, smith
from diffrax.padic_flow import EllipticCurve, FormalGroupFlow, logarithmic_exponents


# 389a1: rank 2 with generators (0, 0) and (1, 0); 37a1: rank 1 with generator (0, 0).
CURVE_389 = (0, 1, 1, -2, 0)
CURVE_37 = (0, 0, 1, -1, 0)


def _v(x, p):
    x = Fraction(x)
    if x == 0:
        return math.inf
    return valuation(abs(x.numerator), p, 10**4) - valuation(x.denominator, p, 10**4)


def _residue(x, modulus):
    x = Fraction(x)
    return x.numerator * pow(x.denominator, -1, modulus) % modulus


def _log_series(omega, z, p, digits, terms=80):
    """`sum_n omega_{n-1} z^n / n` modulo `p^digits`, directly."""
    m = p ** (digits + 8)
    zi, total = _residue(z, m), 0
    for n in range(1, terms):
        v = valuation(n, p, 64)
        term = _residue(omega[n - 1], m) * pow(zi, n, m) % m
        assert term % p**v == 0
        total += term // p**v * pow(n // p**v, -1, m)
    return total % p**digits


def _kernel_multiple(curve, P, p, depth):
    for n in range(1, 10_000):
        Q = curve.multiply(n, P)
        if curve.depth(Q, p) >= depth:
            return n, Q
    raise AssertionError


def test_weierstrass_differential():
    # y^2 = x^3 + a x + b: omega = 1 + 2a z^4 + 3b z^6 + 6a^2 z^8 + ...
    omega = weierstrass_differential((0, 0, 0, 2, 3), 9)
    assert omega[:9] == [1, 0, 0, 0, 4, 0, 9, 0, 24]
    assert all(c.denominator == 1 for c in weierstrass_differential(CURVE_389, 40))


@pytest.mark.parametrize("p,K", [(3, 10), (5, 8)])
def test_multiplicative_group_flow(p, K):
    group = FormalGroupFlow.multiplicative(p, K)
    alphas = [1 + p**2 * k for k in (1, 2, 7, 11)]
    z = np.array([a - 1 for a in alphas], dtype=object)
    expected = [
        _log_series([(-1) ** n for n in range(90)], a - 1, p, K + 1) for a in alphas
    ]
    assert list(group.logarithm(z)) == expected
    # [t] z = (1 + z)^t - 1.
    for t in (2, 5, p, p**3 + 1):
        got = group.multiple(np.full(z.shape, t, dtype=object), z)
        want = [(pow(a, t, p ** (K + 1)) - 1) % p ** (K + 1) for a in alphas]
        assert list(got) == want


@pytest.mark.parametrize("p,K,curve", [(5, 10, CURVE_389), (3, 12, CURVE_37)])
def test_elliptic_logarithm_is_a_rectifying_coordinate(p, K, curve):
    E = EllipticCurve(curve)
    group = FormalGroupFlow.weierstrass(p, K, curve)
    omega = weierstrass_differential(curve, 90)
    (P, *_) = [(Fraction(0), Fraction(0))]
    assert E.contains(P)
    n, Q = _kernel_multiple(E, P, p, 2)
    multiples = [E.multiply(k, Q) for k in (1, 2, 3, p)]
    z = np.array([_residue(E.parameter(R), p ** (K + 1)) for R in multiples], object)
    logs = group.logarithm(z)
    for R, w in zip(multiples, logs):
        assert w == _log_series(omega, E.parameter(R), p, K + 1)
    # The logarithm is a homomorphism and an isometry on the kernel.
    for k, w in zip((1, 2, 3, p), logs):
        assert w == k * logs[0] % p ** (K + 1)
    # Local divisibility, from the orbit theorem: v(z([k] Q)) = v(z(Q)) + v(k).
    for k in range(1, 2 * p + 1):
        R = E.multiply(k, Q)
        assert E.depth(R, p) == E.depth(Q, p) + _v(k, p)
    # Multiples at p-adic times, and agreement with exact rational multiples.
    for k in (4, p + 1):
        got = group.multiple(np.array([k], object), z[:1])[0]
        assert got == _residue(E.parameter(E.multiply(k, Q)), p ** (K + 1))


def test_logarithmic_exponents_are_smith_exponents():
    p, K = 5, 10
    E = EllipticCurve(CURVE_389)
    group = FormalGroupFlow.weierstrass(p, K, CURVE_389)
    points = []
    for P in [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))]:
        _, Q = _kernel_multiple(E, P, p, 2)
        points.append(Q)
    z = np.array([_residue(E.parameter(Q), p ** (K + 1)) for Q in points], object)
    lam = [int(w) for w in group.logarithm(z)]
    W = K + 1
    for n in range(2, 6):
        exact = logarithmic_exponents(points, n, p, curve=E)
        assert all(s < math.inf for s in exact.values())
        L = homological_matrix([[lam[0], 0], [0, lam[1]]], _monomials(2, n))
        computed = smith(L % p**W, p, W)[3]
        assert sorted(computed) == sorted(min(s, W) for s in exact.values())
        # Height certificate: 2 s log p <= h(x(Q)).
        for (m, i), s in exact.items():
            k = [m[j] - (j == i) for j in range(2)]
            R = E.combination(k, points)
            assert 2 * s * math.log(p) <= E.naive_height(R) + 1e-9


def test_logarithmic_exponents_multiplicative():
    p = 3
    alphas = [Fraction(10), Fraction(28, 19)]  # 1 mod 9 and 1 mod 3
    for n in range(2, 8):
        exact = logarithmic_exponents(alphas, n, p)
        for (m, i), s in exact.items():
            q = Fraction(1)
            for j, a in enumerate(alphas):
                q *= a ** (m[j] - (j == i))
            # v(q - 1) log p <= h(q - 1) <= h(q) + log 2 <= (n + 1) max h + log 2.
            h = max(math.log(max(a.numerator, a.denominator)) for a in alphas)
            assert s * math.log(p) <= (n + 1) * h + math.log(2) + 1e-9
    # A resonance: alpha_2 = alpha_1^2 gives 2 lambda_1 = lambda_2.
    exact = logarithmic_exponents([Fraction(4), Fraction(16)], 2, p)
    assert exact[((2, 0), 1)] == math.inf
    assert sum(s == math.inf for s in exact.values()) == 1
