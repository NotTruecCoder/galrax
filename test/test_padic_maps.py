from fractions import Fraction

import numpy as np
import pytest
from galrax.padic import Ball, PAdicEvent, PAdicFlow, PAdicMap


def _orbit(mapping, x, count, level):
    out = [x % mapping.p**level]
    for _ in range(count - 1):
        out.append(mapping(out[-1], level))
    return out


def test_quadratic_generator_jump_ahead():
    generator = PAdicMap(2, [1, 3, 2])  # x -> 2x^2 + 3x + 1 mod 2^16, full period
    orbit = _orbit(generator, 7, 3000, 16)
    assert [generator.iterate(7, n, 16) for n in range(3000)] == orbit
    n = 10**40
    assert generator.iterate(7, n + 1, 16) == generator(generator.iterate(7, n, 16), 16)
    for n in np.random.default_rng(0).integers(0, 3000, 50):
        assert generator.index(7, orbit[n], 16) == orbit.index(orbit[n])
    assert generator.index(7, generator.iterate(7, n, 16), 16) == n % 2**16


@pytest.mark.parametrize("p,level", [(2, 7), (3, 5), (5, 3), (7, 3)])
def test_jump_ahead_matches_brute_force(p, level):
    rng = np.random.default_rng(p)
    checked = 0
    for _ in range(40):
        mapping = PAdicMap(p, rng.integers(-6, 7, int(rng.integers(2, 6))).tolist())
        x = int(rng.integers(p**level))
        try:
            mapping.iterate(x, 0, level)
        except ValueError:
            continue
        checked += 1
        orbit = _orbit(mapping, x, 400, level)
        assert [mapping.iterate(x, n, level) for n in range(400)] == orbit
        for y in orbit[:5] + rng.integers(0, p**level, 5).tolist():
            expected = orbit.index(y) if y in orbit else None
            assert mapping.index(x, y, level) == expected
    assert checked > 10


def test_tail_and_attracting_orbits():
    mapping = PAdicMap(3, [0, 1, 1])  # x -> x^2 + x: 1 -> 2 -> 0 modulo 3
    orbit = _orbit(mapping, 1, 200, 5)
    assert [mapping.iterate(1, n, 5) for n in range(200)] == orbit
    with pytest.raises(ValueError, match="periodic"):
        mapping.iterate(1, -1, 5)
    square = PAdicMap(3, [0, 0, 1])  # x -> x^2 from 3: attracted to 0
    assert [square.iterate(3, n, 4) for n in range(6)] == _orbit(square, 3, 6, 4)
    with pytest.raises(ValueError, match="polynomial"):
        PAdicMap(2, lambda x: x + 1).iterate(0, 5, 4)


def test_inverse_and_roots_of_permutation_polynomials():
    level = 12
    mba = PAdicMap(2, [3, 5, 2])  # 2x^2 + 5x + 3, a permutation of Z/2^12
    rng = np.random.default_rng(1)
    for x in rng.integers(0, 2**level, 30).tolist():
        assert mba.iterate(mba(x, level), -1, level) == x
        y = x
        for _ in range(3):
            y = mba.iterate(y, Fraction(1, 3), level)
        assert y == mba(x, level)
    with pytest.raises(ValueError, match="p-adic"):
        mba.iterate(1, Fraction(1, 2), level)


def test_negative_iterates_without_a_power_of_p_period():
    doubling = PAdicMap(5, [0, 2])  # period 4 modulo 5
    for n in range(1, 6):
        assert doubling.iterate(1, -n, 4) == pow(2, -n, 5**4)
        assert doubling.iterate(doubling.iterate(7, -n, 4), n, 4) == 7
    with pytest.raises(ValueError, match="power of p"):
        doubling.iterate(1, Fraction(1, 3), 4)


def test_inverse_and_roots_agree_with_the_flow():
    level = 10
    near_identity = [4, 5, 4]  # 4x^2 + 5x + 4 = id mod 4
    mapping = PAdicMap(2, near_identity)
    flow = PAdicFlow.from_time_one(2, level, near_identity)
    xs = np.arange(0, 2**level, 37)
    for t in (Fraction(-1), Fraction(1, 3), Fraction(-5, 7)):
        time = t.numerator * pow(t.denominator, -1, 2**level)
        expected = np.asarray(flow.flow(time, xs)).tolist()
        assert [mapping.iterate(int(x), t, level) for x in xs] == expected


@pytest.mark.parametrize("p,level", [(2, 8), (3, 4), (5, 3)])
def test_hitting_times_match_brute_force(p, level):
    rng = np.random.default_rng(10 + p)
    checked = 0
    for _ in range(40):
        coefficients = rng.integers(-6, 7, 3).tolist()
        coefficients[1] = coefficients[1] * p + 1
        mapping = PAdicMap(p, coefficients)
        x = int(rng.integers(p**level))
        bad = rng.integers(-5, 6, 3).tolist()
        depth = int(rng.integers(1, level + 1))
        try:
            events = mapping.hitting_times(x, PAdicEvent(bad, depth), level)
        except ValueError:
            continue
        checked += 1
        orbit = _orbit(mapping, x, 3 * p**level, level)
        for n, y in enumerate(orbit):
            value = (bad[0] + bad[1] * y + bad[2] * y * y) % p**depth
            assert (n in events) == (value == 0)
        hits = [n for n, y in enumerate(orbit) if n in events]
        assert events.first() == (hits[0] if hits else None)
    assert checked > 10


def test_unreachable_over_the_integers():
    # x <- 1 - 3x from x = 1: 1, -2, 7, -20, 61, ... is 1 mod 11 at n = 0, but
    # (-3)^10 = 1 mod 121, so it is never 100 modulo 121, hence never 100.
    loop = PAdicMap(11, [1, -3])
    mod_p = loop.hitting_times(1, PAdicEvent([-100, 1], 1), 3)
    assert mod_p.balls == [Ball((0,), 0)] and mod_p.period == 10  # n = 0 mod 10
    assert not loop.hitting_times(1, PAdicEvent([-100, 1], 2), 3)
    x = 1
    for n in range(500):
        assert (n in mod_p) == ((x - 100) % 11 == 0)
        assert x != 100
        x = 1 - 3 * x


def test_numpy_points_and_non_periodic_period():
    T = PAdicMap(3, [1, 1, 3])
    x = 3**30 + 7
    assert T(np.int64(x), 40) == T(x, 40) and isinstance(T(np.int64(x), 40), int)
    with pytest.raises(ValueError, match="not periodic"):
        PAdicMap(3, [0, 0, 1]).period(2, 3)
