import numpy as np
import pytest
from galrax.padic import Orbit, PAdicEvent, PAdicMap, recurrence_times
from galrax.padic.polynomial import evaluate_exact


def _orbit(g, x, count, m):
    out = [[v % m for v in x]]
    for _ in range(count - 1):
        out.append([evaluate_exact(f, out[-1], m) for f in g])
    return out


def _random_map(rng, p, d):
    g = []
    for i in range(d):
        f = {
            tuple(int(j == k) for j in range(d)): int(rng.integers(-3, 4))
            * (p if k < i else 1)
            for k in range(d)
        }
        f[tuple(int(j == i) for j in range(d))] = int(rng.integers(1, 4)) * p + 1
        f[(0,) * d] = int(rng.integers(-3, 4))
        f[tuple(2 * int(j == i) for j in range(d))] = int(rng.integers(-2, 3)) * p
        g.append(f)
    return g


@pytest.mark.parametrize(
    "p,d,K", [(2, 1, 6), (3, 1, 5), (5, 1, 4), (2, 2, 4), (3, 2, 3), (5, 2, 2)]
)
def test_return_set_matches_a_full_period(p, d, K):
    rng = np.random.default_rng(10 * p + d)
    m, checked = p**K, 0
    for _ in range(6):
        g = _random_map(rng, p, d)
        x = rng.integers(0, m, d).tolist()
        bad = {(0,) * d: int(rng.integers(-5, 6))}
        bad.update(
            {
                tuple(int(j == k) for j in range(d)): int(rng.integers(-3, 4))
                for k in range(d)
            }
        )
        depth = int(rng.integers(1, K + 1))
        try:
            orbit = Orbit(p, K, g, x)
        except ValueError:
            continue
        checked += 1
        count = orbit.tail + orbit.step * p**K
        points = _orbit(g, x, count, m)
        assert [list(np.atleast_1d(v)) for v in orbit(np.arange(count))] == points
        events = orbit.times(PAdicEvent(bad, depth))
        hits = [
            n for n, y in enumerate(points) if evaluate_exact(bad, y, p**depth) == 0
        ]
        assert [n for n in range(count) if n in events] == hits
        assert events.first() == (hits[0] if hits else None)
    assert checked >= 3


@pytest.mark.parametrize("p,level", [(2, 6), (3, 4), (5, 3)])
def test_one_dimensional_orbits_match_padic_map(p, level):
    rng = np.random.default_rng(p)
    compared = 0
    for _ in range(10):
        coefficients = rng.integers(-6, 7, 3).tolist()
        coefficients[1] = coefficients[1] * p + 1
        x = int(rng.integers(p**level))
        bad = PAdicEvent(
            rng.integers(-5, 6, 3).tolist(), int(rng.integers(1, level + 1))
        )
        try:
            expected = PAdicMap(p, coefficients).hitting_times(x, bad, level)
        except ValueError as error:  # a tail is fine for Orbit, attraction is not
            if "attracting" in str(error):
                with pytest.raises(ValueError):
                    Orbit(p, level, coefficients, x)
            continue
        found = Orbit(p, level, coefficients, x).times(bad)
        assert (found.balls, found.period) == (expected.balls, expected.period)
        compared += 1
    assert compared >= 3


def test_berstel_sequence_has_six_zeros():
    # a_{n+3} = 2 a_{n+2} - 4 a_{n+1} + 4 a_n, a = 0, 0, 1: Strassmann over 3-adic
    # time leaves room for exactly the six known zeros.
    events = recurrence_times(3, 6, [4, -4, 2], [0, 0, 1])
    a = [0, 0, 1]
    for _ in range(2000):
        a.append(2 * a[-1] - 4 * a[-2] + 4 * a[-3])
    zeros = [n for n, v in enumerate(a) if v == 0]
    assert zeros == [0, 1, 4, 6, 13, 52]
    assert events.zero_bound == 6
    assert all((n in events) == (v % 3**6 == 0) for n, v in enumerate(a))


def test_degenerate_recurrence_vanishes_on_a_progression():
    events = recurrence_times(3, 5, [4, 0], [1, 0])  # a_{n+2} = 4 a_n: zero at odd n
    assert events.balls == [events.balls[0]] and events.balls[0].level == 0
    assert events.period == 2 and events.first() == 1 and events.zero_bound is None
    assert all((n in events) == (n % 2 == 1) for n in range(200))


def test_orbit_inside_the_variety():
    p = 3
    g = [{(1, 0): 1, (0, 0): p}, {(0, 1): 1, (1, 0): 2 * p, (0, 0): p * p}]
    events = Orbit(p, 5, g, (1, 1)).times(PAdicEvent({(0, 1): 1, (2, 0): -1}))
    assert events.measure() == 1 and events.zero_bound is None
    assert all(n in events for n in range(100))


def test_attracting_cycle_is_an_error():
    with pytest.raises(ValueError, match="invertible"):
        Orbit(3, 4, [{(2, 0): 1}, {(0, 1): 1}], (3, 1))
    with pytest.raises(ValueError, match="invertible"):
        recurrence_times(2, 4, [4, -4, 2], [0, 0, 1])
