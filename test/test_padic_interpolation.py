import itertools

import numpy as np
import pytest
from diffrax._padic_flow.arithmetic import valuation
from diffrax._padic_flow.interpolation import _parse
from diffrax._padic_flow.polynomial import evaluate_exact
from diffrax.padic_flow import MinimalAction, PAdicFlow, PowerSeries, time_one_field


def _apply(g, p, K, x):
    """`g(x)` modulo `p^K`, exactly."""
    point = [int(v) for v in np.atleast_1d(x)]
    return [evaluate_exact(f, point, p**K) for f in _parse(g, p, K)]


def _points(p, K, d, count, seed=0):
    rng = np.random.default_rng(seed)
    return [[int(v) for v in rng.integers(0, p**K, d)] for _ in range(count)]


MAPS = [
    (3, 8, [3, 1, 3]),
    (5, 6, [10, 1, 0, 5]),
    (2, 10, [4, 1, 4]),
    (3, 6, [{(1, 0): 1, (0, 2): 3, (0, 0): 3}, {(0, 1): 1, (1, 0): 9, (1, 1): 3}]),
]


@pytest.mark.parametrize("p, K, g", MAPS)
def test_time_one_map_and_iterates(p, K, g):
    flow = PAdicFlow.from_time_one(p, K, g)
    d = flow.dimension
    for x in _points(p, K, d, 6):
        iterate = list(x)
        for n in range(5):
            got = np.atleast_1d(np.asarray(flow.flow(n, x[0] if d == 1 else x)))
            assert [int(v) for v in got] == iterate
            iterate = _apply(g, p, K, iterate)
        # t = -1 is the inverse map.
        image = _apply(g, p, K, x)
        back = flow.flow(p**K - 1, image[0] if d == 1 else image)
        assert [int(v) for v in np.atleast_1d(np.asarray(back))] == x


@pytest.mark.parametrize("p, K, g", MAPS)
def test_time_one_field_is_the_displacement_to_first_order(p, K, g):
    field = time_one_field(p, K, g)
    d = len(field)
    c = 2 if p == 2 else 1
    m = p ** (c + 1)
    moved = _parse(g, p, K)
    for j in range(d):
        unit = tuple(int(i == j) for i in range(d))
        displacement = dict(moved[j])
        displacement[unit] = displacement.get(unit, 0) - 1
        keys = set(displacement) | set(field[j])
        for e in keys:
            assert (field[j].get(e, 0) - displacement.get(e, 0)) % m == 0
        assert all(valuation(v, p, K) >= c for v in field[j].values())


def test_jang_congruence():
    # Phi(t, x) = x + t (g(x) - x) modulo p^2 for c = 1.
    p, K, g = 3, 6, [{(1, 0): 1, (0, 2): 3, (0, 0): 3}, {(0, 1): 1, (1, 1): 3}]
    flow = PAdicFlow.from_time_one(p, K, g)
    rng = np.random.default_rng(1)
    for x in _points(p, K, 2, 8):
        t = int(rng.integers(0, p**K))
        moved = [(a - b) for a, b in zip(_apply(g, p, K, x), x)]
        got = [int(v) for v in np.asarray(flow.flow(t, x))]
        assert all((g_ - x_ - t * m_) % p**2 == 0 for g_, x_, m_ in zip(got, x, moved))


def test_analytic_time_one_map():
    p, K = 3, 7
    g = PowerSeries.variable() + 3 * PowerSeries.geometric(3)
    flow = PAdicFlow.from_time_one(p, K, [g])
    for (x,) in _points(p, K, 1, 6):
        assert int(flow.flow(1, x)) == _apply([g], p, K, [x])[0]


def test_time_one_validation():
    with pytest.raises(ValueError, match="identity modulo 3"):
        PAdicFlow.from_time_one(3, 6, [1, 1, 3])
    with pytest.raises(ValueError, match="identity modulo 4"):
        PAdicFlow.from_time_one(2, 8, [2, 1])
    assert time_one_field(3, 2, [9, 1]) == [{}]


def _orbit_closure(maps, p, digits, start):
    seen, stack = {tuple(start)}, [list(start)]
    while stack:
        x = stack.pop()
        for g in maps:
            y = tuple(_apply(g, p, digits, x))
            if y not in seen:
                seen.add(y)
                stack.append(list(y))
    return seen


ACTIONS = [
    # k = 1: Jang's case, displacements (3, 0) and (0, 3).
    (
        3,
        6,
        [
            [{(1, 0): 1, (0, 0): 3, (0, 2): 3}, {(0, 1): 1, (1, 0): 9}],
            [{(1, 0): 1, (0, 1): 9}, {(0, 1): 1, (0, 0): 3, (2, 0): 3}],
        ],
        (0, 0),
    ),
    # k = 2 > c = 1: no element moves a point by exactly 1/3.
    (
        3,
        6,
        [
            [{(1, 0): 1, (0, 0): 9, (1, 1): 3}, {(0, 1): 1, (2, 0): 3}],
            [{(1, 0): 1, (0, 2): 3}, {(0, 1): 1, (0, 0): 9, (1, 1): 3}],
        ],
        (9, 18),
    ),
    (5, 5, [[{(1,): 1, (0,): 5, (2,): 5}]], (2,)),
]


@pytest.mark.parametrize("p, K, maps, point", ACTIONS)
def test_minimal_action_chart(p, K, maps, point):
    action = MinimalAction(p, K, maps, point)
    k, d = action.level, action.dimension
    assert action.ball.level == k
    rng = np.random.default_rng(2)
    times = rng.integers(0, p ** (K - k), (12, d))
    for t, s in itertools.combinations(times[:6], 2):
        a = np.atleast_1d(np.asarray(action.chart(t if d > 1 else t[0]), dtype=object))
        b = np.atleast_1d(np.asarray(action.chart(s if d > 1 else s[0]), dtype=object))
        distance = min(valuation(int(v) % p**K, p, K) for v in a - b)
        gap = min(valuation(int(v) % p**K, p, K) for v in t - s)
        assert distance == min(k + gap, K)
    values = action.chart(times if d > 1 else times[:, 0])
    back = np.asarray(action.coordinates(values), dtype=object).reshape(-1, d)
    assert np.array_equal(back % p ** (K - k), times.astype(object) % p ** (K - k))
    inside = np.asarray(values, dtype=object).reshape(-1, d)
    assert all(action.ball.contains([int(v) for v in row], p) for row in inside)


@pytest.mark.parametrize("p, K, maps, point", ACTIONS)
def test_minimal_action_is_transitive_on_every_level(p, K, maps, point):
    action = MinimalAction(p, K, maps, point)
    k, d = action.level, action.dimension
    for extra in (1, 2):
        digits = k + extra
        orbit = _orbit_closure(maps, p, digits, [v % p**digits for v in point])
        assert len(orbit) == p ** (extra * d)
        assert all(action.ball.contains(x, p) for x in orbit)


def test_minimal_action_validation():
    g = [{(1, 0): 1, (0, 0): 3}, {(0, 1): 1, (1, 0): 3}]
    with pytest.raises(ValueError, match="do not form a basis"):
        MinimalAction(3, 5, [g, g], (0, 0))
    with pytest.raises(ValueError, match="needs 2 maps"):
        MinimalAction(3, 5, [g], (0, 0))
    action = MinimalAction(3, 5, [[{(1,): 1, (0,): 3}]], (0,))
    with pytest.raises(ValueError, match="outside the ball"):
        action.coordinates(1)
