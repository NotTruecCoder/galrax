import itertools

import pytest
from galrax.padic import OrbitClosure
from galrax.padic.arithmetic import valuation
from galrax.padic.interpolation import _parse
from galrax.padic.polynomial import evaluate_exact


# (x + 3, y) and (x, y + 3x): the displacements at 0 span only 3 Z_3 x 0, and the
# commutator moves y by -9. The orbit of 0 is 3 Z_3 x 9 Z_3.
SHEAR = [
    [{(1, 0): 1, (0, 0): 3}, {(0, 1): 1}],
    [{(1, 0): 1}, {(0, 1): 1, (1, 0): 3}],
]
# (x + 5, y, z) and (x, y + 5, z + 5xy + 5x^2): levels 1, 1 and 3.
HEISENBERG = [
    [{(1, 0, 0): 1, (0, 0, 0): 5}, {(0, 1, 0): 1}, {(0, 0, 1): 1}],
    [
        {(1, 0, 0): 1},
        {(0, 1, 0): 1, (0, 0, 0): 5},
        {(0, 0, 1): 1, (1, 1, 0): 5, (2, 0, 0): 5},
    ],
]
# Both preserve the line y = 0.
LINE = [
    [{(1, 0): 1, (0, 0): 3}, {(0, 1): 4}],
    [{(1, 0): 1, (2, 0): 3}, {(0, 1): 1, (1, 1): 3}],
]


def _orbit(maps, p, digits, point):
    """The orbit of `point` modulo `p^digits`, by search: the maps permute the
    classes, so forward images suffice."""
    m = p**digits
    maps = [_parse(g, p, digits) for g in maps]
    seen, frontier = {tuple(point)}, [tuple(point)]
    while frontier:
        new = []
        for x in frontier:
            for g in maps:
                y = tuple(evaluate_exact(f, x, m) for f in g)
                if y not in seen:
                    seen.add(y)
                    new.append(y)
        frontier = new
    return seen


@pytest.mark.parametrize(
    "p, maps, levels, words, outside",
    [
        (3, SHEAR, (1, 2), ("g1", "[g1, g2]"), (0, 3)),
        (5, HEISENBERG, (1, 1, 3), ("g1", "g2", "[g1, g2]"), (0, 0, 25)),
    ],
)
def test_ball_in_orbit_closure(p, maps, levels, words, outside):
    d = len(maps[0])
    closure = OrbitClosure(p, 6, maps, (0,) * d)
    assert closure.levels == levels and closure.words == words
    k = closure.level
    assert closure.ball.level == k == max(levels)
    digits = k + 1
    orbit = _orbit(maps, p, digits, (0,) * d)
    ball = itertools.product(range(0, p**digits, p**k), repeat=d)
    assert all(x in orbit for x in ball)
    assert outside not in orbit  # the ball of radius p^{-(k - 1)} is not in it


def test_chart_is_an_isometry_onto_the_ball():
    closure = OrbitClosure(3, 6, SHEAR, (0, 0))
    m, k = 3**6, closure.level
    times = list(itertools.product(range(9), repeat=2))
    images = {t: closure.chart(t) for t in times}
    assert len({tuple(v % 3 ** (k + 2) for v in z) for z in images.values()}) == 81
    for s, t in itertools.combinations(times[:20], 2):
        gap = [(a - b) % m for a, b in zip(images[s], images[t])]
        step = min(valuation((a - b) % m, 3, 6) for a, b in zip(s, t))
        assert min(valuation(v, 3, 6) for v in gap) == k + step


def test_invariant_line_gives_an_arc():
    closure = OrbitClosure(3, 6, LINE, (0, 0))
    assert closure.rank == 1 and closure.levels == (1,) and closure.ball is None
    for t in range(1, 30):
        x, y = closure.chart([t])
        assert y == 0 and valuation(x, 3, 6) == 1 + valuation(t, 3, 6)


def test_agrees_with_a_basis_of_displacements():
    g = [{(1, 0): 1, (0, 0): 3, (0, 2): 3}, {(0, 1): 1, (1, 0): 9}]
    h = [{(1, 0): 1, (0, 1): 9}, {(0, 1): 1, (0, 0): 3, (2, 0): 3}]
    closure = OrbitClosure(3, 8, [g, h], (0, 0))
    assert closure.levels == (1, 1) and closure.depth == 1  # MinimalAction's level 1


def test_rejects_maps_far_from_the_identity():
    with pytest.raises(ValueError):
        OrbitClosure(3, 6, [[{(1, 0): 2}, {(0, 1): 1}]], (0, 0))
    with pytest.raises(ValueError):
        OrbitClosure(2, 6, [[{(1, 0): 1, (0, 0): 2}, {(0, 1): 1}]], (0, 0))
