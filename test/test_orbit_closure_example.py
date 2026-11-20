import pathlib
import random
import sys

import pytest
from galrax.padic import OrbitClosure
from galrax.padic.polynomial import evaluate_exact


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

import orbit_closure as oc  # noqa: E402


def _lifted(surface, q, free, u, m):
    """The point of the surface with free coordinates `q + p u`, by Hensel."""
    p = surface.p
    solved = next(j for j in range(3) if j not in free)
    z = list(q)
    for a, v in zip(free, u):
        z[a] = (q[a] + p * v) % m
    inverse = pow(surface.gradient(q, m)[solved], -1, m)
    for _ in range(2 * surface.top):
        z[solved] = (z[solved] - surface.value(surface.F, z, m) * inverse) % m
    assert surface.value(surface.F, z, m) == 0
    return z


@pytest.mark.parametrize("kind, p, parameter", [("markoff", 5, 3), ("wehler", 3, 2)])
def test_chart_maps_are_the_surface_maps(kind, p, parameter):
    surface = oc.Surface(
        {"markoff": oc.markoff, "wehler": oc.wehler}[kind](parameter), p
    )
    q = surface.lift(surface.points()[0], random.Random(1))
    maps = surface.generators(q, conjugate=True)
    _, free = surface.chart(q)
    m, M = p ** (oc.K + 1), p**oc.K
    rng = random.Random(2)
    for g, (h, (i, j), power) in zip(maps, surface.words):
        u = [rng.randrange(M) for _ in range(2)]
        z = _lifted(surface, q, free, u, m)
        for a in h:
            z = surface.involution(z, a, m)
        for _ in range(power):
            z = surface.involution(surface.involution(z, j, m), i, m)
        for a in reversed(h):
            z = surface.involution(z, a, m)
        assert [(z[a] - q[a]) // p % M for a in free] == [
            evaluate_exact(f, u, M) for f in g
        ]
        assert all(oc._affine(f, a, p) is False for a, f in enumerate(g))


def test_ball_is_in_the_orbit_of_the_involutions():
    p = 3
    surface = oc.Surface(oc.wehler(2), p)
    q = surface.lift(surface.points()[0], random.Random(1))
    closure = OrbitClosure(p, oc.K, surface.generators(q, conjugate=True), (0, 0))
    k = closure.level
    assert closure.rank == 2
    digits = k + 2  # the chart ball of radius p^-k is q + p^(k+1) on the surface
    m = p**digits
    start = tuple(v % m for v in q)
    seen, frontier = {start}, [start]
    while frontier:
        new = []
        for z in frontier:
            for a in range(3):
                w = tuple(surface.involution(list(z), a, m))
                if w not in seen:
                    seen.add(w)
                    new.append(w)
        frontier = new
    ball = [z for z in seen if all((a - b) % p ** (k + 1) == 0 for a, b in zip(z, q))]
    _, free = surface.chart(q)
    assert len(ball) == p ** (2 * (digits - k - 1))  # every class of the ball
    assert len({tuple(z[a] for a in free) for z in ball}) == len(ball)


def test_commutators_lower_the_level_on_a_wehler_surface():
    p, q = 3, [3368, 6524, 1701]  # a lift found by `run("wehler", 3, 2, [4])`
    surface = oc.Surface(oc.wehler(4), p)
    maps = surface.generators(q)
    assert OrbitClosure(p, oc.K, maps, (0, 0), depth=1).levels == (1, 3)
    closure = OrbitClosure(p, oc.K, maps, (0, 0))
    assert closure.levels == (1, 2) and closure.words[1] == "[g1, g2]"
