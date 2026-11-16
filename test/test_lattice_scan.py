import pathlib
import random
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

import lattice_scan as scan  # noqa: E402
import markoff_smith as markoff  # noqa: E402


p, P = 5, 5**scan.W


def test_regression_on_a_fixed_seed():
    total, checked = scan.run(p, (0, 0, 0), [2, 4], 3, 1, sample=5, seed=0)
    assert dict(total) == {(1, 1): 37, (2, 2): 1}
    total, general = scan.run(p, (1, 2, 3), [0, 1, 2], 3, 1, sample=5, seed=0)
    assert dict(total) == {(1, 1): 71}
    assert set(checked) == set(general) == {"agree"}


def test_fields_vanish_on_finite_orbits():
    # (0, 0, 2) on x^2 + y^2 + z^2 + xyz = 4, and (1, 1, -1) on ... = 2: every
    # rotation has finite order there, so the fields of its powers are zero.
    for surface, q in (
        (scan.Surface(0, 0, 0, 4), [0, 0, 2]),
        (scan.Surface(0, 0, 0, 2), [1, 1, P - 1]),
    ):
        assert surface.equation(q, P) == 0
        assert all(surface.field(q, i, p) == [0, 0, 0] for i in range(3))
        exponents, _ = scan.lattice(surface, q, p, 2)
        assert exponents[2] == [scan.W] * 3


def test_minimal_action_on_a_two_two_point():
    surface, rng = scan.Surface(0, 0, 0, 4), random.Random(3)
    q = surface.lift([0, 0, 2], p, rng, near=2)
    exponents, found = scan.lattice(surface, q, p, 3)
    assert exponents[3][:2] == [2, 2]
    action = scan.minimal_action(surface, q, p, found, 2)
    assert action.level == 2
    digits = scan.K - action.level
    t = np.array(rng.sample(range(p**digits), 6), dtype=object).reshape(3, 2)
    z = action.chart(t)
    assert np.array_equal(action.coordinates(z), t)
    assert np.array_equal(action.chart(action.coordinates(z)), z)


def test_markoff_case_agrees():
    rng = random.Random(0)
    for D in (1, 3):
        surface = scan.Surface(0, 0, 0, D)
        for point in surface.points(p)[:8]:
            q = surface.lift(point, p, rng)
            ours, _ = scan.lattice(surface, q, p, 2)
            theirs = markoff.lattice([q[0], q[1], -q[2] % P], p, 2)
            assert ours[2][:2] == sorted(theirs[2])[:2]


def test_exponents_are_invariant_under_the_group():
    rng = random.Random(2)
    surface = scan.Surface(1, 2, 3, 2)
    for point in surface.points(p)[:6]:
        q = surface.lift(point, p, rng)
        exponents, _ = scan.lattice(surface, q, p, 3)
        for i in range(3):
            moved, _ = scan.lattice(surface, surface.involution(q, i, P), p, 3)
            assert moved[3] == exponents[3]
