import pathlib
import random
import sys
from fractions import Fraction

import pytest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from adelic_sieve import henon_cases, quadratic_cases, solve  # noqa: E402
from galrax.padic import henon_zeros, map_zeros, recurrence_zeros  # noqa: E402
from galrax.padic.adelic import (  # noqa: E402
    depth_record,
    exact_zeros,
    henon_real,
    real_record,
    reversed_recurrence,
    stepper,
    verdict,
)
from nonlinear_skolem import exact_hits  # noqa: E402


def test_nilpotent_branch_bounds_the_zeros():
    # x -> x^2 from 3: x_n = 3^(2^n) tends 3-adically to the attracting point 0.
    step = stepper([{(2,): 1}])
    record = depth_record(step, (3,), 9, 3, "nilpotent", "forward")
    assert record["cover"] == [] and record["k"] == 3
    assert verdict([record], [1], 10) is record


def test_branches_without_an_etale_prime():
    cases = [c for c in quadratic_cases(2, random.Random(0)) if c[2]]
    assert len(cases) == 4  # two per dimension
    for case in cases:
        if case[1] == 2:  # no etale prime below 200 (for d = 3, below 100 only)
            assert not solve(case, "sieve")["certified"]
        result = solve(case, "abs")
        assert result["certified"] and result["by"] in ("nilpotent", "mixed")


def test_real_place_agrees_with_exact_orbit():
    how, zeros, period = henon_real(1, [0, 0, 0], (0, 1), 0)  # (0, 1) <-> (1, 0)
    assert (how, zeros, period) == ("periodic", [0], 2)  # x_n = 0 at every even n
    for case in henon_cases(40, random.Random(1)):
        _, c, P, start, target = case
        how, zeros, period = henon_real(c, P, start, target)
        if how == "periodic":  # every hit recurs with the period
            hits = exact_hits(c, P, start, target, 2 * period)[0]
            assert hits == [z + k * period for k in range(-3, 3) for z in zeros
                            if abs(z + k * period) <= 2 * period]
        if how == "escape":
            near = exact_hits(c, P, start, target, 12)[0]
            assert [n for n in zeros if abs(n) <= 12] == near


def test_reversed_recurrence():
    r, u = (3, -2, 4), (1, 0, 2)
    rb, s = reversed_recurrence(r, u)
    v = [Fraction(x) for x in u]  # u_{-m} by the backward recurrence
    for _ in range(8):
        v.insert(0, (v[2] - 3 * v[1] + 2 * v[0]) / 4)
    S = list(s)
    while len(S) < 9:
        S.append(sum(a * b for a, b in zip(rb, reversed(S[-3:]))))
    assert S == [4**m * v[8 - m] for m in range(9)]


def test_real_place_for_recurrences():
    r, u = (10, 1), (1, 0)  # zero at n = 1, then |u_n| grows
    record = real_record(r, u, "forward")
    assert record is not None and record["T"] >= 1
    assert [n for n in exact_zeros(list(r), list(u), 60) if n >= 0] == [1]
    assert real_record((0, 1), (1, 0), "forward") is None  # no dominant coefficient


def test_recurrence_over_z(monkeypatch):
    case = ("recurrence", 2, (1, 1), (2, 3))  # Fibonacci from F_3: zero at n = -3
    a, b = solve(case, "sieve"), solve(case, "abs")
    assert a["certified"] and b["certified"] and a["zeros"] == b["zeros"] == [-3]
    # u_n = u_{n-1} + 2 u_{n-2}: singular at 2, so n < 0 needs the reversed orbit
    case = ("recurrence", 2, (1, 2), (1, -1))
    # u_n = (-1)^n: the coefficient of the root 2 is 0, so n >= 0 needs the prime 2,
    # while n < 0 (roots -2 and 1) has a dominant root
    assert solve(case, "abs")["by"] == "mixed / real"
    monkeypatch.setenv("ABS_REAL", "0")  # the cone alone leaves both sides to p = 2
    a, b = solve(case, "sieve"), solve(case, "abs")
    assert b["certified"] and b["by"].startswith("mixed") and b["zeros"] == a["zeros"] == []


def test_dominant_roots_bound_the_zeros():
    pytest.importorskip("flint")
    from galrax.padic.adelic import dominant
    from galrax.padic.baker import complex_pair

    assert dominant((1, 1), (0, 1)) == 1  # Fibonacci: the zero n = 0 only
    assert dominant((0, -1), (1, 0)) is None and complex_pair((0, -1), (1, 0)) is None  # i^n
    # (x - 1)(x^2 - 2x + 5) from (1, 2, 3): the reduction gives no bound here
    assert complex_pair((3, -7, 5), (1, 2, 3)) is None
    rng, bounds = random.Random(3), {"real": 0, "pair": 0}
    while min(bounds.values()) < 5:
        k = rng.randint(3, 6)
        r = tuple(rng.randint(-9, 9) for _ in range(k))
        u = tuple(rng.randint(-9, 9) for _ in range(k))
        if not r[-1]:
            continue
        N, kind = dominant(r, u), "real"
        if N is None:
            N, kind = complex_pair(r, u), "pair"
        if N is not None:
            bounds[kind] += 1
            assert not [n for n in exact_zeros(list(r), list(u), N + 100) if n >= N], (r, u)


def test_public_zero_sets():
    pytest.importorskip("flint")
    import jax

    jax.config.update("jax_enable_x64", True)
    # u_n = u_{n-1} + 2 u_{n-2} from (0, 1): u_n = (2^n - (-1)^n) / 3, zero at n = 0 only
    found = recurrence_zeros((1, 2), (0, 1))
    assert found.certified and found.zeros == (0,)
    # x -> x^2 + 1 from 0: 0, 1, 2, 5, ...; modulo 3 the cycle is the fixed point 2,
    # so the target 3 can only be hit before it, and it is not
    found = map_zeros([{(2,): 1, (0,): 1}], (0,), 3)
    assert found.certified and found.zeros == ()
    assert henon_zeros(1, [0, 0, 0], (0, 1), 0) == henon_zeros(1, (0, 0, 0), [0, 1], 0)
    assert henon_zeros(1, [0, 0, 0], (0, 1), 0).period == 2
    with pytest.raises(ValueError):
        recurrence_zeros((1, 0), (1, 1))
    assert recurrence_zeros((2,), (1,)).zeros == ()  # order 1: u_n = 2^n
