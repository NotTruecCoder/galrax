import pathlib
import random
from fractions import Fraction
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from adelic_sieve import (  # noqa: E402
    depth_record,
    henon_cases,
    henon_real,
    quadratic_cases,
    real_record,
    reversed_recurrence,
    solve,
    stepper,
    verdict,
)
from nonlinear_skolem import exact_hits  # noqa: E402
from skolem import exact_zeros  # noqa: E402


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
    for case in henon_cases(40, random.Random(1)):
        _, c, P, start, target = case
        how, zeros = henon_real(c, P, start, target)
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


def test_recurrence_over_z():
    case = ("recurrence", 2, (1, 1), (2, 3))  # Fibonacci from F_3: zero at n = -3
    a, b = solve(case, "sieve"), solve(case, "abs")
    assert a["certified"] and b["certified"] and a["zeros"] == b["zeros"] == [-3]
    # u_n = u_{n-1} + 2 u_{n-2}: singular at 2, so n < 0 needs the reversed orbit
    case = ("recurrence", 2, (1, 2), (1, -1))
    a, b = solve(case, "sieve"), solve(case, "abs")
    assert b["certified"] and b["by"].startswith("mixed") and b["zeros"] == a["zeros"] == []
