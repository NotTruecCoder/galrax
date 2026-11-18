import pathlib
import random
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from adelic_sieve import (  # noqa: E402
    depth_record,
    henon_cases,
    henon_real,
    quadratic_cases,
    solve,
    stepper,
    verdict,
)
from nonlinear_skolem import exact_hits  # noqa: E402


def test_nilpotent_branch_bounds_the_zeros():
    # x -> x^2 from 3: x_n = 3^(2^n) tends 3-adically to the attracting point 0.
    step = stepper([{(2,): 1}])
    record = depth_record(step, (3,), 9, 3, "nilpotent")
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


def test_recurrence_agrees_with_sieve():
    case = ("recurrence", 2, (1, 1), (0, 1))  # Fibonacci: zero at n = 0 only
    a, b = solve(case, "sieve"), solve(case, "abs")
    assert a["certified"] and b["certified"]
    assert [n for n in a["zeros"] if n >= 0] == [n for n in b["zeros"] if n >= 0] == [0]
