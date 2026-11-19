import jax
import pytest
from galrax.padic import (
    charpoly_to_r,
    hypersurface_times,
    loop_exit,
    map_zeros,
    recurrence_zeros,
    value_times,
)
from galrax.padic.adelic import stepper


jax.config.update("jax_enable_x64", True)


def test_value_times():
    pytest.importorskip("flint")
    fib = ((1, 1), (0, 1))
    assert value_times(*fib, 1).zeros == (-1, 1, 2)  # F_{-1} = F_1 = F_2 = 1
    assert value_times(*fib, 2).zeros == (-3, 3)
    assert value_times(*fib, 0) == recurrence_zeros(*fib)
    with pytest.raises(ValueError):
        value_times((1,), (3,), 3)  # u_n = 3
    assert charpoly_to_r([-1, -1, 1]) == (1, 1)  # x^2 - x - 1


def test_hypersurface_times_against_brute_force():
    # a quadratic map of Z^2 and g = x - y
    F = [{(0, 0): 1, (0, 1): -2, (1, 1): -2, (0, 2): 1}, {(0, 0): -1, (1, 0): -1, (0, 1): -1}]
    found = hypersurface_times(F, (1, 1), {(1, 0): 1, (0, 1): -1})
    assert found.certified
    step, x, brute = stepper(F), (1, 1), []
    for n in range(21):  # the orbit grows doubly exponentially: 21 exact steps
        if x[0] == x[1]:
            brute.append(n)
        x = step(x)
    assert brute == [n for n in found.zeros if n <= 20] == [0, 2]


def test_loop_exit():
    F = [{(1, 0): 1, (0, 0): 1}, {(0, 1): 2}]  # (i, x) -> (i + 1, 2 x) from (0, 1)
    assert loop_exit(F, (0, 1), {(0, 1): 1, (1, 0): -2, (0, 0): -22}) == 5  # 2^5 = 2*5 + 22
    assert loop_exit(F, (0, 1), {(0, 1): 1, (0, 0): -48}) is None  # certified never


def test_sage_inputs():
    pytest.importorskip("sage.all")
    pytest.importorskip("flint")
    from sage.all import BinaryRecurrenceSequence, PolynomialRing, ZZ

    R = PolynomialRing(ZZ, "x,y")
    x, y = R.gens()
    plain = map_zeros([{(0, 1): 1}, {(1, 1): 1, (0, 0): 1}], (1, 1), 7)
    assert map_zeros([y, x * y + 1], (ZZ(1), ZZ(1)), ZZ(7)) == plain
    assert plain.certified and plain.zeros == (4,)
    B = BinaryRecurrenceSequence(1, 1, 0, 1)
    assert recurrence_zeros(B) == recurrence_zeros((1, 1), (0, 1))
    assert value_times((ZZ(1), ZZ(1)), (ZZ(0), ZZ(1)), ZZ(2)).zeros == (-3, 3)
    assert charpoly_to_r(PolynomialRing(ZZ, "t")([-1, -1, 1])) == (1, 1)
    F = [{(1, 0): 1, (0, 0): 1}, {(0, 1): 2}]
    assert loop_exit(F, (0, 1), y - 2 * x - 22) == 5


def test_exit_found_without_a_certificate():
    # (i, s) -> (i + 1, s + i): s_n = n (n - 1) / 2 = 10 at n = 5 and at n = -4, which
    # the forward count cannot exclude; the exact search still gives the exit
    F = [{(0, 0): 1, (1, 0): 1}, {(0, 1): 1, (1, 0): 1}]
    guard = {(0, 1): 1, (0, 0): -10}
    assert loop_exit(F, (0, 0), guard) == 5
    found = hypersurface_times(F, (0, 0), guard)
    assert not found.certified and found.known == (5,) and found.searched >= 120
    found = recurrence_zeros((1, 1), (2, 3))
    assert found.known == found.zeros == (-3,) and found.searched == 200
