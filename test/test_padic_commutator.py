import numpy as np
from diffrax._padic_flow.polynomial import evaluate_exact
from diffrax.padic_flow import commutator, lie_bracket


p, digits = 5, 6
m = p**digits
units = [(1, 0), (0, 1)]
identity = [{(1, 0): 1}, {(0, 1): 1}]
g = [{(1, 0): 1, (0, 0): 5, (0, 2): 5}, {(0, 1): 1, (1, 0): 25}]
h = [{(1, 0): 1, (0, 1): 5, (1, 1): 10}, {(0, 1): 1, (0, 0): 5, (2, 0): 5}]


def _apply(f, x):
    return [evaluate_exact(poly, x, m) for poly in f]


def _minus_identity(f, modulus):
    out = []
    for poly, unit in zip(f, units):
        poly = {**poly, unit: poly.get(unit, 0) - 1}
        out.append({e: v % modulus for e, v in poly.items() if v % modulus})
    return out


def _add(total, f, modulus):
    for out, poly in zip(total, f):
        for e, v in poly.items():
            out[e] = (out.get(e, 0) + v) % modulus
    return [{e: v for e, v in poly.items() if v} for poly in total]


def test_commutator_on_points():
    c = commutator(p, digits, g, h)
    for x in np.random.default_rng(0).integers(0, m, (50, 2)).tolist():
        assert _apply(c, _apply(h, _apply(g, x))) == _apply(g, _apply(h, x))


def test_commutator_of_commuting_maps():
    assert commutator(p, digits, g, g) == identity
    assert commutator(p, digits, g, identity) == identity


def test_commutator_is_minus_the_bracket():
    # c = 1, so c - id = -[g - id, h - id] modulo p^3.
    X, Y = _minus_identity(g, m), _minus_identity(h, m)
    c = _minus_identity(commutator(p, 3, g, h), p**3)
    assert _add(c, lie_bracket(p, 3, X, Y), p**3) == [{}, {}]


def test_bracket_is_a_lie_bracket():
    X, Y = _minus_identity(g, m), _minus_identity(h, m)
    Z = [{(2, 0): 5}, {(1, 1): 5, (0, 0): 5}]
    assert _add(lie_bracket(p, digits, X, Y), lie_bracket(p, digits, Y, X), m) == [
        {},
        {},
    ]
    total = [{}, {}]
    for a, b, c in [(X, Y, Z), (Y, Z, X), (Z, X, Y)]:
        total = _add(total, lie_bracket(p, digits, a, lie_bracket(p, digits, b, c)), m)
    assert total == [{}, {}]
