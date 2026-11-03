import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from .arithmetic import Residues, scan_horner, valuation


def parse_field(field, p: int, extra_variables: int = 0) -> list[dict]:
    """A polynomial vector field as one `{exponents: coefficient}` dictionary per
    component. One-dimensional fields may also be given as the coefficient list
    `[c_0, c_1, ...]` of `c_0 + c_1 x + ...`. Coefficients are integers or fractions
    whose denominators are not divisible by `p`. With `extra_variables = 1` the
    components are functions of one more variable (time) than there are components.
    """
    if isinstance(field, dict):
        field = [field]
    field = list(field)
    if len(field) == 0:
        raise ValueError("The vector field needs at least one component.")
    if not isinstance(field[0], dict):
        field = [{(i,): c for i, c in enumerate(field)}]
    first = next(iter(field[0]), None)
    if first is not None:
        d = len(np.atleast_1d(first))
    else:
        d = len(field) + extra_variables
    out = []
    for component in field:
        if not isinstance(component, dict):
            raise ValueError("Give every component as {exponents: coefficient}.")
        clean = {}
        for exponents, coefficient in component.items():
            exponents = tuple(int(e) for e in np.atleast_1d(exponents))
            if len(exponents) != d or min(exponents) < 0:
                raise ValueError(f"Exponents must be {d} non-negative integers.")
            coefficient = Fraction(coefficient)
            if coefficient.denominator % p == 0:
                raise ValueError(
                    "Coefficients must be p-adic integers: their denominators must "
                    "not be divisible by p."
                )
            if coefficient != 0:
                clean[exponents] = clean.get(exponents, 0) + coefficient
        out.append(clean)
    if len(out) + extra_variables != d:
        raise ValueError(
            f"Exponents have {d} entries, so {d - extra_variables} components are "
            f"needed; got {len(out)}."
        )
    return out


def parse_polynomial(poly, d: int, p: int) -> dict:
    """One polynomial in `d` variables, given as `{exponents: coefficient}` (or, for
    `d = 1`, as the coefficient list `[c_0, c_1, ...]`).
    """
    if not isinstance(poly, dict):
        poly = {(i,): c for i, c in enumerate(poly)}
    (parsed,) = parse_field([poly], p, extra_variables=d - 1) if poly else [{}]
    if parsed and len(next(iter(parsed))) != d:
        raise ValueError(f"The polynomial must be in {d} variables.")
    return parsed


def field_valuation(field: list[dict], p: int, cap: int) -> int:
    """The smallest valuation of a coefficient of `field`, capped at `cap`."""
    values = [
        valuation(c.numerator, p, cap)
        for component in field
        for c in component.values()
    ]
    return min(values, default=cap)


def reduce(field: list[dict], modulus: int) -> list[dict]:
    out = []
    for component in field:
        out.append(
            {
                e: c.numerator * pow(c.denominator, -1, modulus) % modulus
                for e, c in component.items()
            }
        )
    return out


def _derivative(poly: dict, i: int, modulus: int) -> dict:
    out = {}
    for e, c in poly.items():
        if e[i]:
            lowered = e[:i] + (e[i] - 1,) + e[i + 1 :]
            out[lowered] = (out.get(lowered, 0) + c * e[i]) % modulus
    return out


def _multiply_add(out: dict, a: dict, b: dict, modulus: int) -> None:
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(x + y for x, y in zip(ea, eb))
            out[e] = (out.get(e, 0) + ca * cb) % modulus


def lie_series(
    field: list[dict], terms: int, modulus: int, start: list[dict] | None = None
) -> list[list[dict]]:
    """`[g, Dg, D^2 g, ...]`, `terms` entries, for the derivation
    `D = sum_i f_i d/dx_i` and the polynomials `g = start` (by default the coordinates
    `x_1, ..., x_d`). Coefficients are reduced modulo `modulus`.
    """
    d = len(next(iter(field[0]), ())) if field[0] else len(field)
    if start is None:
        current = [{tuple(int(j == i) for j in range(d)): 1} for i in range(d)]
    else:
        current = [{e: c % modulus for e, c in g.items()} for g in start]
    series = [current]
    for _ in range(terms - 1):
        nxt = []
        for g in current:
            out = {}
            for i, f in enumerate(field):
                _multiply_add(out, f, _derivative(g, i, modulus), modulus)
            nxt.append({e: c for e, c in out.items() if c})
        current = nxt
        series.append(current)
    return series


def degree(polys) -> int:
    return max((sum(e) for poly in polys for e in poly), default=0)


def to_dense(polys: list[dict], d: int, deg: int) -> np.ndarray:
    """Polynomials as an array of shape `(len(polys),) + (deg + 1,) * d`."""
    out = np.zeros((len(polys),) + (deg + 1,) * d, dtype=np.int64)
    for k, poly in enumerate(polys):
        for e, c in poly.items():
            out[(k,) + e] = c
    return out


def evaluate(residues: Residues, coefficients, x):
    """Values of dense polynomials at points.

    `coefficients` has shape `(batch or 1, *rest, deg + 1, ..., deg + 1)` with the last
    `d` axes indexing the exponents of `x_0, ..., x_{d-1}`; `x` has shape
    `(batch, d)`. Returns shape `(batch, *rest)`.
    """
    d = x.shape[-1]
    out = jnp.asarray(coefficients)
    for j in reversed(range(d)):
        point = x[:, j].reshape((-1,) + (1,) * (out.ndim - 2))
        out = scan_horner(residues, out, point)
    return out


def evaluate_exact(poly: dict, point, modulus: int) -> int:
    """`poly(point)` modulo `modulus`, with Python integers."""
    total = 0
    for e, c in poly.items():
        term = c
        for x, k in zip(point, e):
            term = term * pow(x, k, modulus) % modulus
        total += term
    return total % modulus


def taylor_shift(poly: dict, center, modulus: int) -> dict:
    """The coefficients of `y -> poly(center + y)`: the Hasse derivatives of `poly` at
    `center`, modulo `modulus`.
    """
    out = {}
    for e, c in poly.items():
        expansions = [
            [(j, math.comb(k, j) * pow(x, k - j, modulus)) for j in range(k + 1)]
            for x, k in zip(center, e)
        ]
        partial = {(): c % modulus}
        for options in expansions:
            partial = {
                key + (j,): value * weight % modulus
                for key, value in partial.items()
                for j, weight in options
            }
        for key, value in partial.items():
            out[key] = (out.get(key, 0) + value) % modulus
    return out
