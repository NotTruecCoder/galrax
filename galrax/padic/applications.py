"""Questions that reduce to one ABS zero set: values of a recurrence, hits of a
hypersurface along an orbit (DML), and the exit time of a polynomial loop.

Not covered: `u_n = v_m` (two indices, not one orbit), positivity, `x_n = x_m`.
"""

from .adelic import orbit_zeros, recurrence_zeros, ZeroSet


def charpoly_to_r(P):
    """`r` of `u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}` from its monic integer
    characteristic polynomial `P`, given by coefficients `[P_0, ..., P_k]` or as a
    Sage polynomial: `r_i = -P_{k-1-i}`."""
    k = int(P.degree()) if hasattr(P, "degree") else len(P) - 1
    if k < 1 or P[k] != 1:
        raise ValueError("Need a monic polynomial of degree >= 1.")
    return tuple(-int(P[k - 1 - i]) for i in range(k))


def value_times(r, u, c, **kwargs) -> ZeroSet:
    """Every `n` in `Z` with `u_n = c`, for the recurrence of `recurrence_zeros`.
    `v_n = u_n - c` satisfies the recurrence of `(x - 1) P(x)`, of order `k + 1`."""
    r, u, c = tuple(int(v) for v in r), [int(v) for v in u], int(c)
    if not c:
        return recurrence_zeros(r, u, **kwargs)
    if len(r) != len(u):
        raise ValueError("Need len(r) == len(u).")
    u.append(sum(a * b for a, b in zip(r, reversed(u))))  # u_k
    v = tuple(x - c for x in u)
    if not any(v):
        raise ValueError(f"u_n = {c} for every n.")
    rv = (r[0] + 1, *(b - a for a, b in zip(r, r[1:])), -r[-1])
    return recurrence_zeros(rv, v, **kwargs)


def hypersurface_times(F, x0, g, **kwargs) -> ZeroSet:
    """Every `n >= 0` with `g(x_n) = 0` along the integer orbit of `x0` under the
    polynomial map `F`; `F` and `g` are `{exponents: coefficient}` or Sage polynomials."""
    return orbit_zeros(F, x0, g, **kwargs)


def loop_exit(F, x0, guard, **kwargs):
    """The number of steps of `while guard(x) != 0: x = F(x)` from `x0`, or `None`
    when the loop is certified never to exit; `RuntimeError` when ABS cannot tell."""
    found = hypersurface_times(F, x0, guard, **kwargs)
    if not found.certified:
        raise RuntimeError("not certified")
    return min(found.zeros, default=None)
