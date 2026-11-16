"""Set-up and query times of p-adic jump-ahead: `PAdicMap.iterate` on Z_p (scalar
and batched), an `Orbit` on Z_p^2, and `matrix_power_index` for Chebyshev degrees.

    python benchmarks/padic_jump.py
"""

import time

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

from diffrax.padic_flow import matrix_power_index, PAdicMap  # noqa: E402


def timed(label, fn, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    print(f"{label:48s} {(time.perf_counter() - start) / repeat * 1e3:10.2f} ms")
    return out


if __name__ == "__main__":
    n = np.random.default_rng(0).integers(0, 2**62, 10_000)
    for level in (16, 30, 40):
        mapping = PAdicMap(2, [1, 3, 2])
        timed(f"Z_2, 2^{level}: set-up", lambda: mapping.iterate(5, 0, level))
        timed(
            f"Z_2, 2^{level}: one query", lambda: mapping.iterate(5, 10**18, level), 100
        )
        mapping.iterate(5, n, level)  # warm-up: eager JAX compiles per shape
        timed(
            f"Z_2, 2^{level}: 10^4 queries batched",
            lambda: mapping.iterate(5, n, level),
            5,
        )

    plane = PAdicMap(
        3, [{(1, 0): 1, (0, 1): 1, (0, 0): 1}, {(1, 0): 1, (0, 0): 3, (2, 0): 3}]
    )
    timed("Z_3^2, 3^4: set-up", lambda: plane.iterate((1, 2), 0, 4))
    plane.iterate((1, 2), n, 4)
    timed("Z_3^2, 3^4: 10^4 queries batched", lambda: plane.iterate((1, 2), n, 4), 3)

    p, k, x = 101, 9, 123456789
    timed(
        "Chebyshev degree, p = 101, k = 9 (set-up + query)",
        lambda: matrix_power_index(p, k, [[0, 1], [-1, 2 * x]], (1, x), 42),
    )
