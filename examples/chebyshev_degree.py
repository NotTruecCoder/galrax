"""Recover the degree n of a Chebyshev polynomial from x and T_n(x) modulo p^k, the
secret of Chebyshev key exchange over Z/p^k.
"""

import random

import jax


jax.config.update("jax_enable_x64", True)

from diffrax import padic_flow as pf  # noqa: E402


def chebyshev_degree(x: int, y: int, p: int, k: int) -> int | None:
    """The smallest `n >= 0` with `T_n(x) = y mod p^k`, or `None`: the first zero of
    `T_n - y` along the recurrence `T_{n+2} = 2x T_{n+1} - T_n`, `T_0 = 1`, `T_1 = x`.
    Any `x` works, `x = +-1 mod p` included.
    """
    return pf.recurrence_times(p, k, [-1, 2 * x], [1, x], y).first()


if __name__ == "__main__":
    rng = random.Random(0)
    p, k = 101, 9
    modulus = p**k

    def chebyshev(n, x):
        low, high = 1, x  # T_n, T_{n+1} by doubling
        for bit in bin(n)[2:]:
            middle = (2 * low * high - x) % modulus
            if bit == "0":
                low, high = (2 * low * low - 1) % modulus, middle
            else:
                low, high = middle, (2 * high * high - 1) % modulus
        return low

    for x in (rng.randrange(modulus), 1 + p * rng.randrange(modulus)):
        secret = rng.getrandbits(60)
        found = chebyshev_degree(x, chebyshev(secret, x), p, k)
        same = chebyshev(found, x) == chebyshev(secret, x)
        print(f"x = {x % p} mod p: secret {secret}, found {found}, same T: {same}")
