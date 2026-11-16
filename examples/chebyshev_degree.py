"""Recover the degree n of a Chebyshev polynomial from x and T_n(x) modulo p^k, the
secret of Chebyshev key exchange over Z/p^k.
"""

import random

import jax


jax.config.update("jax_enable_x64", True)

from diffrax import padic_flow as pf  # noqa: E402


def chebyshev_degree(x: int, y: int, p: int, k: int) -> int | None:
    """The smallest `n >= 0` with `T_n(x) = y mod p^k`, or `None`.

    `(T_{n+1}, T_n) = M^n (x, 1)` with `M = [[2x, -1], [1, 0]]`. For `N` the order of
    `M` modulo `p` (modulo `4` for `p = 2`), `M^N` is the identity modulo `p`, so it
    is the time-one map of a flow `phi`, and `T_{r + N q}(x) = (M^r)_1 phi_q(x, 1)`:
    the `q` that solve it are an event set of the flow, for each `r < N` with
    `T_r(x) = y mod p`. Any `x` works, `x = +-1 mod p` included.
    """
    big = p ** (3 * k)

    def power(n, modulus):
        def mul(a, b):
            return [
                [(a[i][0] * b[0][j] + a[i][1] * b[1][j]) % modulus for j in range(2)]
                for i in range(2)
            ]

        result, base = [[1, 0], [0, 1]], [[2 * x % modulus, -1 % modulus], [1, 0]]
        while n:
            if n & 1:
                result = mul(result, base)
            base, n = mul(base, base), n >> 1
        return result

    e = 2 if p == 2 else 1
    N = 1
    while power(N, p**e) != [[1, 0], [0, 1]]:
        N += 1
    (a, b), (c, d) = power(N, big)
    flow = pf.PAdicFlow.from_time_one(
        p, k, [{(1, 0): a, (0, 1): b}, {(1, 0): c, (0, 1): d}]
    )
    best = None
    for r in range(N):
        _, (u, v) = power(r, p**k)
        if (u * x + v - y) % p:
            continue
        event = pf.PAdicEvent({(1, 0): u, (0, 1): v, (0, 0): -y})
        q, _ = flow.first_event(event, (x % p**k, 1))
        if q is not None and (best is None or r + N * q < best):
            best = r + N * q
    return best


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
