"""Recover the degree n of a Chebyshev polynomial from x and T_n(x) modulo p^k, the
secret of Chebyshev key exchange over Z/p^k.
"""

import random


def chebyshev_degree(x: int, y: int, p: int, k: int) -> int | None:
    """The smallest `n >= 0` with `T_n(x) = y mod p^k`, or `None`, for `x != +-1 mod p`.

    `(T_{n+1}, T_n) = M^n (x, 1)` with `M = [[2x, -1], [1, 0]]`, so `T_n(x) mod p^m`
    depends on `n` modulo the order `P_m` of `M` modulo `p^m`, and `P_{m+1}` is
    `P_m` or `p P_m`: the solutions are found modulo `p` and lifted a digit at a time.
    """
    one = [[1, 0], [0, 1]]

    def power(n, modulus):
        def mul(a, b):
            return [
                [(a[i][0] * b[0][j] + a[i][1] * b[1][j]) % modulus for j in range(2)]
                for i in range(2)
            ]

        result, base = one, [[2 * x % modulus, -1 % modulus], [1, 0]]
        while n:
            if n & 1:
                result = mul(result, base)
            base, n = mul(base, base), n >> 1
        return result

    def chebyshev(n, modulus):
        row = power(n, modulus)[1]
        return (row[0] * x + row[1]) % modulus

    period = 1
    while power(period, p) != one:
        period += 1
    found = [n for n in range(period) if chebyshev(n, p) == y % p]
    for m in range(2, k + 1):
        step = period if power(period, p**m) == one else p * period
        found = [
            n + period * i
            for n in found
            for i in range(step // period)
            if chebyshev(n + period * i, p**m) == y % p**m
        ]
        period = step
    return min(found, default=None)


if __name__ == "__main__":
    rng = random.Random(0)
    p, k = 101, 40
    modulus = p**k
    x, secret = (
        rng.randrange(2, p - 1) + p * rng.randrange(p ** (k - 1)),
        rng.getrandbits(200),
    )

    def chebyshev(n):
        low, high = 1, x  # T_n, T_{n+1} by doubling
        for bit in bin(n)[2:]:
            middle = (2 * low * high - x) % modulus
            if bit == "0":
                low, high = (2 * low * low - 1) % modulus, middle
            else:
                low, high = middle, (2 * high * high - 1) % modulus
        return low

    found = chebyshev_degree(x, chebyshev(secret), p, k)
    print(f"secret degree {secret}\nrecovered     {found}")
    print("T_found(x) = T_secret(x):", chebyshev(found) == chebyshev(secret))
