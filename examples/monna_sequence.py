"""u_n = monna(T^n(0)) for the full-period generator T(x) = 2x^2 + 3x + 1 on Z_2: T is
minimal and keeps the Haar measure, so u_n is equidistributed in [0, 1], and the n-th
term costs the same for n = 5 and n = 10^100.
"""

from galrax.padic import PAdicMap
from galrax.ultrametric import monna


T, digits = PAdicMap(2, [1, 3, 2]), 32
u = [float(monna(T.iterate(0, n, digits), 2, digits)) for n in range(4096)]
print("first terms:", [round(v, 4) for v in u[:6]])
print("mean, share below 1/3:", sum(u) / len(u), sum(v < 1 / 3 for v in u) / len(u))
far = float(monna(T.iterate(0, 10**100 + 7, digits), 2, digits))
print("u_(10^100 + 7) = u_7 (the period is 2^32):", far, far == u[7])
