"""Markoff lattice scanner: Smith forms of the tangent lattice of the Markoff
action x^2 + y^2 + z^2 = xyz + D.

For a point q in X_D(Z_p) we collect the generators of the rotations
rot_a^{n_a} (n_a = order of rot_a mod p), which are = id mod p on {a = q_a mod p},
at q itself (local) and transported back from h(q) along reduced words h (global):
X_{h^-1 g h}(q) = Dh(q)^-1 X_g(h q).  The Z_p-span is a lattice in T_q with
Smith exponents k1 <= k2 (k3 must be >= W: tangency check).

Usage: python markoff_smith.py p depth lifts [D1,D2,...]

Lifts whose global exponents are not (1, 1) are printed with the finite orbit they
are congruent to modulo p^2. The 7830 lifts of the paper, of which 28 have (2, 2):
18 near (0, 0, +-sqrt D), 7 near the Cayley cubic, 2 near the orbit of (1, 1, 1) on
X_2 and 1 near that of (1, sqrt 2, 0) on X_3 (about 20 minutes in all):

    python markoff_smith.py 5 8 3
    python markoff_smith.py 7 7 2 $(seq -s, 0 41)
    python markoff_smith.py 11 6 1 0,1,2,3,4,5,6,8,15,30,40,60,121,242,7,9
"""

import random
import sys
from collections import Counter, defaultdict


W = 16  # p-adic digits


def val(a, p, cap):
    a %= p**cap
    if a == 0:
        return cap
    v = 0
    while a % p == 0:
        a //= p
        v += 1
    return v


def mul(A, B, P):
    return [
        [(A[i][0] * B[0][j] + A[i][1] * B[1][j]) % P for j in range(2)]
        for i in range(2)
    ]


def mpow(A, n, P):
    R = [[1, 0], [0, 1]]
    while n:
        if n & 1:
            R = mul(R, A, P)
        A = mul(A, A, P)
        n >>= 1
    return R


def rot(t, P):
    return [[(t * t - 1) % P, -t % P], [t % P, P - 1]]


def order_mod_p(t, p):
    R, M, n = rot(t, p), rot(t, p), 1
    while M != [[1, 0], [0, 1]]:
        M, n = mul(M, R, p), n + 1
    return n


def log_unipotent(M, p):
    """log M for M = I mod p, result mod p^W."""
    E = 4
    P = p ** (W + E)
    N = [[(M[0][0] - 1) % P, M[0][1] % P], [M[1][0] % P, (M[1][1] - 1) % P]]
    out = [[0, 0], [0, 0]]
    Nj = [[1, 0], [0, 1]]
    PW = p**W
    for j in range(1, W + E + 2):
        Nj = mul(Nj, N, P)
        e, u = 0, j
        while u % p == 0:
            u //= p
            e += 1
        c = pow(u, -1, PW) * (-1) ** (j - 1)
        for i in range(2):
            for k in range(2):
                assert Nj[i][k] % p**e == 0
                out[i][k] = (out[i][k] + c * (Nj[i][k] // p**e)) % PW
    return out


_cache = {}


def generator(q, a, p):
    """Value at q of the generator of rot_a^{n_a}."""
    b, c = (a + 1) % 3, (a + 2) % 3
    key = (a, q[a])
    if key not in _cache:
        n = order_mod_p(q[a] % p, p)
        _cache[key] = log_unipotent(mpow(rot(q[a], p ** (W + 4)), n, p ** (W + 4)), p)
    A = _cache[key]
    PW = p**W
    X = [0, 0, 0]
    X[b] = (A[0][0] * q[b] + A[0][1] * q[c]) % PW
    X[c] = (A[1][0] * q[b] + A[1][1] * q[c]) % PW
    return X


def involution(q, a, P):
    b, c = (a + 1) % 3, (a + 2) % 3
    r = list(q)
    r[a] = (q[b] * q[c] - q[a]) % P
    return r


def jac(q, a, P):
    """Jacobian of s_a at q; it is its own inverse."""
    b, c = (a + 1) % 3, (a + 2) % 3
    J = [[int(i == j) for j in range(3)] for i in range(3)]
    J[a][a] = P - 1
    J[a][b] = q[c] % P
    J[a][c] = q[b] % P
    return J


def mul3(A, B, P):
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) % P for j in range(3)]
        for i in range(3)
    ]


def apply3(A, v, P):
    return [sum(A[i][k] * v[k] for k in range(3)) % P for i in range(3)]


def smith(vecs, p):
    """Smith exponents of the Z_p-span of 3-vectors mod p^W."""
    vecs = [list(v) for v in vecs]
    exps, cols = [], [0, 1, 2]
    PW = p**W
    while cols and vecs:
        best = min(
            ((val(v[j], p, W), i, j) for i, v in enumerate(vecs) for j in cols),
            default=None,
        )
        k, i, j = best
        exps.append(k)
        if k >= W:
            break
        w = vecs.pop(i)
        uinv = pow(w[j] // p**k, -1, PW)
        for v in vecs:
            f = (v[j] // p**k) * uinv % PW
            for t in range(3):
                v[t] = (v[t] - f * w[t]) % PW
        cols.remove(j)
    return exps + [W] * (3 - len(exps))


def surface_points(D, p):
    pts = []
    for x in range(p):
        for y in range(p):
            for z in range(p):
                if (x * x + y * y + z * z - x * y * z - D) % p:
                    continue
                grad = ((2 * x - y * z) % p, (2 * y - x * z) % p, (2 * z - x * y) % p)
                if any(grad):
                    pts.append((x, y, z))
    return pts


def lift(pt, D, p, rng):
    P = p**W
    x = list(pt)
    grad = [
        (2 * x[0] - x[1] * x[2]) % p,
        (2 * x[1] - x[0] * x[2]) % p,
        (2 * x[2] - x[0] * x[1]) % p,
    ]
    j = next(i for i in range(3) if grad[i])
    for i in range(3):
        if i != j:
            x[i] = (x[i] + p * rng.randrange(p ** (W - 1))) % P
    for _ in range(W + 2):
        F = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - x[0] * x[1] * x[2] - D) % P
        b, c = [i for i in range(3) if i != j]
        dF = (2 * x[j] - x[b] * x[c]) % P
        x[j] = (x[j] - F * pow(dF, -1, P)) % P
    assert (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - x[0] * x[1] * x[2] - D) % P == 0
    return x


def lattice(q, p, depth):
    """Smith exponents of the span of the transported generators, for words of
    length <= depth.
    """
    P = p**W
    out = {}
    vecs = []
    stack = [(q, [[int(i == j) for j in range(3)] for i in range(3)], -1, 0)]
    by_len = defaultdict(list)
    while stack:
        r, inv, last, L = stack.pop()
        for a in range(3):
            by_len[L].append(apply3(inv, generator(r, a, p), P))
        if L < depth:
            for a in range(3):
                if a != last:
                    stack.append(
                        (involution(r, a, P), mul3(inv, jac(r, a, P), P), a, L + 1)
                    )
    for L in range(depth + 1):
        vecs += by_len[L]
        out[L] = smith(vecs, p)
    return out


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def chebyshev(n, t, P):
    """a^n + a^-n mod P for t = a + 1/a, by doubling."""
    low, high = 2, t % P
    for bit in bin(n)[2:]:
        middle = (low * high - t) % P
        if bit == "0":
            low, high = (low * low - 2) % P, middle
        else:
            low, high = middle, (high * high - 2) % P
    return low


def orbit(start, P):
    """The orbit of a point under the three involutions, modulo P."""
    seen, stack = set(), [tuple(v % P for v in start)]
    while stack:
        r = stack.pop()
        if r not in seen:
            seen.add(r)
            stack += [tuple(involution(r, a, P)) for a in range(3)]
    return seen


def finite_orbit(q, D, p):
    """The finite orbit that q is congruent to modulo p^2, or None.

    a + 1/a with a a root of unity of order prime to p is the t with
    t = a^(p^2) + a^-(p^2) mod p^2 (a Teichmueller lift).
    """
    P = p * p
    q, D = tuple(v % P for v in q), D % P
    if sorted(v == 0 for v in q) == [False, True, True]:
        return "(0,0,+-sqrt D)"
    if D == 4 and all(chebyshev(P, v, P) == v for v in q):
        return "Cayley cubic X_4"
    if D == 2 and q in orbit((1, 1, 1), P):
        return "orbit of (1,1,1) on X_2"
    roots = [s for s in range(P) if s * s % P == 2]
    if D == 3 and any(q in orbit((1, s, 0), P) for s in roots):
        return "orbit of (1,sqrt 2,0) on X_3"
    return None


def run(p, Ds, depth, lifts, seed=0):
    rng = random.Random(seed)
    print(f"\n=== p = {p}, words of length <= {depth}, {lifts} lift(s) per point ===")
    total, kinds = Counter(), Counter()
    for D in Ds:
        Dl = (D + p**2 * rng.randrange(p ** (W - 2))) % p**W
        tag = "D=0 mod p^2" if Dl % p**2 == 0 else f"((D-4)/p)={legendre(Dl - 4, p):+d}"
        loc, glob = Counter(), Counter()
        for pt in surface_points(Dl, p):
            for _ in range(lifts):
                q = lift(pt, Dl, p, rng)
                res = lattice(q, p, depth)
                assert res[depth][2] >= W - 2, ("not tangent", res)
                loc[tuple(res[0][:2])] += 1
                glob[tuple(res[depth][:2])] += 1
                if tuple(res[depth][:2]) != (1, 1):
                    kind = finite_orbit(q, Dl, p)
                    kinds[kind] += 1
                    near = tuple(v % p**2 for v in q)
                    print(f"  D={D} q={near} mod p^2 {tuple(res[depth][:2])}: {kind}")
        an_loc = sum(c for (a, b), c in loc.items() if a < b)
        an_glob = sum(c for (a, b), c in glob.items() if a < b)
        n = sum(loc.values())
        total["points"] += n
        total["aniso_local"] += an_loc
        total["aniso_global"] += an_glob
        print(
            f"D={D:4d} {tag:16s} points={n:4d}  "
            f"local aniso={an_loc:4d}  global aniso={an_glob:4d}  "
            f"global (k1,k2): {dict(sorted(glob.items()))}"
        )
        sys.stdout.flush()
    print("TOTAL", dict(total))
    print("EXCEPTIONAL", dict(kinds))


if __name__ == "__main__":
    p = int(sys.argv[1])
    depth = int(sys.argv[2])
    lifts = int(sys.argv[3])
    Ds = range(p**2) if len(sys.argv) < 5 else [int(d) for d in sys.argv[4].split(",")]
    run(p, Ds, depth, lifts)
