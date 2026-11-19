"""Baker--Davenport at the real place, for a recurrence whose dominant roots are one
complex pair `l, conj(l)` with every other root smaller.

Write `u_n = 2 Re(c l^n) + w_n`, `|w_n| <= W rho^n`, `q = rho / |l| < 1`. A zero
gives `|cos(n theta + phi)| <= (W / 2|c|) q^n` (`theta = arg l`, `phi = arg c`), and
since `|cos y| >= (2 / pi) dist(y, pi/2 + pi Z)`,

    |n theta' - m + mu| <= A q^n,   theta' = theta / pi,  mu = phi / pi - 1/2,
                                    A = W / (4 |c|).

The linear form `Lambda = n Log(l / conj l) + Log(c / conj c) + (2m - 1) Log(-1)`
has `|Lambda| = 2 pi |n theta' - m + mu|`. Matveev's bound (Izv. Math. 64, 2000,
Corollary 2.3, in the form of Riffaut, arXiv:1710.03547, Theorem 2.11: for
`alpha_j` not all real, any fixed logarithms and `Lambda != 0`, with no linear
independence assumed, `log |Lambda| > -2^(6r+20) d^(2+r) A_1...A_r log(ed) log(eH)`,
`A_j >= max(h(alpha_j), |log alpha_j| / d, 0.16 / d)`, `H = max |b_j|`; here `r = 3`
and `d <= D`, which only weakens it), with its constant raised 10^6 times as a
margin, gives `n <= N0` when `Lambda != 0`; `Lambda = 0` happens for
at most one `n` (the recurrence is non-degenerate), and then
`n h(l / conj l) = h(c / conj c)`, bounded through Voutier's
`h(a) >= 2 / (d (log 3d)^3)`. Below `N0` the lemma of Baker--Davenport, in the form
of Dujella--Petho (Quart. J. Math. 49, 1998, Lemma 5a), cuts the range: for a
convergent `P/Q` of `theta'` and `eps = ||Q mu|| - N0 ||Q theta'|| > 0`, every
solution has `n < log(A Q / eps) / log(1 / q)`. The margin on Matveev's constant
changes the final bound only through `log N0`.

All of it is done in ball arithmetic (Arb, through python-flint).
"""

import math

import flint
from flint import arb


def _height_bound(values, d):
    """An upper bound for `h(b)`, `b` an algebraic integer of degree dividing `d`
    whose conjugates, each repeated `d / deg b` times, are `values`.
    """
    return sum(max(arb(1), v.abs_upper()).log() for v in values) / d


def complex_pair(r, u, cap=10**5, margin=10**6):
    """`N` with `u_n != 0` for every `n >= N`, or `None`: see the module docstring."""
    k = len(r)
    P = flint.fmpz_poly([-c for c in reversed(r)] + [1])
    if P.gcd(P.derivative()).degree() > 0:
        return None
    old = flint.ctx.prec
    try:
        for prec in (256, 1024):
            flint.ctx.prec = prec
            found = _attempt(P, k, u, cap, margin, prec)
            if found != "precision":
                return found
        return None
    finally:
        flint.ctx.prec = old


def _attempt(P, k, u, cap, margin, prec):
    roots = sorted((z for z, _ in P.complex_roots()), key=lambda z: -z.abs_upper().mid())
    top, rest = roots[:2], roots[2:]
    l = top[0] if top[0].imag > 0 else top[1]
    if not (l.imag > 0 and top[0].overlaps(top[1].conjugate())):
        return None
    rho = max((z.abs_upper() for z in rest), default=arb(0))
    if not l.abs_lower() > rho or not rho > 0:
        return None  # rho = 0 (a zero root) would make q = 0 and log q infinite
    V = flint.acb_mat([[z**n for z in roots] for n in range(k)])
    coeffs = V.solve(flint.acb_mat([[v] for v in u])).entries()
    c = coeffs[0] if roots[0] is l else coeffs[1]
    if not c.abs_lower() > 0:
        return None
    W = sum((x.abs_upper() for x in coeffs[2:]), arb(0))
    if W == 0:
        return None  # u_n = 2 Re(c l^n): a zero is Lambda = 0, left to the p-adic side
    q = rho / l.abs_lower()
    A = W / (4 * c.abs_lower())
    theta = l.arg() / arb.pi()
    mu = c.arg() / arb.pi() - arb(1) / 2

    # Matveev: 3 logarithms in K = Q(l, conj l), of degree D <= d (d - 1), d the
    # degree of the factor f of P with f(l) = 0. The conjugates of l are the roots of
    # f, and those of c, in Q(l), the coefficients of these roots.
    f = next(g for g, _ in P.factor()[1] if g(l).contains(0))
    d = f.degree()
    own = [i for i, z in enumerate(roots) if f(z).contains(0)]
    if len(own) != d:
        return "precision"
    D = d * (d - 1)
    # Lambda = 0 has at most one solution only if l / conj l is not a root of unity;
    # a root of unity of degree <= D has order m with phi(m) <= D, so m <= 2 D^2.
    z = l / l.conjugate()
    if any((z**m - 1).contains(0) for m in range(1, 2 * D * D + 1)):
        return None
    disc = abs(int(P.discriminant()))  # disc / P'(l) is an algebraic integer, so is disc c
    h_l = 2 * _height_bound([roots[i] for i in own], d)  # h(l / conj l) <= 2 h(l)
    h_c = 2 * (_height_bound([coeffs[i] * disc for i in own], d) + arb(disc).log())
    logs = [(l / l.conjugate()).log(), (c / c.conjugate()).log(), flint.acb(-1).log()]
    A_j = [max(h.upper(), (abs(g) / D).upper(), arb(0.16) / D)
           for h, g in zip((h_l, h_c, arb(0)), logs)]
    e = arb(1).exp()
    # |b_3| = |2m - 1| <= 2n + 3 + |Lambda| / pi <= 2n + 3 + 2 A, as |Lambda| <= 2 pi A q^n
    def matveev(n):  # a lower bound for log |Lambda| at time n
        return (-margin * arb(2) ** 38 * arb(D) ** 5 * A_j[0] * A_j[1] * A_j[2]
                * (e * D).log() * (e * (2 * n + 3 + 2 * A)).log())

    def possible(n):  # can |Lambda| <= 2 pi A q^n hold?
        return not (matveev(n) > (2 * arb.pi() * A).log() + n * q.log())

    N0 = 1
    while possible(N0):
        N0 *= 2
    lo, hi = N0 // 2, N0  # possible(lo) or lo = 0; not possible(hi), nor beyond
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if possible(mid) else (lo, mid)
    # Lambda = 0: n h(l / conj l) = h(c / conj c), with Voutier's lower bound for h.
    voutier = arb(2) / (D * (3 * arb(D)).log() ** 3)
    n_star = (h_c / voutier).upper()
    N0 = max(hi, int(float(n_star.mid())) + 1)

    # Baker--Davenport reduction with the continued fraction of theta.
    if not theta.rad() < arb(2) ** (-4 * N0.bit_length() - 64):
        return "precision"
    x, (p0, q0), (p1, q1) = theta, (0, 1), (1, 0)
    for _ in range(4 * N0.bit_length() + 200):
        a = x.floor().unique_fmpz()
        if a is None:
            return "precision"
        a = int(a)
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        if q1 > 6 * N0:
            Q = q1
            d_mu, d_theta = _dist(Q * mu), _dist(Q * theta)
            if d_mu is None or d_theta is None:
                return "precision"
            eps = d_mu - N0 * d_theta
            if eps > 0:
                n = ((A * Q / eps).log() / (1 / q).log()).upper()
                if not n.is_finite():
                    return None
                N = max(0, math.floor(float(n.mid())) + 1)
                return N if N <= cap else None
            if q1 > 10**6 * N0:
                return None  # mu in Z theta' + Z up to precision: Lambda = 0 somewhere
        frac = x - a
        if not frac > 0:
            return None  # theta' rational: l / conj l a root of unity
        x = 1 / frac
    return None


def _dist(x):
    """`||x||`, the distance to the nearest integer, as a ball; `None` if the ball
    of `x` meets a half-integer.
    """
    m = (x.mid() + arb(1) / 2).floor().unique_fmpz()
    d = abs(x - int(m))
    return d if d < arb(1) / 2 else None
