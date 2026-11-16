import math
from fractions import Fraction

import numpy as np

from .arithmetic import factorial_valuation, valuation
from .equilibria import Equilibrium
from .polynomial import _derivative, evaluate_exact, parse_field, reduce, taylor_shift


class _Retry(Exception):
    pass


def _valuations(array, p: int, cap: int) -> np.ndarray:
    out = np.full(array.shape, cap, dtype=np.int64)
    for index, value in np.ndenumerate(array):
        if value:
            out[index] = valuation(int(value), p, cap)
    return out


def smith(matrix, p: int, digits: int):
    """`P M Q = diag(p^{s_0}, p^{s_1}, ...)` over `Z/p^digits`, with `P` and `Q`
    invertible: the Smith normal form of an integer matrix over the `p`-adic integers,
    to `digits` digits. Returns `(P, P^{-1}, Q, s)`, with `s_i = digits` for the
    directions in which `M` vanishes to that precision.
    """
    m = p**digits
    work = np.array(matrix, dtype=object) % m
    rows, cols = work.shape
    left = np.eye(rows, dtype=object)
    left_inverse = np.eye(rows, dtype=object)
    right = np.eye(cols, dtype=object)
    s = []
    for k in range(min(rows, cols)):
        values = _valuations(work[k:, k:], p, digits)
        i, j = np.unravel_index(np.argmin(values), values.shape)
        v = int(values[i, j])
        if v >= digits:
            s.extend([digits] * (min(rows, cols) - k))
            break
        i, j = i + k, j + k
        work[[k, i]] = work[[i, k]]
        left[[k, i]] = left[[i, k]]
        left_inverse[:, [k, i]] = left_inverse[:, [i, k]]
        work[:, [k, j]] = work[:, [j, k]]
        right[:, [k, j]] = right[:, [j, k]]
        unit = int(work[k, k]) // p**v
        inverse = pow(unit, -1, m)
        work[k] = work[k] * inverse % m
        left[k] = left[k] * inverse % m
        left_inverse[:, k] = left_inverse[:, k] * unit % m
        factors = np.array([int(c) // p**v for c in work[k + 1 :, k]], dtype=object)
        work[k + 1 :] = (work[k + 1 :] - factors[:, None] * work[k]) % m
        left[k + 1 :] = (left[k + 1 :] - factors[:, None] * left[k]) % m
        left_inverse[:, k] = (
            left_inverse[:, k] + left_inverse[:, k + 1 :] @ factors
        ) % m
        factors = np.array([int(c) // p**v for c in work[k, k + 1 :]], dtype=object)
        work[:, k + 1 :] = (work[:, k + 1 :] - work[:, k : k + 1] * factors) % m
        right[:, k + 1 :] = (right[:, k + 1 :] - right[:, k : k + 1] * factors) % m
        s.append(v)
    return left, left_inverse, right, s


def _solve(matrix, vector, p: int, digits: int):
    """`x` with `M x = b` modulo `p^digits`, for `b` divisible enough; `None` if not."""
    left, _, right, s = smith(matrix, p, digits)
    c = left @ np.array(vector, dtype=object) % p**digits
    y = []
    for value, v in zip(c, s):
        if v >= digits:
            if value % p**digits:
                return None
            y.append(0)
        elif value % p**v:
            return None
        else:
            y.append(int(value) // p**v)
    return [int(v) % p**digits for v in right @ np.array(y, dtype=object)]


def _monomials(d: int, n: int) -> list[tuple[int, ...]]:
    if d == 1:
        return [(n,)]
    return [(k,) + rest for k in range(n, -1, -1) for rest in _monomials(d - 1, n - k)]


def _restricted_operator(matrix, basis) -> np.ndarray:
    """`L_A Phi = DPhi . Az - A Phi` on the span of the `z^e eps_i`, `(i, e)` in
    `basis`, which must be invariant.
    """
    d = len(matrix)
    index = {b: k for k, b in enumerate(basis)}
    L = np.zeros((len(basis), len(basis)), dtype=object)
    for column, (i, e) in enumerate(basis):
        for j in range(d):
            if not e[j]:
                continue
            for l in range(d):
                if matrix[j][l]:
                    f = e[:j] + (e[j] - 1,) + e[j + 1 :]
                    f = f[:l] + (f[l] + 1,) + f[l + 1 :]
                    L[index[(i, f)], column] += e[j] * matrix[j][l]
        for row in range(d):
            if matrix[row][i]:
                L[index[(row, e)], column] -= matrix[row][i]
    return L


def homological_matrix(matrix, monomials) -> np.ndarray:
    """The matrix of `L_A Phi = DPhi . Az - A Phi` on homogeneous vector polynomials,
    in the basis `z^e eps_i` ordered by `i` and then as `monomials`.
    """
    basis = [(i, e) for i in range(len(matrix)) for e in monomials]
    return _restricted_operator(matrix, basis)


def _rank(matrix) -> int:
    """The rank over `Q` of an integer matrix, by fraction-free elimination."""
    rows = [[int(c) for c in row] for row in matrix]
    rank, previous = 0, 1
    columns = len(rows[0]) if rows else 0
    for col in range(columns):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        top = rows[rank][col]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][col]
            rows[i] = [
                (top * a - factor * b) // previous for a, b in zip(rows[i], rows[rank])
            ]
        previous = top
        rank += 1
    return rank


def smith_exponent_bound(matrix, n: int, p: int) -> tuple[int, int | None]:
    """The rank `rho_n` over `Q` of the homological operator `L_n` of a rational matrix
    `A`, and an upper bound for every finite Smith exponent of `L_n` over `Z_p`
    (`None` when `rho_n = 0`):

    `s_j(n) <= floor(rho_n log_p(sqrt(rho_n) (n + 1) H)) - (rho_n - 1) a`,

    where `a = min v_p(A_ij)` and `H` bounds the entries of `qA`, `q` the common
    denominator. The minors of order `rho_n` of `L_n` are non-zero integers bounded by
    Hadamard's inequality. A Smith decomposition computed modulo `p^W` with `W` above
    the bound has exactly `M_n - rho_n` factors that vanish, and they are exact zeros.
    """
    p, n = int(p), int(n)
    A = [[Fraction(c) for c in row] for row in matrix]
    d = len(A)
    q = math.lcm(*(c.denominator for row in A for c in row))
    if q % p == 0:
        raise ValueError("The entries of `A` must be p-adic integers.")
    integral = [[int(c * q) for c in row] for row in A]
    entries = [c for row in integral for c in row if c]
    if not entries:
        return 0, None
    a = min(valuation(abs(c), p, abs(c).bit_length() + 1) for c in entries)
    L = homological_matrix(integral, _monomials(d, n))
    rho = _rank(L)
    if rho == 0:
        return 0, None
    H = max(1, max(abs(c) for c in entries))
    # The largest B with p^(2B) <= rho^rho ((n + 1) H)^(2 rho), in exact integers.
    target = rho**rho * ((n + 1) * H) ** (2 * rho)
    B = 0
    while p ** (2 * (B + 1)) <= target:
        B += 1
    return rho, B - (rho - 1) * a


def _matmul(a, b, m: int):
    return [
        [sum(x * y for x, y in zip(row, column)) % m for column in zip(*b)] for row in a
    ]


def _matpow(a, e: int, m: int):
    out = [[int(i == j) for j in range(len(a))] for i in range(len(a))]
    while e:
        if e & 1:
            out = _matmul(out, a, m)
        a = _matmul(a, a, m)
        e >>= 1
    return out


def _idempotent(B, p: int, digits: int):
    """The idempotent commuting with `B` onto the part on which `B` is invertible
    modulo `p`, along the part on which it is nilpotent: `lim (B^N)^{p^k}`, where
    `N` is a multiple of the order of every element of `GL_d(F_p)` with `N >= d`.
    """
    d, m = len(B), p**digits
    order = 1
    for i in range(1, d + 1):
        order = math.lcm(order, p**i - 1)
    unipotent = 1
    while unipotent < d:
        unipotent *= p
    U = _matpow(B, order * unipotent, m)
    for _ in range(digits):
        U = _matpow(U, p, m)
    return U


def _residual_splitting(matrix, p: int, digits: int):
    """`(T, T^{-1}, groups)` with `T` in `GL_d(Z_p)` and `T^{-1} A T` block diagonal
    modulo `p^digits`, the coordinates of each block listed in `groups`: one block
    for each eigenvalue in `F_p` of `A / p^a` modulo `p` (`a = v(A)`), and one for
    the remaining eigenvalues of each degree over `F_p`. `None` if there is only
    one block.
    """
    d, m = len(matrix), p**digits
    a = min((valuation(c, p, digits) for row in matrix for c in row), default=digits)
    if d == 1 or a >= digits:
        return None
    reduced = [[c // p**a for c in row] for row in matrix]
    eye = [[int(i == j) for j in range(d)] for i in range(d)]

    def minus(x, y):
        return [[(u - v) % m for u, v in zip(r, s)] for r, s in zip(x, y)]

    projectors, rest = [], eye
    shifts = [[[alpha * u for u in row] for row in eye] for alpha in range(p)]
    for k in range(1, d + 1):
        if not any(c % p for row in rest for c in row):
            break
        if k == 1:
            candidates = [minus(reduced, shift) for shift in shifts]
        else:
            candidates = [minus(_matpow(reduced, p**k, m), reduced)]
        for B in candidates:
            F = _idempotent(B, p, digits)
            E = _matmul(minus(eye, F), rest, m)
            if any(c % p for row in E for c in row):
                projectors.append(E)
                rest = _matmul(rest, F, m)
    if any(c % p for row in rest for c in row):
        projectors.append(rest)
    if len(projectors) < 2:
        return None
    columns, groups = [], []
    for E in projectors:
        chosen, basis = [], []
        for j in range(d):
            v = [E[i][j] % p for i in range(d)]
            for b, lead in basis:
                if v[lead]:
                    f = v[lead] * pow(b[lead], -1, p) % p
                    v = [(x - f * y) % p for x, y in zip(v, b)]
            lead = next((i for i, x in enumerate(v) if x), None)
            if lead is not None:
                basis.append((v, lead))
                chosen.append(j)
        groups.append(list(range(len(columns), len(columns) + len(chosen))))
        columns += [[E[i][j] for i in range(d)] for j in chosen]
    # The projectors are complementary idempotents, so these hold by construction;
    # failing them is a bug, not a reason to fall back to the dense operator.
    if len(columns) != d:
        raise RuntimeError("The residual projectors do not span Z_p^d.")
    T = [list(row) for row in zip(*columns)]
    left, _, right, s = smith(T, p, digits)
    if any(s):
        raise RuntimeError("The residual splitting is not invertible over Z_p.")
    inverse = [[int(c) % m for c in row] for row in right @ left % m]
    conjugate = _matmul(_matmul(inverse, matrix, m), T, m)
    block = {j: g for g, group in enumerate(groups) for j in group}
    if any(conjugate[i][j] for i in range(d) for j in range(d) if block[i] != block[j]):
        raise RuntimeError("The residual splitting does not make A block diagonal.")
    return T, inverse, groups


def _change(polys: list[dict], T, inverse, m: int) -> list[dict]:
    """`T^{-1} g(T w)` for polynomial maps `g`, modulo `m`."""
    d = len(T)
    maps = [
        {tuple(int(i == k) for i in range(d)): T[j][k] for k in range(d) if T[j][k]}
        for j in range(d)
    ]
    top = max((sum(e) for g in polys for e in g), default=0)
    moved = [_substitute(g, maps, top, m) for g in polys]
    out = []
    for i in range(d):
        total = {}
        for j in range(d):
            if inverse[i][j]:
                for e, c in moved[j].items():
                    total[e] = (total.get(e, 0) + inverse[i][j] * c) % m
        out.append({e: c for e, c in total.items() if c})
    return out


def _multiply(a: dict, b: dict, degree: int, m: int) -> dict:
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(i + j for i, j in zip(ea, eb))
            if sum(e) <= degree:
                out[e] = (out.get(e, 0) + ca * cb) % m
    return {e: c for e, c in out.items() if c}


def _substitute(poly: dict, maps: list[dict], degree: int, m: int) -> dict:
    """`poly(maps)` up to total degree `degree`, modulo `m`, for polynomial maps
    without constant term.
    """
    d = len(maps)
    one = (0,) * d
    powers = [[{one: 1}] for _ in range(d)]
    out = {}
    for e, c in poly.items():
        if sum(e) > degree:
            continue
        term = {one: c % m}
        for j, k in enumerate(e):
            while len(powers[j]) <= k:
                powers[j].append(_multiply(powers[j][-1], maps[j], degree, m))
            if k:
                term = _multiply(term, powers[j][k], degree, m)
        for f, value in term.items():
            out[f] = (out.get(f, 0) + value) % m
    return {f: c for f, c in out.items() if c}


def _evaluate(polys: list[dict], points: np.ndarray, m: int) -> np.ndarray:
    """Polynomials with integer coefficients at points of shape `(B, d)`, modulo `m`,
    with Python integers.
    """
    out = np.zeros((points.shape[0], len(polys)), dtype=object)
    for i, poly in enumerate(polys):
        total = np.zeros(points.shape[0], dtype=object)
        for e, c in poly.items():
            term = np.full(points.shape[0], c, dtype=object)
            for j, k in enumerate(e):
                if k:
                    term = term * points[:, j] ** k % m
            total = total + term
        out[:, i] = total % m
    return out


def _scaled_field(source, p: int, centre, r: int, digits: int) -> list[dict]:
    """`f~(z) = p^{-r} f(centre + p^r z)` modulo `p^digits`, one dictionary per
    component, constant terms included. `centre` must hold at least `digits + r`
    digits.
    """
    big = p ** (digits + r)
    m = p**digits
    out = []
    for component in reduce(parse_field(source, p, digits=digits + r), big):
        shifted = taylor_shift(component, centre, big)
        scaled = {}
        for e, c in shifted.items():
            n = sum(e)
            if n == 0:
                if c % p**r:
                    raise ValueError("The centre is not an equilibrium of the field.")
                value = c // p**r % m
            else:
                value = c * p ** (r * (n - 1)) % m
            if value:
                scaled[e] = value
        out.append(scaled)
    return out


def _degree_valuations(polys: list[dict], p: int, digits: int) -> dict[int, int]:
    """The least valuation of the coefficients of each degree `>= 2`, among those
    that do not vanish modulo `p^digits`.
    """
    out = {}
    for poly in polys:
        for e, c in poly.items():
            n = sum(e)
            if n >= 2 and c % p**digits:
                out[n] = min(out.get(n, digits), valuation(c, p, digits))
    return out


def _majorant(phi, normal, field, p: int, K: int, degree: int):
    """The integer `b` and the slope `eta` that make the tail bound of
    `Certificate` reach `K` at `degree`, or `None`: the largest `b + eta degree` over
    `b = 0, ..., K` with `eta > 0`, `v(Phi_n) >= eta (n - 1)`,
    `v(R_n) >= b + eta (n - 1)` and `v(f~_n) >= b + eta (n - 1)`.
    """
    vphi = _degree_valuations(phi, p, K)
    vrest = _degree_valuations(normal, p, K)
    vfield = _degree_valuations(field, p, K)
    if not (vphi or vrest or vfield):
        return K, Fraction(1)
    best = None
    for b in range(K + 1):
        bounds = [Fraction(v, n - 1) for n, v in vphi.items()] + [
            Fraction(v - b, n - 1)
            for table in (vrest, vfield)
            for n, v in table.items()
        ]
        eta = min(bounds)
        if eta <= 0:
            break
        reach = b + eta * degree
        if reach >= K and (best is None or reach > best[0]):
            best = (reach, b, eta)
    return None if best is None else best[1:]


class Certificate:
    """A checkable proof that a `NormalForm` is exact. Let `Phi` and `N` be the
    computed polynomials (coefficients reduced modulo `p^K`, degree at most `M`) and
    `f~(z) = p^{-r} f(c + p^r z)` the field around the centre `c`. If

    1. `Phi = z + H` with every coefficient of `H` divisible by `p`, and `N` and `f~`
       have coefficients divisible by `p` (`4` for `p = 2`);
    2. the defect `E = DPhi . N - f~ o Phi` vanishes modulo `p^K` in every degree
       `n <= M`;
    3. `v(Phi_n) >= eta (n - 1)`, `v(N_n) >= b + eta (n - 1)` and
       `v(f~_n) >= b + eta (n - 1)` for `n >= 2`, with `b + eta M >= K`,

    then the parts of `E` of degree above `M` vanish modulo `p^K` as well, and

    `phi_t(x) = c + p^r Phi(psi_t(Phi^{-1}((x - c) / p^r)))`  modulo `p^{K + r}`

    for every `t` in `Z_p` and every `x` with `|x - c| <= p^{-r}`, where `psi_t` is
    the flow of `N`. `verify` checks 1-3 from the field itself, independently of the
    construction; the analytic tail of a `PowerSeries` field enters only through its
    declared `bound`.

    **Attributes:** `precision` (`K`), `radius` (`r`), `centre` (`c`, to `K + r`
    digits), `degree` (`M`), `margin` (`b`) and `slope` (`eta`, a `Fraction`).
    """

    def __init__(self, form, degree: int, margin: int, slope: Fraction):
        self._form = form
        self.p, self.precision = form.p, form.precision
        self.radius = form.radius
        self.centre = tuple(
            int(c) % form.p ** (form.precision + form.radius) for c in form._centre
        )
        self.degree = int(degree)
        self.margin = int(margin)
        self.slope = Fraction(slope)

    def verify(self) -> bool:
        """Checks conditions 1-3 with exact integer arithmetic."""
        form = self._form
        p, K, r, M = self.p, self.precision, self.radius, self.degree
        d = form.dimension
        m = p**K
        needed = 2 if p == 2 else 1
        field = _scaled_field(form.flow._source, p, self.centre, r, K)
        normal = form.normal_form
        phi = [
            {**{tuple(int(i == j) for i in range(d)): 1}, **component}
            for j, component in enumerate(form.transformation)
        ]
        for j in range(d):
            terms = list(form.transformation[j]) + list(normal[j])
            if any(sum(e) < 1 or sum(e) > M for e in terms):
                return False
            if any(sum(e) < 2 or c % p for e, c in form.transformation[j].items()):
                return False
            values = list(normal[j].values()) + list(field[j].values())
            if any(c % p**needed for c in values):
                return False
        if self.slope <= 0 or self.margin + self.slope * M < K:
            return False
        tables = (
            (_degree_valuations(form.transformation, p, K), 0),
            (_degree_valuations(normal, p, K), self.margin),
            (_degree_valuations(field, p, K), self.margin),
        )
        for table, offset in tables:
            if any(v < offset + self.slope * (n - 1) for n, v in table.items()):
                return False
        for i in range(d):
            defect = {}
            for b in range(d):
                product = _multiply(_derivative(phi[i], b, m), normal[b], M, m)
                for e, c in product.items():
                    defect[e] = (defect.get(e, 0) + c) % m
            for e, c in _substitute(field[i], phi, M, m).items():
                defect[e] = (defect.get(e, 0) - c) % m
            if any(defect.values()):
                return False
        return True

    def __repr__(self) -> str:
        return (
            f"Certificate(p={self.p}, precision={self.precision}, "
            f"radius={self.radius}, degree={self.degree}, margin={self.margin}, "
            f"slope={self.slope})"
        )


class NormalForm:
    """The Poincare--Dulac normal form of a flow around an equilibrium, in any
    dimension: an analytic change of coordinates `x = x* + p^r Phi(z)`,
    `Phi(z) = z + O(z^2)`, in which the field becomes

    `dz/dt = N(z) = A z + R(z)`,  `A = Df(x*)`,

    where `R` holds only the terms that cannot be removed. When `R = 0` the flow is
    linearised, `phi_t(x) = x* + p^r Phi(e^{At} Phi^{-1}((x - x*) / p^r))`.

    Degree by degree, `Phi_n` and `R_n` solve the homological equation
    `D Phi_n . A z - A Phi_n + R_n = G_n`, where `G_n` comes from the lower degrees. The
    operator on the left is a matrix on homogeneous polynomials of degree `n`, and is
    put in Smith normal form over `Z_p`: its elementary divisors `p^{s_i}` are the
    small divisors `<m, lambda> - lambda_i` of the classical theory, obtained without
    computing the eigenvalues, which may lie in an extension of `Q_p`. A uniform
    treatment covers:

    - non-resonant eigenvalues, in `Q_p` or in an extension (complex pairs);
    - Jordan blocks;
    - resonances `<m, lambda> = lambda_i`, including zero eigenvalues: the
        corresponding directions (`s_i` equal to the working precision) stay in `R`;
    - near-resonances: directions with `s_i > resonance_digits` stay in `R` as well.
        Keeping a removable term in `R` never breaks the conjugacy, so the result is
        exact either way; the threshold only trades the size of the domain against the
        number of terms left in the normal form.

    The coordinates are scaled, `x - x* = p^r z`. The radius `r` is the smallest one
    for which the construction is certified, and the series are cut at the first
    degree `M` at which they are: the result carries a `Certificate`, checked when it
    is built, that proves the flow through the normal form exact modulo `p^K` on the
    ball `|x - x*| <= p^{-r}`. Every degree is solved exactly modulo the working
    power of `p`, so no digits are lost in the recursion; the tail beyond `M` is
    bounded from the valuations of the computed coefficients. The normal form is a
    polynomial field divisible by `p`, so its flow is computed exactly by a
    `PAdicFlow`.

    **Arguments:**

    - `flow`: a `PAdicFlow`.
    - `equilibrium`: an `Equilibrium` or a point. Simple equilibria are refined by
        Newton's method; degenerate ones (`det Df(x*) = 0`) must be given exactly.
    - `radius`: `r`; found automatically by default.
    - `resonance_digits`: divisors of valuation above it are treated as resonant.
        Defaults to `K // 2`.
    - `max_degree`: the largest degree computed before giving up on a radius.
    """

    def __init__(
        self,
        flow,
        equilibrium,
        radius: int | None = None,
        resonance_digits: int | None = None,
        max_degree: int = 60,
    ):
        from .flow import PAdicFlow

        p, K, d = flow.p, flow.precision, flow.dimension
        threshold = K // 2 if resonance_digits is None else int(resonance_digits)
        if isinstance(equilibrium, Equilibrium):
            point = list(equilibrium.point)
        else:
            point = [
                int(v) for v in np.atleast_1d(np.asarray(equilibrium, dtype=object))
            ]
        if len(point) != d:
            raise ValueError(f"The equilibrium must be a point of Z_p^{d}.")
        self.flow = flow
        self.p, self.precision, self.dimension = p, K, d
        self.resonance_digits = threshold
        self._needed = 2 if p == 2 else 1
        self._smith = {}
        radii = range(1, K) if radius is None else [int(radius)]
        for r in radii:
            work = K + max(threshold, 0) + 2
            field = parse_field(flow._source, p, digits=work + r + 2)
            centre = self._refine(field, point, work + r + 2)
            try:
                self._build(centre, r, work, threshold, max_degree)
                break
            except _Retry:
                continue
        else:
            raise ValueError(
                "No radius makes the normal form converge; try a larger "
                "`resonance_digits` or `max_degree`."
            )
        modulus = p**K
        self.equilibrium = tuple(c % modulus for c in centre)
        self.radius = r
        self._centre = centre
        normal = [
            {e: c % modulus for e, c in component.items() if c % modulus}
            for component in self._normal
        ]
        self.normal_form = normal
        self.resonant_terms = [
            {e: c % modulus for e, c in component.items() if c % modulus and sum(e) > 1}
            for component in self._normal
        ]
        self.linearizable = not any(self.resonant_terms)
        self.transformation = [
            {e: c % modulus for e, c in component.items() if c % modulus}
            for component in self._phi
        ]
        self.certificate = Certificate(self, self.degree, *self._majorant)
        if not self.certificate.verify():
            raise RuntimeError("The normal form failed its own certificate.")
        self._normal_flow = PAdicFlow(p, K, normal)

    #
    # Construction
    #

    def _refine(self, field, point, digits: int) -> list[int]:
        """The equilibrium to `digits` digits, by Newton's method when `Df(x*)` is
        invertible over `Q_p`.
        """
        p, d = self.p, self.dimension
        m = p**digits
        polys = reduce(field, m)
        x = [v % m for v in point]
        for _ in range(digits.bit_length() + 3):
            values = [evaluate_exact(f, x, m) for f in polys]
            if not any(values):
                return x
            jac = [
                [evaluate_exact(_derivative(f, j, m), x, m) for j in range(d)]
                for f in polys
            ]
            step = _solve(jac, values, p, digits)
            if step is None:
                break
            x = [(a - b) % m for a, b in zip(x, step)]
        if any(evaluate_exact(f, x, m) for f in polys):
            raise ValueError(
                "The point is not an equilibrium to the working precision; "
                "degenerate equilibria must be given exactly."
            )
        return x

    def _operator(self, n: int, matrix, monomials, groups, digits: int):
        """The homological operator of degree `n`, block by block: for a block
        diagonal `A` with coordinate blocks `groups`, the span of the `z^e eps_i` with
        given degrees of `e` in each block and `i` in a given block is invariant.
        Returns, for every such piece, its positions in the basis
        `z^e eps_i -> i * len(monomials) + k` and the Smith decomposition of `L_A`
        on it.
        """
        key = (n, digits)
        if key not in self._smith:
            block = {j: g for g, group in enumerate(groups) for j in group}
            pieces = {}
            for i in range(self.dimension):
                for k, e in enumerate(monomials):
                    degrees = tuple(sum(e[j] for j in group) for group in groups)
                    pieces.setdefault((degrees, block[i]), []).append((i, k, e))
            out = []
            for members in pieces.values():
                L = _restricted_operator(matrix, [(i, e) for i, _, e in members])
                positions = [i * len(monomials) + k for i, k, _ in members]
                out.append((positions, smith(L % self.p**digits, self.p, digits)))
            self._smith[key] = out
        return self._smith[key]

    def _build(self, centre, r: int, work: int, threshold: int, top: int):
        p, K, d = self.p, self.precision, self.dimension
        m = p**work
        try:
            scaled = _scaled_field(self.flow._source, p, centre, r, work)
        except ValueError:
            raise ValueError("The equilibrium is not exact enough.") from None
        if any(f.get((0,) * d, 0) for f in scaled):
            raise ValueError("The equilibrium is not exact enough.")
        unit = [tuple(int(i == j) for i in range(d)) for j in range(d)]
        original = [[scaled[i].get(unit[j], 0) for j in range(d)] for i in range(d)]
        splitting = _residual_splitting(original, p, work)
        if splitting is None:
            groups = [list(range(d))]
        else:
            T, inverse, groups = splitting
            scaled = _change(scaled, T, inverse, m)
        matrix = [[scaled[i].get(unit[j], 0) for j in range(d)] for i in range(d)]
        nonlinear = [{e: c for e, c in f.items() if sum(e) >= 2} for f in scaled]
        field = [{e: c % p**K for e, c in f.items() if c % p**K} for f in scaled]
        phi = [dict() for _ in range(d)]
        rest = [dict() for _ in range(d)]
        divisors = {}
        for n in range(2, top + 1):
            monomials = _monomials(d, n)
            size = len(monomials)
            coordinates = [{**{unit[j]: 1}, **phi[j]} for j in range(d)]
            powers = {}

            def power(j, k):
                if (j, k) not in powers:
                    powers[(j, k)] = (
                        {(0,) * d: 1}
                        if k == 0
                        else _multiply(power(j, k - 1), coordinates[j], n, m)
                    )
                return powers[(j, k)]

            target = np.zeros(d * size, dtype=object)
            position = {e: k for k, e in enumerate(monomials)}
            for i in range(d):
                for e, c in nonlinear[i].items():
                    if sum(e) > n:
                        continue
                    term = {(0,) * d: c}
                    for j, k in enumerate(e):
                        if k:
                            term = _multiply(term, power(j, k), n, m)
                    for f, value in term.items():
                        if sum(f) == n:
                            target[i * size + position[f]] += value
                for b in range(d):
                    part = {e: c for e, c in rest[b].items() if 2 <= sum(e) < n}
                    if not part:
                        continue
                    derivative = _derivative(
                        {e: c for e, c in phi[i].items() if 2 <= sum(e) < n}, b, m
                    )
                    for f, value in _multiply(derivative, part, n, m).items():
                        if sum(f) == n:
                            target[i * size + position[f]] -= value
            target %= m
            pieces = self._operator(n, matrix, monomials, groups, work)
            projected = []
            for positions, (left, _, _, s) in pieces:
                c = left @ target[positions] % m
                solution = np.zeros(len(positions), dtype=object)
                kept = np.zeros(len(positions), dtype=object)
                for k, (value, v) in enumerate(zip(c, s)):
                    if not value:
                        continue
                    if v > threshold:
                        kept[k] = int(value)
                    elif value % p**v:
                        raise _Retry
                    else:
                        solution[k] = int(value) // p**v
                projected.append((c, solution, kept))
            divisors[n] = sorted(v for _, (_, _, _, s) in pieces for v in s)
            new_phi = np.zeros(d * size, dtype=object)
            new_rest = np.zeros(d * size, dtype=object)
            for (positions, (_, back, right, _)), (_, solution, kept) in zip(
                pieces, projected
            ):
                new_phi[positions] = right @ solution % m
                new_rest[positions] = back @ kept % m
            for i in range(d):
                for k, e in enumerate(monomials):
                    a, b = int(new_phi[i * size + k]), int(new_rest[i * size + k])
                    if a:
                        phi[i][e] = a
                        if a % p:
                            raise _Retry
                    if b:
                        rest[i][e] = b
                        if b % p**self._needed:
                            raise _Retry
            reduced_phi = [{e: c % p**K for e, c in f.items() if c % p**K} for f in phi]
            reduced_rest = [
                {e: c % p**K for e, c in f.items() if c % p**K} for f in rest
            ]
            found = _majorant(reduced_phi, reduced_rest, field, p, K, n)
            if found is not None:
                break
        else:
            raise _Retry
        if splitting is not None:
            phi = _change(phi, inverse, T, m)
            rest = _change(rest, inverse, T, m)
        self.matrix = [[c % p**K for c in row] for row in original]
        self.divisors = divisors
        self.degree = n
        self._majorant = found
        self._phi = phi
        self._normal = [
            {**{unit[j]: original[i][j] for j in range(d) if original[i][j]}, **rest[i]}
            for i in range(d)
        ]

    #
    # Coordinates and flow
    #

    def _points(self, x):
        x = np.asarray(x, dtype=object)
        single = x.ndim == (0 if self.dimension == 1 else 1)
        x = x.reshape(-1, self.dimension)
        return x, single

    def _shape(self, out, single):
        if self.dimension == 1:
            out = out[:, 0]
        return out[0] if single else out

    def forward(self, x):
        """`z = Phi^{-1}((x - x*) / p^r)`, the normal-form coordinates of `x`, by the
        fixed point `z = w - (Phi(z) - z)`, which contracts since every coefficient of
        `Phi(z) - z` is divisible by `p`.
        """
        x, single = self._points(x)
        p, K, r = self.p, self.precision, self.radius
        modulus = p**K
        offset = (x - np.array(self._centre, dtype=object)) % modulus
        if np.any(offset % p**r):
            raise ValueError(f"The point is outside the domain |x - x*| <= p^-{r}.")
        w = offset // p**r
        z = w.copy()
        for _ in range(K + 1):
            z = (w - _evaluate(self._phi, z, modulus)) % modulus
        return self._shape(z, single)

    def inverse(self, z):
        """`x* + p^r Phi(z)` modulo `p^K`."""
        z, single = self._points(z)
        p, K, r = self.p, self.precision, self.radius
        modulus = p**K
        image = z + _evaluate(self._phi, z, modulus)
        x = np.array(self._centre, dtype=object) + p**r * image
        return self._shape(x % modulus, single)

    def flow_map(self, t, x):
        """`phi_t(x)` through the normal form: `x* + p^r Phi(psi_t(Phi^{-1}(...)))`,
        with `psi_t` the flow of `N`.
        """
        x, single = self._points(x)
        z = np.asarray(self.forward(x), dtype=object).reshape(-1, self.dimension)
        points = z[:, 0] if self.dimension == 1 else z
        moved = np.asarray(self._normal_flow.flow(t, points), dtype=object)
        moved = moved.reshape(-1, self.dimension)
        out = np.asarray(self.inverse(moved), dtype=object).reshape(-1, self.dimension)
        return self._shape(out, single)

    def exponential(self, t) -> list[list[int]]:
        """`e^{At}` modulo `p^K`, the flow of the linear part."""
        p, K, d = self.p, self.precision, self.dimension
        t = int(t)
        entries = [c for row in self.matrix for c in row if c]
        rate = min((valuation(c, p, K) for c in entries), default=K)
        terms = 1
        while terms * rate - (terms - 1) // (p - 1) < K:
            terms += 1
        big = p ** (K + factorial_valuation(terms, p))
        total = [[int(i == j) for j in range(d)] for i in range(d)]
        power = [row[:] for row in total]
        for n in range(1, terms + 1):
            power = [
                [
                    sum(power[i][k] * self.matrix[k][j] * t for k in range(d)) % big
                    for j in range(d)
                ]
                for i in range(d)
            ]
            v = factorial_valuation(n, p)
            inverse = pow(math.factorial(n) // p**v, -1, big)
            for i in range(d):
                for j in range(d):
                    total[i][j] += power[i][j] // p**v * inverse
        return [[c % p**K for c in row] for row in total]

    def __repr__(self) -> str:
        kind = "linearizable" if self.linearizable else "resonant"
        return (
            f"NormalForm(p={self.p}, precision={self.precision}, "
            f"dimension={self.dimension}, radius={self.radius}, {kind})"
        )
