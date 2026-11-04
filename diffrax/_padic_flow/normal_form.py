import math

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


def _multiply(a: dict, b: dict, degree: int, m: int) -> dict:
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(i + j for i, j in zip(ea, eb))
            if sum(e) <= degree:
                out[e] = (out.get(e, 0) + ca * cb) % m
    return {e: c for e, c in out.items() if c}


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

    The coordinates are scaled, `x - x* = p^r z`, with `r` (the `radius`) the smallest
    value for which every division by `p^{s_i}` is exact and the coefficients decay:
    the conjugacy holds on the ball `|x - x*| <= p^{-r}`. The series are cut at the
    degree beyond which the observed decay `v >= e (n - 1)` of the coefficients
    passes the working precision. The normal form is a polynomial field divisible by
    `p`, so its flow is computed exactly by a `PAdicFlow`.

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
                self._build(field, centre, r, work, threshold, max_degree)
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

    def _operator(self, n: int, matrix, monomials, digits: int):
        key = (n, digits)
        if key not in self._smith:
            p, d = self.p, self.dimension
            index = {e: k for k, e in enumerate(monomials)}
            size = len(monomials)
            L = np.zeros((d * size, d * size), dtype=object)
            for i in range(d):
                for k, e in enumerate(monomials):
                    column = i * size + k
                    for j in range(d):
                        if not e[j]:
                            continue
                        for l in range(d):
                            f = list(e)
                            f[j] -= 1
                            f[l] += 1
                            L[i * size + index[tuple(f)], column] += e[j] * matrix[j][l]
                    for row in range(d):
                        L[row * size + k, column] -= matrix[row][i]
            self._smith[key] = smith(L % p**digits, p, digits)
        return self._smith[key]

    def _build(self, field, centre, r: int, work: int, threshold: int, top: int):
        p, K, d = self.p, self.precision, self.dimension
        m = p**work
        big = p ** (work + r + 2)
        scaled = []
        for component in reduce(field, big):
            shifted = taylor_shift(component, centre, big)
            if shifted.get((0,) * d, 0) % p ** (work + r):
                raise ValueError("The equilibrium is not exact enough.")
            scaled.append(
                {
                    e: c * p ** (r * (sum(e) - 1)) % m
                    for e, c in shifted.items()
                    if sum(e) >= 1 and c * p ** (r * (sum(e) - 1)) % m
                }
            )
        unit = [tuple(int(i == j) for i in range(d)) for j in range(d)]
        matrix = [[scaled[i].get(unit[j], 0) for j in range(d)] for i in range(d)]
        nonlinear = [{e: c for e, c in f.items() if sum(e) >= 2} for f in scaled]
        reach = max(
            (sum(e) for f in nonlinear for e, c in f.items() if c % p**K), default=2
        )
        phi = [dict() for _ in range(d)]
        rest = [dict() for _ in range(d)]
        divisors = {}
        lowest = []
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
            left, left_inverse, right, s = self._operator(n, matrix, monomials, work)
            c = left @ target % m
            solution = np.zeros(d * size, dtype=object)
            kept = np.zeros(d * size, dtype=object)
            for k, (value, v) in enumerate(zip(c, s)):
                if v > threshold:
                    kept[k] = value
                elif value % p**v:
                    raise _Retry
                else:
                    solution[k] = int(value) // p**v
            divisors[n] = sorted(s)
            new_phi = right @ solution % m
            new_rest = left_inverse @ kept % m
            smallest = K
            for i in range(d):
                for k, e in enumerate(monomials):
                    a, b = int(new_phi[i * size + k]), int(new_rest[i * size + k])
                    if a:
                        phi[i][e] = a
                        if a % p**K:
                            v = valuation(a, p, K)
                            if v < 1:
                                raise _Retry
                            smallest = min(smallest, v)
                    if b:
                        rest[i][e] = b
                        if b % p**K:
                            v = valuation(b, p, K)
                            if v < self._needed:
                                raise _Retry
                            smallest = min(smallest, v)
            lowest.append(smallest)
            slope = min(v / (k + 1) for k, v in enumerate(lowest))
            if n >= reach and slope * n >= K:
                break
        else:
            raise _Retry
        self.matrix = [[c % p**K for c in row] for row in matrix]
        self.divisors = divisors
        self.degree = n
        self._phi = phi
        self._normal = [
            {**{unit[j]: matrix[i][j] for j in range(d) if matrix[i][j]}, **rest[i]}
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
        return out[0] if single else out

    def _phi_values(self, z, modulus):
        return (z + _evaluate(self._phi, z, modulus)) % modulus

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
        x = np.array(self._centre, dtype=object) + p**r * self._phi_values(z, modulus)
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
        modulus = p**K
        t = int(t)
        guard = K
        while True:
            terms = 1
            while (
                terms * self._valuation_of_matrix() - factorial_valuation(terms, p) < K
            ):
                terms += 1
            if terms < guard:
                break
            guard *= 2
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
            factorial = math.factorial(n)
            v = valuation(factorial, p, 10**9)
            inverse = pow(factorial // p**v, -1, big)
            for i in range(d):
                for j in range(d):
                    total[i][j] += power[i][j] // p**v * inverse
        return [[c % modulus for c in row] for row in total]

    def _valuation_of_matrix(self) -> int:
        values = [c for row in self.matrix for c in row if c]
        if not values:
            return self.precision
        return min(valuation(c, self.p, self.precision) for c in values)

    def __repr__(self) -> str:
        kind = "linearizable" if self.linearizable else "resonant"
        return (
            f"NormalForm(p={self.p}, precision={self.precision}, "
            f"dimension={self.dimension}, radius={self.radius}, {kind})"
        )
