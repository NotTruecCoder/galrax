import math
from dataclasses import dataclass
from fractions import Fraction

from .arithmetic import factorial_valuation, valuation
from .polynomial import _derivative, evaluate_exact, reduce, taylor_shift
from .roots import vanishing_set


@dataclass(frozen=True)
class Equilibrium:
    """A zero of the vector field.

    - `point`: an approximation, exact modulo `p^level`.
    - `level`: the equilibrium lies in `point + p^level Z_p^d`, and every point of that
        ball has `f = 0` to the working precision.
    - `certified`: Hensel's lemma (`v(f) > 2 v(det Df)` at `point`) shows that the ball
        contains exactly one exact zero, which is simple.
    """

    point: tuple
    level: int
    certified: bool


def _determinant(matrix: list, modulus: int) -> int:
    matrix = [row[:] for row in matrix]
    size = len(matrix)
    if size == 1:
        return matrix[0][0] % modulus
    total = 0
    for j in range(size):
        minor = [row[:j] + row[j + 1 :] for row in matrix[1:]]
        sign = -1 if j % 2 else 1
        total += sign * matrix[0][j] * _determinant(minor, modulus)
    return total % modulus


def jacobian(flow, point) -> list:
    """`Df(point)` modulo `p^K`, as a list of rows of Python integers."""
    modulus = flow.modulus
    field = reduce(flow.field, modulus)
    point = [int(x) % modulus for x in point]
    return [
        [
            evaluate_exact(_derivative(component, j, modulus), point, modulus)
            for j in range(flow.dimension)
        ]
        for component in field
    ]


def equilibria(flow, max_balls: int = 100_000) -> list[Equilibrium]:
    p, K, d = flow.p, flow.precision, flow.dimension
    modulus = flow.modulus
    field = reduce(flow.field, modulus)
    balls = vanishing_set(field, d, p, K, K, max_balls)
    out = []
    for ball in balls:
        values = [evaluate_exact(f, ball.center, modulus) for f in field]
        size = min(valuation(v, p, K) for v in values)
        delta = valuation(_determinant(jacobian(flow, ball.center), modulus), p, K)
        certified = delta < K and size > 2 * delta
        out.append(Equilibrium(ball.center, ball.level, certified))
    return out


class Linearization:
    """Linearising coordinates around a simple equilibrium `x*` of a one-dimensional
    flow: an analytic `h` with `h(x*) = 0`, `h'(x*) = 1` and

    `h(phi_t(x)) = e^{lambda t} h(x)`,  `lambda = f'(x*)`,

    so that `phi_t(x) = x* + h^{-1}(e^{lambda t} h(x - x*))`. This is the `p`-adic
    Poincare--Siegel linearisation, which always converges in one dimension.

    Writing `f(x* + y) = lambda y + sum_{k>=2} a_k y^k`, the coefficients follow from
    `h' f = lambda h`:

    `(n - 1) lambda h_n = -sum_{k=2}^{n} (n - k + 1) a_k h_{n-k+1}`.

    With `v(a_k) >= a` (the valuation of the field) and `l = v(lambda)` this gives
    `v(h_n) >= (n - 1)(a - l) - v((n - 1)!)`, so `h` converges on `|y| < p^{-c}` with
    `c = l - a + 1/(p - 1)`, and the tail beyond a computable number of terms is
    below `p^{-K}`. The equilibrium is refined by Newton's method to enough extra
    digits for every division by `(n - 1) lambda`.

    **Arguments:**

    - `flow`: a one-dimensional `PAdicFlow`.
    - `equilibrium`: an `Equilibrium` (certified) or an approximation of `x*`.
    - `level`: the domain `|x - x*| <= p^{-level}`; defaults to the largest one the
        bound allows.
    """

    def __init__(self, flow, equilibrium, level: int | None = None):
        if flow.dimension != 1:
            raise ValueError("Linearisation is implemented for d = 1 only.")
        p, K, a = flow.p, flow.precision, flow.field_valuation
        if isinstance(equilibrium, Equilibrium):
            start = equilibrium.point[0]
        else:
            start = int(equilibrium)
        coefficients = {e[0]: c for e, c in flow.field[0].items()}
        derivative = {k - 1: k * c for k, c in coefficients.items() if k}

        def value(poly, x, modulus):
            total = sum(
                c.numerator * pow(c.denominator, -1, modulus) * pow(x, k, modulus)
                for k, c in poly.items()
            )
            return total % modulus

        slope = value(derivative, start, p**K)
        ell = valuation(slope, p, K)
        if ell >= K:
            raise ValueError("The equilibrium is degenerate: f'(x*) = 0.")
        bound = Fraction(ell - a) + Fraction(1, p - 1)
        level = math.floor(bound) + 1 if level is None else int(level)
        if level <= bound or level < 0:
            raise ValueError(f"`level` must exceed {bound} for the series to converge.")
        terms = 2
        while terms * (level - bound) + bound < K:
            terms += 1
        work = K + terms * (ell + 2) + factorial_valuation(terms, p) + 4
        big = p**work
        x = start % big
        for _ in range(work.bit_length() + 2):
            fx = value(coefficients, x, big)
            dfx = value(derivative, x, big)
            v = valuation(dfx, p, work)
            if fx == 0:
                break
            if fx % p**v:
                raise ValueError("The approximation is too far from a simple zero.")
            step = (fx // p**v) * pow(dfx // p**v, -1, big) % big
            x = (x - step) % big
        if value(coefficients, x, big) % p ** (work - 2 * ell) != 0:
            raise ValueError("Newton's method did not converge to an equilibrium.")

        reduced = {(k,): c for k, c in coefficients.items()}
        shifted = taylor_shift(reduce([reduced], big)[0], (x,), big)
        a_k = {e[0]: c for e, c in shifted.items()}
        lam = a_k.get(1, 0)
        h = [Fraction(0), Fraction(1)]
        for n in range(2, terms + 1):
            total = sum(
                (n - k + 1) * a_k.get(k, 0) * h[n - k + 1] for k in range(2, n + 1)
            )
            h.append(-Fraction(total) / ((n - 1) * lam))

        self.flow = flow
        self.p, self.precision = p, K
        self.equilibrium = x % p**K
        self.multiplier = lam % p**K
        self.multiplier_valuation = ell
        self.level = level
        self.terms = terms
        self._big = big
        self._x = x
        self._lambda = lam
        self._h = [self._padic(c) for c in h]

    def _padic(self, c: Fraction):
        if c == 0:
            return (0, 0)
        p = self.p
        num_v = valuation(c.numerator, p, 10**9)
        den_v = valuation(c.denominator, p, 10**9)
        unit = (c.numerator // p**num_v) * pow(c.denominator // p**den_v, -1, self._big)
        return (num_v - den_v, unit % self._big)

    def _check(self, y: int) -> int:
        if y and valuation(y, self.p, self.precision) < self.level:
            raise ValueError(
                f"The point is outside the domain |x - x*| <= p^-{self.level}."
            )
        return y

    def _series(self, y: int, derivative: bool = False) -> int:
        p, modulus = self.p, self.p**self.precision
        if y == 0:
            return int(derivative)
        s = valuation(y, p, self.precision)
        unit_y = y // p**s
        total = 0
        for n, (v, u) in enumerate(self._h):
            if u == 0 or (derivative and n == 0):
                continue
            power = n - 1 if derivative else n
            exponent = v + power * s
            factor = n if derivative else 1
            fv = valuation(factor, p, 64)
            exponent += fv
            if exponent >= self.precision:
                continue
            if exponent < 0:
                raise RuntimeError("Linearisation series left its domain.")
            term = u * (factor // p**fv) * pow(unit_y, power, modulus) * p**exponent
            total += term
        return total % modulus

    def forward(self, x) -> int:
        """`h(x - x*)` modulo `p^K`."""
        y = (int(x) - self._x) % self.p**self.precision
        return self._series(self._check(y))

    def inverse(self, w) -> int:
        """The point `x` with `h(x - x*) = w`, by Newton's method."""
        p, K = self.p, self.precision
        modulus = p**K
        w = int(w) % modulus
        self._check(w)
        y = w
        for _ in range(math.ceil(math.log2(K)) + 3):
            slope = self._series(y, derivative=True)
            y = (y - (self._series(y) - w) * pow(slope, -1, modulus)) % modulus
        return (y + self._x) % modulus

    def _exp(self, z: int) -> int:
        """`e^z` modulo `p^K` for `z` divisible by `p` (by `4` when `p = 2`)."""
        p, K = self.p, self.precision
        modulus = p**K
        if z == 0:
            return 1
        vz = valuation(z, p, 10**9)
        total, n, power, factorial = 0, 0, 1, 1
        while n == 0 or n * vz - Fraction(n - 1, p - 1) < K:
            v = valuation(factorial, p, 10**9)
            total += power // p**v * pow(factorial // p**v, -1, modulus)
            n += 1
            power *= z
            factorial *= n
        return total % modulus

    def _log(self, u: int, digits: int) -> int:
        """`log(u)` modulo `p^digits` for `u = 1 + z` with `p` dividing `z`."""
        p = self.p
        modulus = p**digits
        z = (u - 1) % modulus
        if z == 0:
            return 0
        vz = valuation(z, p, 10**9)
        total, n, power = 0, 1, z
        while n * vz - math.log(n, p) < digits + 1:
            v = valuation(n, p, 64)
            term = power // p**v * pow(n // p**v, -1, modulus)
            total += term if n % 2 else -term
            n += 1
            power *= z
        return total % modulus

    def flow_map(self, t, x) -> int:
        """`phi_t(x) = x* + h^{-1}(e^{lambda t} h(x - x*))` modulo `p^K`."""
        growth = self._exp(self._lambda * int(t))
        return self.inverse(self.forward(x) * growth % self.p**self.precision)

    def orbit_time(self, x, z) -> int:
        """The time `t` with `phi_t(x) = z`: `t = log(h(z) / h(x)) / lambda`, exact
        modulo `p^{K - k}` with `k = v(f(x))`.
        """
        p, K = self.p, self.precision
        hx, hz = self.forward(x), self.forward(z)
        s = valuation(hx, p, K)
        if s >= K:
            return 0
        digits = K - s
        modulus = p**digits
        ratio = (hz // p**s) * pow(hx // p**s, -1, modulus) % modulus
        logarithm = self._log(ratio, digits)
        ell = self.multiplier_valuation
        unit = (self._lambda // p**ell) % modulus
        return logarithm // p**ell * pow(unit, -1, modulus) % p ** (digits - ell)
