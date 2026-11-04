import math
from collections.abc import Callable
from fractions import Fraction

from .arithmetic import simplex, valuation


def _valuation(c, p: int) -> int | float:
    c = Fraction(c)
    if c == 0:
        return math.inf
    if c.denominator % p == 0:
        raise ValueError("Coefficients must be p-adic integers.")
    return valuation(c.numerator, p, cap=abs(c.numerator).bit_length() + 1)


def _digits_of_log(n: int, p: int) -> int:
    """`floor(log_p n)` for `n >= 1`."""
    k, power = 0, p
    while power <= n:
        k += 1
        power *= p
    return k


class PowerSeries:
    """A power series `sum_e a_e x^e` in `d` variables whose coefficients are
    `p`-adic integers tending to zero, so that it converges on all of `Z_p^d`.

    Modulo every `p^n` such a series is a polynomial, and that is how it enters a
    `PAdicFlow`: it is cut to the terms that can be non-zero modulo `p^{K + g}`, the
    working precision plus the guard digits of the Lie series, so the flow of an
    analytic field is as exact modulo `p^K` as that of a polynomial one.

    Series add, subtract and multiply with each other, with polynomials (given as
    `{exponents: coefficient}`) and with numbers. Besides `exp`, `log1p`,
    `geometric`, `polynomial` and `variable`, any series can be given directly.

    **Arguments:**

    - `coefficient`: `coefficient(e)` is the coefficient of `x^e`, an integer or a
        fraction whose denominator is prime to `p`.
    - `bound`: `bound(n, p)` is a lower bound for `v_p(a_e)` over every `e` of total
        degree at least `n`; it must tend to infinity.
    - `dimension`: `d`.
    """

    def __init__(
        self,
        coefficient: Callable[[tuple[int, ...]], int | Fraction],
        bound: Callable[[int, int], int | float],
        dimension: int = 1,
    ):
        dimension = int(dimension)

        def terms(p, digits):
            n = 0
            while bound(n, p) < digits:
                n += 1
                if n > 100_000:
                    raise ValueError(
                        "The series does not converge on Z_p: `bound` does not grow."
                    )
            out = {}
            for e in simplex(dimension, n - 1) if n else []:
                c = Fraction(coefficient(e))
                if c:
                    out[e] = c
            return out

        self._setup(dimension, terms, lambda p: bound(0, p))

    def _setup(self, dimension, terms, lowest):
        self.dimension = dimension
        self._terms = terms
        self._lowest = lowest

    @classmethod
    def _build(cls, dimension, terms, lowest) -> "PowerSeries":
        out = cls.__new__(cls)
        out._setup(dimension, terms, lowest)
        return out

    def terms(self, p: int, digits: int) -> dict:
        """The polynomial that agrees with the series modulo `p^digits`, as
        `{exponents: coefficient}`.
        """
        return dict(self._terms(int(p), int(digits)))

    def lowest(self, p: int) -> int | float:
        """A lower bound for the valuation of every coefficient."""
        return self._lowest(int(p))

    def __repr__(self) -> str:
        return f"PowerSeries(dimension={self.dimension})"

    #
    # Constructors
    #

    @classmethod
    def polynomial(cls, poly, dimension: int | None = None) -> "PowerSeries":
        """A polynomial, `{exponents: coefficient}` or (for `d = 1`) the list
        `[c_0, c_1, ...]`.
        """
        if not isinstance(poly, dict):
            poly = {(i,): c for i, c in enumerate(poly)}
        poly = {
            tuple(int(k) for k in (e if isinstance(e, tuple) else (e,))): Fraction(c)
            for e, c in poly.items()
            if c
        }
        if dimension is None:
            dimension = len(next(iter(poly))) if poly else 1
        if any(len(e) != dimension for e in poly):
            raise ValueError(f"Exponents must have {dimension} entries.")

        def lowest(p):
            return min((_valuation(c, p) for c in poly.values()), default=math.inf)

        return cls._build(dimension, lambda p, digits: poly, lowest)

    @classmethod
    def variable(cls, index: int = 0, dimension: int = 1) -> "PowerSeries":
        """The coordinate `x_index`."""
        exponents = tuple(int(j == index) for j in range(dimension))
        return cls.polynomial({exponents: 1}, dimension)

    @classmethod
    def exp(cls, a, variable: int = 0, dimension: int = 1) -> "PowerSeries":
        """`exp(a x_variable) = sum_n a^n x^n / n!`, for `v_p(a) > 1/(p - 1)`."""
        a = Fraction(a)

        def coefficient(e):
            n = e[variable]
            if sum(e) != n:
                return 0
            return a**n / math.factorial(n)

        def bound(n, p):
            v = _valuation(a, p)
            if v <= Fraction(1, p - 1):
                raise ValueError(
                    "exp(a x) converges on Z_p only for v_p(a) > 1/(p - 1)."
                )
            if n == 0 or v == math.inf:
                return 0 if n == 0 else math.inf
            return math.ceil(n * v - Fraction(n - 1, p - 1))

        return cls(coefficient, bound, dimension)

    @classmethod
    def log1p(cls, a, variable: int = 0, dimension: int = 1) -> "PowerSeries":
        """`log(1 + a x_variable) = sum_{n >= 1} (-1)^{n+1} a^n x^n / n`, for
        `v_p(a) >= 1`.
        """
        a = Fraction(a)

        def coefficient(e):
            n = e[variable]
            if n == 0 or sum(e) != n:
                return 0
            return (-1) ** (n + 1) * a**n / n

        def bound(n, p):
            v = _valuation(a, p)
            if v < 1:
                raise ValueError("log(1 + a x) converges on Z_p only for v_p(a) >= 1.")
            n = max(n, 1)
            return n * v - _digits_of_log(n, p)

        return cls(coefficient, bound, dimension)

    @classmethod
    def geometric(cls, a, variable: int = 0, dimension: int = 1) -> "PowerSeries":
        """`1 / (1 - a x_variable) = sum_n a^n x^n`, for `v_p(a) >= 1`."""
        a = Fraction(a)

        def coefficient(e):
            n = e[variable]
            return a**n if sum(e) == n else 0

        def bound(n, p):
            v = _valuation(a, p)
            if v < 1:
                raise ValueError("1 / (1 - a x) converges on Z_p only for v_p(a) >= 1.")
            return n * v

        return cls(coefficient, bound, dimension)

    #
    # Arithmetic
    #

    def _coerce(self, other) -> "PowerSeries":
        if isinstance(other, PowerSeries):
            if other.dimension != self.dimension:
                raise ValueError("Series must have the same number of variables.")
            return other
        if isinstance(other, dict):
            return PowerSeries.polynomial(other, self.dimension)
        return PowerSeries.polynomial({(0,) * self.dimension: other}, self.dimension)

    def __add__(self, other) -> "PowerSeries":
        other = self._coerce(other)

        def terms(p, digits):
            out = dict(self._terms(p, digits))
            for e, c in other._terms(p, digits).items():
                out[e] = out.get(e, 0) + c
            return {e: c for e, c in out.items() if c}

        def lowest(p):
            return min(self._lowest(p), other._lowest(p))

        return PowerSeries._build(self.dimension, terms, lowest)

    __radd__ = __add__

    def __neg__(self) -> "PowerSeries":
        return self * -1

    def __sub__(self, other) -> "PowerSeries":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "PowerSeries":
        return self._coerce(other) - self

    def __mul__(self, other) -> "PowerSeries":
        other = self._coerce(other)

        def terms(p, digits):
            first = self._terms(p, digits - min(other._lowest(p), digits))
            second = other._terms(p, digits - min(self._lowest(p), digits))
            out = {}
            for ea, ca in first.items():
                for eb, cb in second.items():
                    e = tuple(i + j for i, j in zip(ea, eb))
                    out[e] = out.get(e, 0) + ca * cb
            return {e: c for e, c in out.items() if c and _valuation(c, p) < digits}

        def lowest(p):
            return self._lowest(p) + other._lowest(p)

        return PowerSeries._build(self.dimension, terms, lowest)

    __rmul__ = __mul__
