import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .arithmetic import factorial_valuation, is_prime
from .events import EventSet, PAdicEvent
from .polynomial import parse_polynomial, reduce
from .roots import _merge, Ball, vanishing_set


@dataclass
class Cycle:
    """A cycle of the map on `Z/p^level`, and how it lifts to `Z/p^{level+1}`.

    - `points`: the cycle, in order, as residues modulo `p^level`.
    - `fiber`: the map induced by `T^L` on the `p` lifts of `points[0]`, as the list of
        images of `0, ..., p - 1`, where `L = len(points)`. For a polynomial it is the
        affine map `t -> a t + b` with `a = (T^L)'` and `b = (T^L(x) - x) / p^level`.
    - `behaviour`: `"grows"` (one lift, `p` times longer), `"splits"` (`p` lifts of
        the same length), `"partially splits"`, `"grows tails"` (one periodic lift,
        the other lifts fall onto it), or `"permutes"` for a general fibre map.
    - `children`: the cycles it lifts to.
    - `final`: set when the lifts are known for every higher level, see
        `PAdicMap.cycle_tree`.
    """

    level: int
    points: tuple
    fiber: tuple = ()
    multiplier: int | None = None
    behaviour: str = ""
    children: list = field(default_factory=list)
    final: str | None = None

    @property
    def length(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class Component:
    """A piece of the minimal decomposition of the periodic part of the map.

    - `kind`: `"minimal"`: the union of the balls `points[i] + p^level Z_p`, on which
        the map is minimal (every orbit is dense) and conjugate to the odometer;
        `"attracting"` or `"periodic"`: a single periodic orbit, of period
        `len(points)`, given modulo `p^level`; `"unresolved"`: a cycle at the last
        level computed whose fate is not decided.
    """

    kind: str
    level: int
    points: tuple


class PAdicMap:
    """A discrete dynamical system `x -> T(x)` on the `p`-adic integers, through its
    cycles on every `Z/p^k` (Fan--Liao, Desjardins--Zieve, Anashin).

    `T` must be 1-Lipschitz: `T(x) mod p^k` depends on `x mod p^k` only. Polynomials
    with `p`-adic integer coefficients are, and so are T-functions built from `+`,
    `*`, `&`, `|`, `^` on 2-adic integers. Each cycle of `T` on `Z/p^k` acts on the
    `p` balls of radius `p^{-k-1}` above its first point through a map of `Z/p`; the
    cycles of that map give the cycles on `Z/p^{k+1}`. Doing this level by level
    builds the cycle tree.

    For polynomials the fibre map is affine, `t -> a t + b`, and growth is permanent
    beyond a fixed level: a growing cycle at level `k` grows at every higher level if
    `p >= 5`, or `p = 3` and `k >= 2`, or `p = 2`, `k >= 2` and `a = 1 mod 4`
    (Desjardins--Zieve; Fan--Liao). The balls of such a cycle form a minimal
    component, so the tree stops there. Attracting and indifferent periodic orbits
    are followed down to the last level.

    **Arguments:**

    - `p`: a prime.
    - `map`: the polynomial `[c_0, c_1, ...]` (coefficients `p`-adic integers, e.g.
        `Fraction`s with denominators prime to `p`), or a function taking a
        non-negative Python integer and returning an integer.
    """

    def __init__(self, p: int, map: list | Callable[[int], int]):
        p = int(p)
        if not is_prime(p):
            raise ValueError("`p` must be a prime.")
        self.p = p
        if callable(map):
            self.coefficients = None
            self._function = map
        else:
            coefficients = [Fraction(c) for c in map]
            for c in coefficients:
                if c.denominator % p == 0:
                    raise ValueError("Coefficients must be p-adic integers.")
            self.coefficients = coefficients
            self._function = None
        self._jumps = {}

    @property
    def is_polynomial(self) -> bool:
        return self.coefficients is not None

    def __call__(self, x: int, level: int) -> int:
        """`T(x) mod p^level`."""
        modulus = self.p**level
        if self._function is not None:
            return int(self._function(int(x) % modulus)) % modulus
        total = 0
        for c in reversed(self.coefficients):
            unit = c.numerator * pow(c.denominator, -1, modulus)
            total = (total * x + unit) % modulus
        return total

    def derivative(self, x: int, level: int) -> int:
        """`T'(x) mod p^level` for a polynomial."""
        if not self.is_polynomial:
            raise ValueError("Only polynomial maps have a derivative.")
        modulus = self.p**level
        total = 0
        for k in reversed(range(1, len(self.coefficients))):
            c = self.coefficients[k] * k
            unit = c.numerator * pow(c.denominator, -1, modulus)
            total = (total * x + unit) % modulus
        return total

    def table(self, level: int) -> np.ndarray:
        """`T` on all of `Z/p^level`, as an integer array."""
        return np.array([self(x, level) for x in range(self.p**level)], dtype=np.int64)

    def is_bijective(self, level: int = 2) -> bool:
        """Whether `T` permutes `Z/p^level`. For a polynomial, a permutation of
        `Z/p^2` permutes every `Z/p^k` and preserves the Haar measure of `Z_p`.
        """
        image = self.table(level)
        return len(np.unique(image)) == len(image)

    #
    # Cycles
    #

    def _orbit(self, x: int, length: int, level: int) -> tuple:
        points = [x]
        for _ in range(length - 1):
            points.append(self(points[-1], level))
        return tuple(points)

    def _fiber(self, cycle: Cycle) -> tuple:
        p, k, L = self.p, cycle.level, cycle.length
        x0 = cycle.points[0]
        images = []
        for t in range(p):
            y = x0 + p**k * t
            for _ in range(L):
                y = self(y, k + 1)
            images.append(((y - x0) // p**k) % p)
        return tuple(images)

    @staticmethod
    def _periodic_cycles(images: tuple) -> list:
        """The cycles of a map of `{0, ..., n - 1}`, given as the list of images."""
        n = len(images)
        seen = [False] * n
        cycles = []
        for start in range(n):
            path, x = [], start
            index = {}
            while not seen[x] and x not in index:
                index[x] = len(path)
                path.append(x)
                x = images[x]
            if x in index:
                cycles.append(path[index[x] :])
            for y in path:
                seen[y] = True
        return cycles

    def _classify(self, cycle: Cycle) -> None:
        p, k = self.p, cycle.level
        fiber = cycle.fiber
        if self.is_polynomial:
            a = 1
            digits = 2 if p == 2 else 1
            for x in cycle.points:
                a = a * self.derivative(x, digits) % p**digits
            cycle.multiplier = a
            b = fiber[0]
            if a % p == 0:
                cycle.behaviour = "grows tails"
            elif a % p != 1:
                cycle.behaviour = "partially splits"
            else:
                cycle.behaviour = "grows" if b else "splits"
            if cycle.behaviour == "grows":
                if p >= 5 or (p == 3 and k >= 2) or (p == 2 and k >= 2 and a == 1):
                    cycle.final = "minimal"
            return
        orbits = self._periodic_cycles(fiber)
        if len(orbits) == 1 and len(orbits[0]) == p:
            cycle.behaviour = "grows"
        elif all(len(o) == 1 for o in orbits) and len(orbits) == p:
            cycle.behaviour = "splits"
        else:
            cycle.behaviour = "permutes"

    def _lift(self, cycle: Cycle) -> list:
        p, k = self.p, cycle.level
        x0 = cycle.points[0]
        children = []
        for orbit in self._periodic_cycles(cycle.fiber):
            start = x0 + p**k * orbit[0]
            length = cycle.length * len(orbit)
            children.append(Cycle(k + 1, self._orbit(start, length, k + 1)))
        return children

    def cycle_tree(self, max_level: int, max_cycles: int = 100_000) -> list[Cycle]:
        """The cycles on `Z/p`, each with its descendants down to `Z/p^max_level`.

        A cycle marked `final = "minimal"` is not expanded: its lifts at every level
        are single cycles covering its balls.
        """
        max_level = int(max_level)
        if max_level < 1:
            raise ValueError("`max_level` must be at least 1.")
        image = [self(x, 1) for x in range(self.p)]
        roots = [
            Cycle(1, self._orbit(orbit[0], len(orbit), 1))
            for orbit in self._periodic_cycles(tuple(image))
        ]
        stack, count = list(roots), len(roots)
        while stack:
            cycle = stack.pop()
            if cycle.level >= max_level:
                continue
            cycle.fiber = self._fiber(cycle)
            self._classify(cycle)
            if cycle.final:
                continue
            cycle.children = self._lift(cycle)
            count += len(cycle.children)
            if count > max_cycles:
                raise RuntimeError(
                    "Too many cycles: lower `max_level` or raise `max_cycles`."
                )
            stack.extend(cycle.children)
        return roots

    def cycles(self, level: int) -> list[tuple]:
        """All cycles of `T` on `Z/p^level`, from the cycle tree. Minimal components
        found earlier contribute one cycle covering all their balls.
        """
        level = int(level)
        out = []
        stack = self.cycle_tree(level)
        while stack:
            cycle = stack.pop()
            if cycle.level == level:
                out.append(cycle.points)
            elif cycle.final == "minimal":
                length = cycle.length * self.p ** (level - cycle.level)
                out.append(self._orbit(cycle.points[0], length, level))
            else:
                stack.extend(cycle.children)
        return sorted(out)

    def minimal_decomposition(self, max_level: int) -> list[Component]:
        """The minimal components of the periodic part of `T` found by
        `max_level`: minimal clopen sets (unions of balls), periodic orbits, and
        cycles left unresolved at `max_level`.
        """
        out = []
        stack = self.cycle_tree(max_level)
        while stack:
            cycle = stack.pop()
            if cycle.final == "minimal":
                out.append(Component("minimal", cycle.level, cycle.points))
            elif cycle.level >= max_level:
                kind = "unresolved"
                if self.is_polynomial and cycle.level > 1:
                    a = 1
                    for x in cycle.points:
                        a = a * self.derivative(x, 1) % self.p
                    if a == 0:
                        kind = "attracting"
                    elif a != 1:
                        kind = "periodic"
                out.append(Component(kind, cycle.level, cycle.points))
            else:
                stack.extend(cycle.children)
        return sorted(out, key=lambda c: (c.level, c.points))

    def is_minimal(self, max_level: int = 4) -> bool:
        """Whether `T` is minimal (equivalently, for a bijection preserving the Haar
        measure, ergodic) on all of `Z_p`: a single cycle on `Z/p` whose lifts keep
        growing. For polynomials the answer is exact once the growth is permanent;
        for other maps it means transitivity on `Z/p^max_level`.
        """
        roots = self.cycle_tree(max_level)
        if len(roots) != 1 or roots[0].length != self.p:
            return False
        cycle = roots[0]
        while True:
            if cycle.final == "minimal":
                return True
            if cycle.level >= max_level:
                return not self.is_polynomial
            if cycle.behaviour != "grows" or len(cycle.children) != 1:
                return False
            cycle = cycle.children[0]

    #
    # Jump-ahead
    #

    def _jump(self, x: int, level: int):
        key = (int(x) % self.p**level, int(level))
        if key not in self._jumps:
            self._jumps[key] = self._jump_table(*key)
        return self._jumps[key]

    def _jump_table(self, x: int, level: int):
        """The tail `x_0, ..., x_{s-1}` before the orbit turns periodic modulo `p^e`
        (`e = 2` for `p = 2`, else `1`), a step `N`, and for every `r < N` the
        polynomial in `q` equal to `x_{s + r + N q}` modulo `p^level`.

        With `L` the period modulo `p^e` and `a` the multiplier of that cycle,
        `T^N` for `N = L ord(a) p^e` is the identity modulo `p^e` on the ball around
        `x_s`, so `q -> x_{s + r + N q}` is a power series whose `q^m` coefficient
        has valuation `>= m (e - 1/(p - 1))`. Its Mahler coefficients come from the
        first `J` terms, `J (e - 1/(p - 1)) >= level`.
        """
        if not self.is_polynomial:
            raise ValueError("Jump-ahead needs a polynomial map.")
        p = self.p
        e = 2 if p == 2 else 1
        seen, y = {}, x % p**e
        while y not in seen:
            seen[y] = len(seen)
            y = self(y, e)
        start = seen[y]
        a = 1
        for point in list(seen)[start:]:
            a = a * self.derivative(point, e) % p**e
        if a % p == 0:
            raise ValueError("The orbit falls into an attracting cycle modulo p.")
        order = 1
        while pow(a, order, p**e) != 1:
            order += 1
        step = (len(seen) - start) * order * p**e
        terms = -(-level * (p - 1) // (e * (p - 1) - 1))
        work = level + factorial_valuation(terms - 1, p)
        orbit = [x % p**work]
        for _ in range(start + step * terms - 1):
            orbit.append(self(orbit[-1], work))
        m = p**level
        tail, orbit = [y % m for y in orbit[:start]], orbit[start:]
        tables = []
        for r in range(step):
            differences = orbit[r::step]
            poly, falling = [0] * terms, [1]
            for j in range(terms):
                v = factorial_valuation(j, p)
                if differences[0] % p**v:
                    raise RuntimeError(f"Delta^{j} is not divisible by {j}!.")
                unit = math.factorial(j) // p**v
                b = differences[0] // p**v * pow(unit, -1, m)
                for i, c in enumerate(falling):
                    poly[i] = (poly[i] + b * c) % m
                falling = [0] + falling
                for i in range(j + 1):
                    falling[i] = (falling[i] - j * falling[i + 1]) % m
                differences = [u - w for w, u in zip(differences, differences[1:])]
            tables.append(poly)
        return tail, step, tables

    def iterate(self, x: int, n, level: int) -> int:
        """`T^n(x) mod p^level` for a polynomial `T`, at a cost independent of `n`
        once `(x, level)` is set up. `n` is a non-negative integer, or any `p`-adic
        integer (negative, or a `Fraction` with denominator prime to `p`) when the
        orbit of `x` carries a `Z_p`-action, as for every permutation polynomial
        modulo `2^w`: `n = -1` is the inverse, `n = 1/m` an `m`-th root.
        """
        tail, step, tables = self._jump(x, level)
        n = Fraction(n)
        if n.denominator == 1 and 0 <= n < len(tail):
            return tail[int(n)]
        if n.denominator == 1 and n >= 0:
            q, r = divmod(int(n) - len(tail), step)
        elif tail or not _is_power(step, self.p):
            raise ValueError(
                "Negative and p-adic iterates need an orbit that is periodic modulo p "
                "with a period that is a power of p."
            )
        elif n.denominator % self.p == 0:
            raise ValueError("`n` must be a p-adic integer.")
        else:
            modulus = step * self.p**level
            t = n.numerator * pow(n.denominator, -1, modulus) % modulus
            q, r = divmod(t, step)
        m = self.p**level
        total = 0
        for c in reversed(tables[r]):
            total = (total * q + c) % m
        return total

    def _hits(self, x: int, conditions: list, level: int, depth: int):
        """For each `r`, the balls of `q` with `v_p(g(x_{s + r + N q})) >= depth` for
        every `g`: `g(x_{s + r + N q})` is a polynomial in `q`, refined digit by digit.
        """
        tail, step, tables = self._jump(x, level)
        if not 1 <= depth <= level:
            raise ValueError(f"`depth` must lie in 1, ..., {level}.")
        m = self.p**level
        conditions = reduce([parse_polynomial(g, 1, self.p) for g in conditions], m)
        hits = []
        for r, f in enumerate(tables):
            polys = []
            for g in conditions:
                h = [0] * len(f)
                for k in reversed(range(max((e for (e,) in g), default=0) + 1)):
                    h = _product(h, f, m)
                    h[0] = (h[0] + g.get((k,), 0)) % m
                polys.append({(i,): c for i, c in enumerate(h) if c})
            hits += [(r, b) for b in vanishing_set(polys, 1, self.p, level, depth)]
        return tail, step, hits

    def index(self, x: int, y: int, level: int) -> int | None:
        """The smallest `n >= 0` with `T^n(x) = y mod p^level`, or `None`."""
        tail, step, hits = self._hits(x, [[-int(y), 1]], level, level)
        y = int(y) % self.p**level
        if y in tail:
            return tail.index(y)
        return min((len(tail) + r + step * b.center[0] for r, b in hits), default=None)

    def hitting_times(self, x: int, event: PAdicEvent, level: int) -> EventSet:
        """All `n >= 0` at which the loop `x <- T(x)` on `Z/p^level`, started at `x`,
        is in `Bad = {v_p(g) >= depth for every condition g of the event}`.

        The answer is a union of classes `n = c mod M p^k`, `M` prime to `p` (`M = 1`
        for `p = 2`): an `EventSet` with `period = M`. Since `Z` lies in `Z_p`, an
        empty set for integer `T`, `x` and `g` proves that the loop over `Z` never
        reaches `g = 0`. `x` must be periodic modulo `p` (modulo `4` for `p = 2`).
        """
        depth = level if event.depth is None else event.depth
        tail, step, hits = self._hits(x, event.condition, level, depth)
        if tail:
            raise ValueError("`x` is not periodic modulo p: start the loop later.")
        s = 0
        while step % self.p ** (s + 1) == 0:
            s += 1
        period = step // self.p**s
        balls = [Ball((r + step * b.center[0],), b.level + s) for r, b in hits]
        if period == 1:
            balls = _merge(balls, self.p, 1)
        balls = sorted(balls, key=lambda b: b.center)
        return EventSet(balls, self.p, depth, None, period)


def _is_power(n: int, p: int) -> bool:
    while n % p == 0:
        n //= p
    return n == 1


def _product(a: list, b: list, m: int) -> list:
    """`a b mod m`, cut to `len(a)` terms."""
    out = [0] * len(a)
    for i, c in enumerate(a):
        if c:
            for j in range(len(a) - i):
                out[i + j] = (out[i + j] + c * b[j]) % m
    return out
