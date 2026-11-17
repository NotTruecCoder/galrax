from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .arithmetic import is_prime
from .events import EventSet, PAdicEvent
from .polynomial import evaluate_exact, parse_field, reduce
from .return_set import linear, Orbit


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
        non-negative Python integer and returning an integer. A polynomial map of
        `Z_p^d` is given like a vector field, one `{exponents: coefficient}` per
        component: points are then tuples; the cycle tree needs `d = 1`. For
        polynomials `iterate`, `index`, `period` and `hitting_times` go through an
        `Orbit`.
    """

    def __init__(self, p: int, map: list | Callable[[int], int]):
        p = int(p)
        if not is_prime(p):
            raise ValueError("`p` must be a prime.")
        self.p = p
        self.dimension, self.components = 1, None
        if not callable(map) and any(isinstance(c, dict) for c in map):
            components = parse_field(map, p)
            if len(components) == 1:
                (poly,) = components
                top = max((e[0] for e in poly), default=0)
                map = [poly.get((k,), 0) for k in range(top + 1)]
            else:
                self.dimension, self.components = len(components), components
        self._jumps = {}
        if self.components is not None:
            self.coefficients = self._function = None
        elif callable(map):
            self.coefficients = None
            self._function = map
        else:
            coefficients = [Fraction(c) for c in map]
            for c in coefficients:
                if c.denominator % p == 0:
                    raise ValueError("Coefficients must be p-adic integers.")
            self.coefficients = coefficients
            self._function = None

    @property
    def is_polynomial(self) -> bool:
        return self.coefficients is not None

    def __call__(self, x, level: int):
        """`T(x) mod p^level`."""
        modulus = self.p**level
        if self.components is not None:
            point = [int(v) for v in x]
            return tuple(
                evaluate_exact(poly, point, modulus)
                for poly in reduce(self.components, modulus)
            )
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
        if self.components is not None:
            raise ValueError("Cycle trees need a map of Z_p (d = 1).")
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

    def _cached(self, key, build):
        if key not in self._jumps:
            if len(self._jumps) >= 64:
                del self._jumps[next(iter(self._jumps))]
            self._jumps[key] = build()
        return self._jumps[key]

    def orbit(self, x, level: int) -> Orbit:
        """The `Orbit` of a point modulo `p^level`, for a polynomial map (cached)."""
        if not (self.components or self.is_polynomial):
            raise ValueError("Orbits need a polynomial map.")
        x = tuple(int(v) % self.p**level for v in np.atleast_1d(x))
        maps = self.components or self.coefficients
        return self._cached((x, int(level)), lambda: Orbit(self.p, level, maps, x))

    def iterate(self, x, n, level: int):
        """`T^n(x) mod p^level` for a polynomial `T`, at a cost independent of `n`
        once `(x, level)` is set up, through `Orbit`.

        `n` is an integer or an array of integers; negative when `x` is periodic
        modulo `p` (modulo `4` for `p = 2`), giving the preimage on the orbit closure,
        the inverse for a bijection; or a `Fraction` with denominator prime to `p`
        when moreover the step `N` of the orbit is a power of `p`, as for every
        permutation polynomial modulo `2^w`: `n = 1/m` is an `m`-th root.

        The set-up iterates `T` about `N J` times, `N = L ord(A) p^e` with `L` the
        period modulo `p^e` and `J ~ level`: fine for small `p`, slow for large ones.
        The last 64 set-ups are kept.
        """
        orbit = self.orbit(x, level)
        if isinstance(n, Fraction) and n.denominator != 1:
            if orbit.tail:
                raise ValueError("p-adic iterates need a point periodic modulo p.")
            if not _is_power(orbit.step, self.p):
                raise ValueError(
                    "p-adic iterates need a period modulo p that is a power of p."
                )
            if n.denominator % self.p == 0:
                raise ValueError("`n` must be a p-adic integer.")
            # x_{N q + r} is analytic in q, so a p-adic n is read modulo N p^level.
            modulus = orbit.step * self.p**level
            n = n.numerator * pow(n.denominator, -1, modulus) % modulus
        elif isinstance(n, Fraction):
            n = int(n)
        out = orbit(n)
        return int(out) if np.ndim(out) == 0 else out

    def index(self, x, y, level: int) -> int | None:
        """The smallest `n >= 0` with `T^n(x) = y mod p^level`, or `None`."""
        return self.orbit(x, level).index(y)

    def hitting_times(self, x, event: PAdicEvent, level: int) -> EventSet:
        """All `n >= 0` at which the loop `x <- T(x)` on `Z/p^level`, started at `x`,
        is in `Bad = {v_p(g) >= depth for every condition g of the event}`.

        The answer is a union of classes `n = c mod M p^k`, `M` prime to `p`, from
        the end of the tail on, plus the hits in the tail: see `Orbit.times`. Since
        `Z` lies in `Z_p`, an empty set for integer `T`, `x` and `g` proves that the
        loop over `Z` never reaches `g = 0`.
        """
        return self.orbit(x, level).times(event)

    def period(self, x, level: int) -> int:
        """The least period of `x` modulo `p^level`, for `x` periodic modulo `p`
        (modulo `4` for `p = 2`): `p^level` for a minimal map; for a Chebyshev
        permutation polynomial and `p > 3`, `N_s l_s p^{level - v}` once
        `level >= v` (Li, Lu, Tan, Chen, arXiv:2209.14658, Theorem 1).
        """
        y = [int(v) % self.p**level for v in np.atleast_1d(x)]
        d = len(y)
        conditions = [
            {tuple(int(j == i) for j in range(d)): 1, (0,) * d: -v}
            for i, v in enumerate(y)
        ]
        times = self.hitting_times(y, PAdicEvent(conditions), level)
        # The times form P Z: classes k P modulo M p^level, all of one level.
        return min(b.center[0] or times.period * self.p**b.level for b in times.balls)


def matrix_power_index(p: int, precision: int, matrix, start, target, coordinate=0):
    """The smallest `n >= 0` with `(M^n v)_coordinate = target mod p^K`, or `None`,
    for `M` invertible modulo `p`.

    Chebyshev, Lucas and Dickson sequences are all `(M^n v)_0` for a `2 x 2` `M`:
    `T_n(x)` with `M = [[0, 1], [-1, 2x]]`, `v = (1, x)`; `V_n(P, Q)` with
    `[[0, 1], [-Q, P]]`, `(2, P)`; `D_n(x, a)` with `[[0, 1], [-a, x]]`, `(2, x)`.
    """
    d = len(matrix)
    unit = tuple(int(j == coordinate) for j in range(d))
    event = PAdicEvent({unit: 1, (0,) * d: -int(target)})
    return Orbit(p, precision, linear(matrix), start).times(event).first()


def _is_power(n: int, p: int) -> bool:
    while n % p == 0:
        n //= p
    return n == 1
