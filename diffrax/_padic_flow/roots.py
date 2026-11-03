import itertools
from dataclasses import dataclass

from .arithmetic import valuation
from .polynomial import taylor_shift


@dataclass(frozen=True)
class Ball:
    """The ball `center + p^level Z_p^d`, with `0 <= center_i < p^level`."""

    center: tuple
    level: int

    def contains(self, point, p: int) -> bool:
        modulus = p**self.level
        return all((x - c) % modulus == 0 for x, c in zip(point, self.center))

    def intersect(self, other: "Ball", p: int) -> "Ball | None":
        small, large = (self, other) if self.level >= other.level else (other, self)
        return small if large.contains(small.center, p) else None


def _classify(polys, center, level, depth, p, precision):
    """`"inside"` if every polynomial has valuation `>= depth` on the whole ball,
    `"outside"` if one of them has a constant valuation `< depth` on it, `None` if
    the ball must be split.
    """
    modulus = p**precision
    inside = True
    for poly in polys:
        shifted = taylor_shift(poly, center, modulus)
        constant = valuation(shifted.get((0,) * len(center), 0), p, precision)
        lowest = min(
            (
                valuation(c, p, precision) + level * sum(e)
                for e, c in shifted.items()
                if any(e)
            ),
            default=precision + level,
        )
        if min(constant, lowest) < depth:
            inside = False
        if constant < depth and constant < lowest:
            return "outside"
    return "inside" if inside else None


def _merge(balls: list[Ball], p: int, d: int) -> list[Ball]:
    balls = set(balls)
    changed = True
    while changed:
        changed = False
        parents = {}
        for ball in balls:
            if ball.level == 0:
                continue
            modulus = p ** (ball.level - 1)
            parent = Ball(tuple(c % modulus for c in ball.center), ball.level - 1)
            parents.setdefault(parent, []).append(ball)
        for parent, children in parents.items():
            if len(children) == p**d:
                balls -= set(children)
                balls.add(parent)
                changed = True
    return sorted(balls, key=lambda b: (b.level, b.center))


def vanishing_set(
    polys, d: int, p: int, precision: int, depth: int, max_balls: int = 100_000
) -> list[Ball]:
    """The set `{x in Z_p^d : v_p(g(x)) >= depth for every g in polys}` as a sorted
    list of disjoint balls, for polynomials with integer coefficients known modulo
    `p^precision` and `depth <= precision`.

    Balls are refined digit by digit. A ball is kept when the Taylor expansion of
    every polynomial around its centre shows valuation `>= depth` throughout, and
    dropped when one polynomial has a constant valuation below `depth` on it, which
    happens as soon as its constant term dominates. At `level = depth` one of the two
    always holds, so the search stops.
    """
    if not 0 <= depth <= precision:
        raise ValueError(f"`depth` must lie in 0, ..., {precision}.")
    found, stack = [], [((0,) * d, 0)]
    visited = 0
    while stack:
        center, level = stack.pop()
        visited += 1
        if visited > max_balls:
            raise RuntimeError(
                "The vanishing set has too many pieces: the polynomials may vanish "
                "on a curve. Lower `depth` or raise `max_balls`."
            )
        verdict = _classify(polys, center, level, depth, p, precision)
        if verdict == "inside":
            found.append(Ball(center, level))
        elif verdict is None:
            step = p**level
            for digits in itertools.product(range(p), repeat=d):
                child = tuple(c + step * k for c, k in zip(center, digits))
                stack.append((child, level + 1))
    return _merge(found, p, d)
