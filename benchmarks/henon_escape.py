"""The archimedean baseline for `nonlinear_skolem.py`: the same 200 orbits of
generalised Henon maps `(x, y) -> (y, c x + P(y))` of `Z^2`, solved by escape.

With `R = |a_1| + |a_0| + 3`, the region `V+ = {|y| >= max(|x|, R)}` is forward
invariant and `|y|` increases strictly in it, since there
`|y'| >= |y|^2 - (|a_1| + 1)|y| - |a_0| > |y|`; the same holds for
`V- = {|x| >= max(|y|, R)}` under the inverse map `(x, y) -> (c (y - P(x)), x)`.
Off `V+` and `V-` the orbit is in the finite box `|x|, |y| < R`. So an integer
orbit is periodic or leaves the box for good in both directions, and once
`|y_n| > |target|` in `V+` no later `x_m = y_{m-1}` can equal the target
(backwards the same with `x`). This decides every case exactly, with an
explicit bound on `|n|`, in integer arithmetic.

    python benchmarks/henon_escape.py --count 200
"""

import argparse
import time

from nonlinear_skolem import make_cases


def solve(c, P, start, target):
    """`(zeros, periodic)`: every `n` in `Z` with `x_n = target`."""
    a0, a1, _ = P
    R = abs(a1) + abs(a0) + 3

    def poly(y):
        return sum(a * y**i for i, a in enumerate(P))

    bound = max(R, abs(target) + 1)
    zeros = []

    def forward(x, y):
        return y, c * x + poly(y)

    def backward(x, y):
        return c * (y - poly(x)), x

    for sign, step, escaped in (
        (1, forward, lambda x, y: abs(y) >= max(abs(x), bound)),
        (-1, backward, lambda x, y: abs(x) >= max(abs(y), bound)),
    ):
        point, n = start, 0
        while True:
            if point[0] == target and (n or sign == 1):
                zeros.append(sign * n)
            if escaped(*point):  # in V+ (resp. V-) beyond the target: no later zero
                break
            point, n = step(*point), n + 1
            if point == start:  # back to the start: the orbit is periodic over Z
                period = n
                return sorted({m for m in zeros if 0 <= m < period}), period
    return sorted(set(zeros)), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    began = time.perf_counter()
    rows = [solve(*case[1:]) for case in make_cases(args.count, args.seed)]
    seconds = time.perf_counter() - began
    periodic = sum(1 for _, per in rows if per)
    hitting = sum(1 for z, per in rows if z and not per)
    print(
        f"{len(rows)} orbits decided in {seconds * 1e3:.1f} ms: {periodic} periodic, "
        f"{hitting} with zeros, {len(rows) - periodic - hitting} never reached"
    )
    largest = max((abs(n) for z, _ in rows for n in z), default=0)
    print("largest |n| of a zero:", largest)


if __name__ == "__main__":
    main()
