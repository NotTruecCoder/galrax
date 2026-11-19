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
explicit bound on `|n|`, in integer arithmetic: `galrax.padic.adelic.henon_real`.

    python benchmarks/henon_escape.py --count 200
"""

import argparse
import time

from galrax.padic.adelic import henon_real
from nonlinear_skolem import make_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    began = time.perf_counter()
    rows = [henon_real(*case[1:])[1:] for case in make_cases(args.count, args.seed)]
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
