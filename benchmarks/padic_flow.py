"""Throughput and set-up times of the exact p-adic flows.

    python benchmarks/padic_flow.py [--batch 1000000] [--repeat 20] [--quick]

Each throughput is the median of `--repeat` calls on one batch of random `(t, x)`
modulo `p^K`, after 5 warm-up calls, in millions of points per second; on a laptop the
GPU clock changes under load, so single timings are not reliable. The full Lie series
is timed on a hundredth of the batch, which it would not fit in memory otherwise, and
only up to degree 40. Every method is checked against the full series before it is
timed. Under WSL2 run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
"""

import argparse
import statistics
import time

import jax
import numpy as np


jax.config.update("jax_enable_x64", True)

import diffrax.padic_flow as pf  # noqa: E402


FLOWS = [
    ("3, 19, 1", 3, 19, [3, -3, 0, 6]),
    ("5, 13, 1", 5, 13, [5, 0, -5]),
    ("7, 11, 1", 7, 11, [0, 7, 14]),
    ("3, 38, 1 (two words)", 3, 38, [3, -3, 0, 6]),
    ("3, 12, 2", 3, 12, [{(0, 1): 3, (2, 0): 3}, {(0, 0): 3, (1, 1): -6}]),
    ("3, 60, 1 (limbs)", 3, 60, [3, -3, 0, 6]),
]


def _median(fn, repeat: int) -> float:
    for _ in range(5):
        jax.block_until_ready(fn())
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _inputs(flow, batch: int, seed: int = 0):
    p, K, d = flow.p, flow.precision, flow.dimension
    rng = np.random.default_rng(seed)
    shape = (batch,) if d == 1 else (batch, d)
    if p**K < 2**62:
        t = rng.integers(0, p**K, batch)
        x = rng.integers(0, p**K, shape)
    else:
        digits = 18
        t = rng.integers(0, p**digits, batch).astype(object) * p ** (K - digits) + 7
        x = rng.integers(0, p**digits, shape).astype(object) * p ** (K - digits) + 5
    return t, x


def flows(batch: int, repeat: int) -> None:
    print("| p, K, d | build (s) | degree → table | flow (M/s) | series (M/s) |")
    print("|---|---|---|---|---|")
    for name, p, K, field in FLOWS:
        start = time.perf_counter()
        flow = pf.PAdicFlow(p, K, field)
        build = time.perf_counter() - start
        size = batch if p**K < 2**62 else batch // 100
        t, x = _inputs(flow, size)
        table = np.asarray(flow.flow(t, x), dtype=object)
        check = min(size, 10_000)
        series = flow.flow(t[:check], x[:check], method="taylor")
        assert np.array_equal(table[:check], np.asarray(series, dtype=object))
        slow_seconds = float("nan")
        if p**K < 2**62:
            fast = jax.jit(flow.flow)
            seconds = _median(lambda: fast(t, x), repeat)
        else:
            seconds = _median(lambda: flow.flow(t, x), max(repeat // 4, 1))
        # Unrolled Horner over the full series of degree ~2K does not compile in a
        # few GB of memory beyond about degree 40.
        if p**K < 2**62 and flow.degree <= 40:
            slow = jax.jit(lambda t, x: flow.flow(t, x, method="taylor"))
            small = max(size // 100, 1)
            slow_seconds = _median(lambda: slow(t[:small], x[:small]), repeat) * 100
        print(
            f"| {name} | {build:.1f} | {flow.degree} → {flow.table_degree} | "
            f"{size / seconds / 1e6:.3g} | {size / slow_seconds / 1e6:.3g} |"
        )


def rectify(batch: int, repeat: int) -> None:
    print("\n| p, K, d | rectify (M/s) |")
    print("|---|---|")
    for name, p, K, field in FLOWS[:5]:
        flow = pf.PAdicFlow(p, K, field)
        t, x = _inputs(flow, batch)
        z = flow.flow(t, x)
        s, w = flow.rectify(x, z)
        assert np.array_equal(np.asarray(flow.flow(s, w)), np.asarray(z))
        fast = jax.jit(flow.rectify)
        seconds = _median(lambda: fast(x, z), repeat)
        print(f"| {name} | {batch / seconds / 1e6:.3g} |")


def interpolation(batch: int) -> None:
    print("\n| time-one map | digits | field (s) | flow (s) | chart (k/s) |")
    print("|---|---|---|---|---|")
    g1 = [{(1, 0): 1, (0, 0): 3, (0, 2): 3}, {(0, 1): 1, (1, 0): 9}]
    g2 = [{(1, 0): 1, (0, 1): 9}, {(0, 1): 1, (0, 0): 3, (2, 0): 3}]
    for K in (8, 12, 16):
        start = time.perf_counter()
        pf.time_one_field(3, K, g1)
        field = time.perf_counter() - start
        start = time.perf_counter()
        action = pf.MinimalAction(3, K, [g1, g2], (0, 0))
        build = time.perf_counter() - start
        times = np.random.default_rng(0).integers(0, 3 ** (K - 1), (batch // 100, 2))
        action.chart(times)
        start = time.perf_counter()
        action.chart(times)
        chart = time.perf_counter() - start
        print(
            f"| g_1, g_2 on Z_3^2, k = 1 | {K} | {field:.2f} | {build:.1f} | "
            f"{len(times) / chart:.3g} |"
        )


def normal_forms() -> None:
    print("\n| normal form | p, K | M | split (s) |")
    print("|---|---|---|---|")
    field = [
        {(1, 0, 0, 0): 3, (0, 1, 0, 0): 6, (2, 0, 0, 0): 3},
        {(1, 0, 0, 0): 3, (0, 1, 0, 0): 3},
        {(0, 0, 1, 0): 6, (1, 1, 0, 0): 3},
        {(0, 0, 1, 0): 3, (0, 0, 0, 1): 12, (0, 2, 0, 0): 3},
    ]
    flow = pf.PAdicFlow(3, 6, field)
    start = time.perf_counter()
    form = flow.normal_form((0, 0, 0, 0))
    seconds = time.perf_counter() - start
    assert form.certificate.verify()
    print(
        f"| d = 4, complex pair and two real | 3, 6 | {form.degree} | {seconds:.1f} |"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="small batch, 3 repeats")
    args = parser.parse_args()
    if args.quick:
        args.batch, args.repeat = 10_000, 3
    print(f"devices: {jax.devices()}\n")
    flows(args.batch, args.repeat)
    rectify(args.batch, args.repeat)
    interpolation(args.batch)
    normal_forms()
