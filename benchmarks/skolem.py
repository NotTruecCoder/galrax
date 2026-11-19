"""Zeros of linear recurrences on the benchmark of the SKOLEM tool (Bilu, Luca,
Nieuwveld, Ouaknine, Purser, Worrell, "Skolem meets Schanuel", MFCS 2022).

The dataset is `datasets/mfcs.tpls` of github.com/davidjpurser/skolem-solver
(AGPL, not copied here): 82,360 random integer recurrences
`u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}`, `k = 2, ..., 10`, coefficients and
initial values in `[-20, 20]`, one `(k, (r_0, ...), (u_0, ...))` per line.

For each sampled recurrence, `recurrence_times` is run at primes `p < 200` not
dividing `r_{k-1}`, so the companion matrix is invertible modulo `p` and the orbit
has no tail, taken in order of `p` times the period of `u_n` modulo `p` (at most
5000): the cost of a prime grows with that product, and it varies by orders of
magnitude between primes. The `classes` of each set bound, by Strassmann's theorem
on each class `n = c + N q` with `q` in `Z_p`, the number of zeros at every integer
`n` of the class, negative ones included. A class none of whose integers lies in
the set of another prime holds no integer zero and is dropped (the
Skolem--Mahler--Lech sieve). The exact zeros with `|n| <= window` are counted in
exact arithmetic: when their number reaches the sum of the remaining bounds, the
zero set over `Z` is certified complete. Otherwise
(after eight primes, for a degenerate recurrence, or past the time limit) the
instance counts as not certified.

    python benchmarks/skolem.py mfcs.tpls --per-order 100 --timeout 60
"""

import argparse
import ast
import math
import multiprocessing as mp
import random
import time
from functools import partial

from galrax.padic.adelic import exact_zeros


PRIMES = tuple(p for p in range(3, 200) if all(p % q for q in range(2, p)))


def cycle(recurrence, initial, p, cap):
    """The period of `u_n` modulo `p` (no tail when `p` does not divide
    `r_{k-1}`), or `None` past `cap`.
    """
    start = state = tuple(v % p for v in initial)
    for n in range(1, cap + 1):
        head = sum(r * v for r, v in zip(recurrence, reversed(state))) % p
        state = state[1:] + (head,)
        if state == start:
            return n
    return None


def cheap_primes(recurrence, initial, count, cap=5000):
    """The `count` primes not dividing `r_{k-1}` with the shortest period of the
    sequence modulo `p`: the cost of a prime grows with `p` times that period.
    """
    periods = []
    for p in PRIMES:
        if recurrence[-1] % p == 0:
            continue
        L = cycle(recurrence, initial, p, cap)
        if L is not None:
            periods.append((L * p, p))
    return [p for _, p in sorted(periods)[:count]]


def sieve(sets):
    """The least, over the primes, of the sum of Strassmann's bounds over the
    classes `n = c mod N` of that prime that meet every other prime's set; `None`
    if a remaining class has no bound.
    """
    best = None
    for main in sets:
        total = 0
        for c, N, bound in main.classes:
            if any(
                all(
                    (c - b.center[0]) % math.gcd(N, other.period * other.p**b.level)
                    for b in other.balls
                )
                for other in sets
                if other is not main
            ):
                continue  # no integer of the class lies in the other set
            if bound is None:
                total = None
                break
            total += bound
        if total is not None and (best is None or total < best):
            best = total
    return best


def solve(case, window=200, primes=8, cap=5000):
    import jax

    jax.config.update("jax_enable_x64", True)
    from galrax.padic import recurrence_times

    k, recurrence, initial = case
    start = time.perf_counter()
    zeros = exact_zeros(recurrence, initial, window)
    result = {"order": k, "zeros": zeros, "certified": False, "bound": None}
    if not any(initial):
        result["certified"], result["zero_sequence"] = True, True
        return result | {"seconds": time.perf_counter() - start}
    sets = []
    for p in cheap_primes(recurrence, initial, primes, cap):
        precision = max(4, int(30 / p.bit_length()))
        events = recurrence_times(p, precision, list(reversed(recurrence)), initial)
        # Soundness: every exact zero n >= 0 must lie in the computed classes.
        assert all(n in events for n in zeros if n >= 0), (case, p)
        sets.append(events)
        bound = sieve(sets)
        result |= {"primes": [e.p for e in sets], "bound": bound}
        if bound is not None and bound <= len(zeros):
            assert bound == len(zeros), case  # the bound counts every integer zero
            result["certified"] = True
            break
        if len(sets) == primes:
            break
    return result | {"seconds": time.perf_counter() - start}


def _worker(connection, solver):
    import galrax.padic  # noqa: F401  (imported before the clock starts)

    connection.send("ready")
    while True:
        case = connection.recv()
        if case is None:
            return
        try:
            connection.send(solver(case))
        except Exception as error:  # reported, not fatal for the run
            connection.send({"order": case[0], "error": repr(error)})


def run(cases, timeout, processes, solver=solve):
    """`solver` on every case in worker processes; a case over `timeout` seconds is
    killed and recorded as timed out.
    """
    pending, results = list(enumerate(cases)), []

    def spawn():
        parent, child = mp.Pipe()
        process = mp.Process(target=_worker, args=(child, solver), daemon=True)
        process.start()
        return {"process": process, "pipe": parent, "task": None, "ready": False}

    workers = [spawn() for _ in range(processes)]
    while pending or any(w["task"] is not None for w in workers):
        for w in workers:
            pipe, task = w["pipe"], w["task"]
            if not w["ready"]:
                if pipe.poll():
                    w["ready"] = pipe.recv() == "ready"
            elif task is None and pending:
                w["task"], w["since"] = pending.pop(), time.perf_counter()
                pipe.send(w["task"][1])
            elif task is not None and pipe.poll():
                results.append(pipe.recv())
                w["task"] = None
            elif task is not None and time.perf_counter() - w["since"] > timeout:
                w["process"].kill()
                results.append({"order": task[1][0], "timeout": True})
                w.update(spawn())
        time.sleep(0.01)
    for w in workers:
        w["pipe"].send(None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--per-order", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--processes", type=int, default=mp.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--primes", type=int, default=8)
    parser.add_argument("--cap", type=int, default=5000)
    args = parser.parse_args()

    with open(args.dataset) as f:
        cases = [ast.literal_eval(line) for line in f if line.strip()]
    rng = random.Random(args.seed)
    sample = []
    for k in range(2, 11):
        of_order = [c for c in cases if c[0] == k]
        sample += rng.sample(of_order, min(args.per_order, len(of_order)))
    solver = partial(solve, primes=args.primes, cap=args.cap)
    results = run(sample, args.timeout, args.processes, solver)

    print("| order | instances | certified | timed out | mean time of certified (s) |")
    print("|---|---|---|---|---|")
    for k in range(2, 11):
        rows = [r for r in results if r["order"] == k]
        done = [r for r in rows if r.get("certified")]
        late = sum(1 for r in rows if r.get("timeout"))
        mean = sum(r["seconds"] for r in done) / len(done) if done else float("nan")
        print(
            f"| {k} | {len(rows)} | {len(done)} ({100 * len(done) / len(rows):.0f}%) "
            f"| {late} | {mean:.2f} |"
        )
    errors = [r for r in results if "error" in r]
    zeros = sum(len(r["zeros"]) for r in results if r.get("certified"))
    print(f"\nzeros in certified instances: {zeros}; errors: {len(errors)}")
    for r in errors[:5]:
        print(r)


if __name__ == "__main__":
    main()
