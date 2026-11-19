"""Values of OEIS linear recurrences: every `n` with `a(n) = c`, `c` in `{0, +-1, +-2}`,
by `value_times` (ABS), for the signatures of "Index to OEIS: Section Rec".

The index and the terms (`oeis.org/search?q=id:A..|id:A..&fmt=json`, ten per request,
with a pause between requests) are downloaded once into `--cache`. Each signature is
checked on every given term first: when it holds only from some term on, the initial
values are taken from there, and a sequence where fewer than `k + 3` terms check is
skipped. Times are reported as OEIS indices (offset included); `from` is the first
index where the recurrence holds, and zeros below it belong to the recurrence
continued backwards, not to the sequence. The `check` column compares the certified
set with the given terms from `from` on.

    python benchmarks/oeis.py --count 300 --max-order 6 --timeout 30 --jobs 4 > oeis.csv

Interesting rows (a zero past the initial values, and any `check` mismatch) are
summarised on stderr; an interrupted download resumes from the cache.
"""

import argparse
import csv
import html
import json
import pathlib
import random
import re
import sys
import time
import urllib.request

from galrax.padic.cli import run


INDEX = "https://oeis.org/wiki/Index_to_OEIS:_Section_Rec"
SEARCH = "https://oeis.org/search?fmt=json&q="
VALUES = (0, 1, -1, 2, -2)


def fetch(url, pause):
    time.sleep(pause)
    request = urllib.request.Request(url, headers={"User-Agent": "galrax-benchmark"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def signatures(cache, pause):
    """`{A-number: signature}` from the index, the first signature listed for each."""
    page = cache / "rec.html"
    if not page.exists():
        page.write_text(fetch(INDEX, pause), encoding="utf-8")
    out = {}
    for sig, rest in re.findall(r"<dd>\s*\(([-\d,\s]+)\)(.*?)</dd>", page.read_text("utf-8"), re.S):
        r = tuple(int(v) for v in sig.split(","))
        for a in re.findall(r"A\d{6}", html.unescape(rest)):
            out.setdefault(a, r)
    return out


def terms(numbers, cache, pause):
    """`{A-number: (offset, [a(offset), ...])}`, downloading what is not cached."""
    missing = [a for a in numbers if not (cache / f"{a}.json").exists()]
    for i in range(0, len(missing), 10):
        batch = missing[i:i + 10]
        found = json.loads(fetch(SEARCH + "|".join(f"id:{a}" for a in batch), pause)) or []
        for entry in found:
            a = f"A{entry['number']:06d}"
            keep = {k: entry[k] for k in ("offset", "data", "name")}
            (cache / f"{a}.json").write_text(json.dumps(keep), encoding="utf-8")
        print(f"downloaded {min(i + 10, len(missing))}/{len(missing)}", file=sys.stderr)
    out = {}
    for a in numbers:
        path = cache / f"{a}.json"
        if path.exists():
            entry = json.loads(path.read_text("utf-8"))
            out[a] = (int(entry["offset"].split(",")[0]), [int(v) for v in entry["data"].split(",")])
    return out


def start_of(r, a):
    """The first position from which `a` follows the recurrence `r` on every given
    term, or `None` when fewer than `k + 3` terms check."""
    k = len(r)
    bad = [i for i in range(k, len(a)) if a[i] != sum(x * a[i - 1 - j] for j, x in enumerate(r))]
    start = bad[-1] - k + 1 if bad else 0
    return start if len(a) - start >= 2 * k + 3 else None


def solve(case):
    import jax

    jax.config.update("jax_enable_x64", True)
    from galrax.padic import value_times

    a, r, u, c = case
    began = time.perf_counter()
    try:
        found = value_times(r, u, c)
    except ValueError as error:  # u_n = c for every n, or r_{k-1} = 0
        return {"A": a, "c": c, "error": str(error)}
    return {"A": a, "c": c, "certified": found.certified, "zeros": list(found.zeros),
            "by": found.by, "seconds": time.perf_counter() - began}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="oeis-cache")
    parser.add_argument("--count", type=int, default=300, help="sequences, sampled")
    parser.add_argument("--max-order", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    cache = pathlib.Path(args.cache)
    cache.mkdir(exist_ok=True)
    sigs = {a: r for a, r in signatures(cache, args.pause).items()
            if len(r) <= args.max_order and r[-1]}
    chosen = sorted(random.Random(args.seed).sample(sorted(sigs), min(args.count, len(sigs))))
    data = terms(chosen, cache, args.pause)

    cases, meta = [], {}
    for a in chosen:
        if a not in data:
            continue
        offset, seq = data[a]
        start = start_of(sigs[a], seq)
        if start is None:
            continue
        meta[a] = (offset + start, seq[start:])
        k = len(sigs[a])
        cases += [(a, sigs[a], tuple(seq[start:start + k]), c) for c in VALUES]
    print(f"{len(meta)} of {len(chosen)} sequences follow their signature", file=sys.stderr)

    results = run(cases, args.timeout, args.jobs, solve)
    writer = csv.writer(sys.stdout)
    writer.writerow(["A", "c", "certified", "zeros", "from", "by", "seconds", "check"])
    interesting = []
    for case, row in zip(cases, results):
        a, c = case[0], case[3]
        first, given = meta[a]
        if not row.get("certified"):
            why = row.get("error") or ("timeout" if row.get("timeout") else "not certified")
            writer.writerow([a, c, False, "", first, why, "", ""])
            continue
        zeros = [n + first for n in row["zeros"]]
        seen = [first + i for i, v in enumerate(given) if v == c]
        check = "ok" if seen == [z for z in zeros if first <= z < first + len(given)] else "MISMATCH"
        writer.writerow([a, c, True, " ".join(map(str, zeros)), first, row["by"],
                         f"{row['seconds']:.3f}", check])
        if check != "ok" or any(z >= first + len(case[1]) for z in zeros):  # past u_0..u_{k-1}
            interesting.append(f"{a} a(n) = {c}: {zeros} (from {first}) {check}")

    done = sum(1 for r in results if r.get("certified"))
    print(f"certified {done} of {len(results)}; interesting:", file=sys.stderr)
    print("\n".join(interesting), file=sys.stderr)


if __name__ == "__main__":
    main()
