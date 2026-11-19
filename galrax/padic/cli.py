"""`galrax-abs`: certified zero sets from the command line.

    galrax-abs rec 1,1 0,1 [--value C]
    galrax-abs map --vars x,y "y" "x*y + 1" --start 1,1 (--target 7 | --until "x - y")
    galrax-abs henon --c 1 --P 0,0,1 --start 0,1 --target 7
    galrax-abs tpls mfcs.tpls [--timeout 30 --jobs 4]

Exit codes: 0 certified (every instance, for `tpls`), 3 not, 2 bad input. A first
argument that starts with `-` goes after `--`: `galrax-abs rec -- -1,1 0,1`.
"""

import argparse
import ast
import json
import multiprocessing as mp
import sys
import time
from functools import partial


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


def run(cases, timeout, processes, solver):
    """`solver` on every case in worker processes, results in the order of `cases`; a
    case over `timeout` seconds is killed and recorded as `{"order": case[0],
    "timeout": True}`, an exception as `{"order": case[0], "error": ...}`.
    """
    pending, results = list(enumerate(cases)), {}

    context = mp.get_context("spawn")  # fork under a running JAX can deadlock

    def spawn():
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, solver), daemon=True)
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
                results[task[0]] = pipe.recv()
                w["task"] = None
            elif task is not None and time.perf_counter() - w["since"] > timeout:
                w["process"].kill()
                results[task[0]] = {"order": task[1][0], "timeout": True}
                w.update(spawn())
        time.sleep(0.01)
    for w in workers:
        w["pipe"].send(None)
    return [results[i] for i in sorted(results)]


# Expressions: a whitelist of the AST, never `eval` (the input is untrusted).

def parse_poly(text, names):
    """`{exponents: coefficient}` of an integer polynomial in `names`, written with
    integers, `+ - *` and `**` to a constant power; anything else is a `ValueError`."""
    d = len(names)

    def add(a, b, sign=1):
        out = dict(a)
        for e, c in b.items():
            out[e] = out.get(e, 0) + sign * c
        return {e: c for e, c in out.items() if c}

    def mul(a, b):
        out = {}
        for ea, ca in a.items():
            for eb, cb in b.items():
                e = tuple(x + y for x, y in zip(ea, eb))
                out[e] = out.get(e, 0) + ca * cb
        return {e: c for e, c in out.items() if c}

    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return {(0,) * d: node.value} if node.value else {}
        if isinstance(node, ast.Name) and node.id in names:
            return {tuple(int(n == node.id) for n in names): 1}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a = walk(node.operand)
            return {e: -c for e, c in a.items()} if isinstance(node.op, ast.USub) else a
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            sign = 1 if isinstance(node.op, ast.Add) else -1
            return add(walk(node.left), walk(node.right), sign)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return mul(walk(node.left), walk(node.right))
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant) and type(node.right.value) is int
                and 0 <= node.right.value <= 1000):
            base, out = walk(node.left), {(0,) * d: 1}
            for _ in range(node.right.value):
                out = mul(out, base)
            return out
        what = (node.id if isinstance(node, ast.Name) else
                type(getattr(node, "op", node)).__name__)
        raise ValueError(f"not allowed in {text!r}: {what}")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot parse {text!r}") from error
    return walk(tree.body)


def _ints(text):
    try:
        return tuple(int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integers separated by commas: {text!r}")


def _record(found, seconds, certificate):
    out = {"certified": found.certified, "zeros": list(found.zeros),
           "period": found.period, "by": found.by, "primes": list(found.primes),
           "seconds": round(seconds, 4)}
    if certificate:
        keys = ("p", "k", "type", "T", "cover", "sides")
        out["records"] = [{key: r.get(key) for key in keys} for r in found.records]
    return out


def _tpls_solve(case, certificate=False):
    import jax

    jax.config.update("jax_enable_x64", True)  # a fresh worker process
    from galrax.padic import recurrence_zeros

    _, r, u = case
    began = time.perf_counter()
    found = recurrence_zeros(r, u)
    return _record(found, time.perf_counter() - began, certificate)


def _show(found, seconds, args, where):
    if args.json:
        print(json.dumps(_record(found, seconds, args.certificate)))
    else:
        if found.certified:
            zeros = "{" + ", ".join(map(str, found.zeros)) + "}"
            if found.period:
                zeros += f" + {found.period}Z"
            print(f"certified: {where}: {zeros}, by {found.by}")
        else:
            print("not certified")
        print(f"primes: {list(found.primes)}; {seconds:.2f} s")
        if args.certificate:
            for r in found.records:
                k = f", k = {r['k']}" if "k" in r else ""
                print(f"  p = {r['p']}{k}, {r['type']}, T = {r['T']}, {r['sides']}, "
                      f"cover {r['cover']}")
    return 0 if found.certified else 3


def main(argv=None):
    import jax

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(prog="galrax-abs", description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="one JSON object")
    common.add_argument("--certificate", action="store_true",
                        help="the records: prime, depth, type, tail, cover")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("rec", parents=[common], help="n in Z with u_n = C for "
                         "u_n = r_0 u_{n-1} + ... + r_{k-1} u_{n-k}")
    rec.add_argument("r", type=_ints)
    rec.add_argument("u", type=_ints)
    rec.add_argument("--value", type=int, default=0, metavar="C")

    m = sub.add_parser("map", parents=[common], help="n >= 0 with x_n = target or g(x_n) = 0")
    m.add_argument("F", nargs="+", help="one polynomial per coordinate")
    m.add_argument("--vars", default="x")
    m.add_argument("--start", type=_ints, required=True)
    goal = m.add_mutually_exclusive_group(required=True)
    goal.add_argument("--target", type=int, help="x_n = TARGET, x the first coordinate")
    goal.add_argument("--until", metavar="G", help="g(x_n) = 0")

    h = sub.add_parser("henon", parents=[common], help="n in Z with x_n = target, Henon maps")
    h.add_argument("--c", type=int, required=True, choices=(1, -1))
    h.add_argument("--P", type=_ints, required=True)
    h.add_argument("--start", type=_ints, required=True)
    h.add_argument("--target", type=int, required=True)

    t = sub.add_parser("tpls", help="lines (k, r, u) as in mfcs.tpls of SKOLEM; JSONL out")
    t.add_argument("path")
    t.add_argument("--timeout", type=float, default=30.0)
    t.add_argument("--jobs", type=int, default=1)
    t.add_argument("--certificate", action="store_true")

    args = parser.parse_args(argv)
    if args.command in ("rec", "tpls"):
        try:
            import flint  # noqa: F401
        except ImportError:
            parser.exit(1, "galrax-abs: the real place of recurrences needs python-flint: "
                           "pip install python-flint\n")
    from galrax.padic import henon_zeros, hypersurface_times, map_zeros, value_times

    began = time.perf_counter()
    if args.command == "tpls":
        try:
            with open(args.path) as f:
                cases = [ast.literal_eval(line) for line in f if line.strip()]
        except (OSError, ValueError, SyntaxError) as error:
            parser.error(f"cannot read {args.path}: {error}")
        if not all(isinstance(c, tuple) and len(c) == 3 for c in cases):
            parser.error("every line must be (k, r, u)")
        results = run(cases, args.timeout, max(1, args.jobs),
                      partial(_tpls_solve, certificate=args.certificate))
        for r in results:
            if "order" in r:  # timed out or failed
                r = {"certified": False} | {k: v for k, v in r.items() if k != "order"}
            print(json.dumps(r))
        return 0 if all(r.get("certified") for r in results) else 3
    try:
        if args.command == "rec":
            found = value_times(args.r, args.u, args.value)
            where = f"n in Z with u_n = {args.value}"
        elif args.command == "henon":
            found = henon_zeros(args.c, args.P, args.start, args.target)
            where = f"n in Z with x_n = {args.target}"
        else:
            names = [v.strip() for v in args.vars.split(",")]
            if len(names) != len(args.F) or len(names) != len(args.start):
                parser.error("need one polynomial and one start value per variable")
            F = [parse_poly(f, names) for f in args.F]
            if args.until is None:
                found = map_zeros(F, args.start, args.target)
                where = f"n >= 0 with {names[0]}_n = {args.target}"
            else:
                found = hypersurface_times(F, args.start, parse_poly(args.until, names))
                where = f"n >= 0 with {args.until} = 0"
    except ValueError as error:
        parser.error(str(error))
    return _show(found, time.perf_counter() - began, args, where)


if __name__ == "__main__":
    sys.exit(main())
