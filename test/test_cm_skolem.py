import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks"))

from cm_skolem import (  # noqa: E402
    Gauss,
    class_series,
    evaluate,
    reduction_table,
    residue,
    setup,
    solve,
    strassmann,
    times,
)


def _target(n, S):
    xs = [times(E, n, P) for E, P in S]
    return Gauss.of(xs[0][0] + xs[1][0])


def test_series_from_real_times_predicts_the_gaussian_direction():
    # Coefficients come from t in Z only; at an inert prime the same series gives
    # h at n0 + (p + 1) i: h is holomorphic in one variable over Z_{p^2}.
    S, p, K = setup(), 7, 5
    c = _target((2, 1), S)
    n0 = (2, 1)
    series = class_series(p, K, n0, c, S)
    assert strassmann(series, p, K) == 1
    J = -(-K * (p - 1) // (p - 2))
    centre = -(J // 2)
    xs = [times(E, (n0[0], n0[1] + p + 1), P) for E, P in S]
    exact = residue(xs[0][0] + xs[1][0] - c, p**K)
    assert evaluate(series, (-centre, 1), p**K) == exact


def test_reduction_table_matches_exact_points():
    S, p = setup(), 11
    for E, P in S:
        table = reduction_table(E, P, p)
        for n in [(1, 0), (0, 1), (3, 5), (6, 6)]:
            R = times(E, n, P)
            assert table[n[0]][n[1]] == (
                None if R is None or residue(R[0], p) is None
                else (residue(R[0], p), residue(R[1], p))
            )


def test_certificates():
    S = setup()
    result = solve(_target((-2, 1), S), S, window=6)
    assert result["certified"] and sorted(result["zeros"]) == [(-2, 1), (2, -1)]
    result = solve(Gauss(7, -3), S, window=6)
    assert result["certified"] and result["zeros"] == []
