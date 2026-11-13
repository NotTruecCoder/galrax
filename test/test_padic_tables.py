import numpy as np
import pytest
from diffrax.padic_flow import PAdicFlow, PowerSeries


CASES = [
    (3, 19, [3, -3, 0, 6], {}),
    (5, 13, [5, 0, -5], {}),
    (3, 12, [{(0, 1): 3, (2, 0): 3}, {(0, 0): 3, (1, 1): -6}], {}),
    (
        3,
        12,
        [{(0, 1): 3, (2, 0): 3}, {(0, 0): 3, (1, 1): -6}],
        {"space_digits": (3, 1)},
    ),
    (3, 8, [{(0, 1): 3, (2, 0): 3}, {(1, 0): -3}], {"storage": "dense"}),
    (3, 30, [3, -3, 0, 6], {}),
    (3, 50, [3, -3, 0, 6], {"space_digits": 3}),
    (2, 12, [PowerSeries.exp(4) * 4], {}),
]


@pytest.mark.parametrize("p, K, field, kwargs", CASES)
def test_host_tables_match_the_series(p, K, field, kwargs):
    flow = PAdicFlow(p, K, field, **kwargs)
    d = flow.dimension
    rng = np.random.default_rng(0)
    shape = (64,) if d == 1 else (64, d)
    digits = min(K, 18)
    t = rng.integers(0, p**digits, 64).astype(object) * p ** (K - digits) + 1
    x = rng.integers(0, p**digits, shape).astype(object) * p ** (K - digits) + 2
    if p**K < 2**62:
        t, x = t.astype(np.int64) % p**K, x.astype(np.int64) % p**K
    table = np.asarray(flow.flow(t, x), dtype=object)
    series = np.asarray(flow.flow(t, x, method="taylor"), dtype=object)
    assert np.array_equal(table, series)
