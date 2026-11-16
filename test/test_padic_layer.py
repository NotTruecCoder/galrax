import itertools
import pathlib
import sys

import numpy as np
import pytest


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "examples"))

import padic_layer as layer  # noqa: E402


def _planted(rng, p, K, d, degree):
    c = 2 if p == 2 else 1
    field = [
        {e: int(rng.integers(p ** (K - c))) * p**c for e in layer.monomials(d, degree)}
        for _ in range(d)
    ]
    return [{e: v % p**K for e, v in f.items() if v % p**K} for f in field]


@pytest.mark.parametrize(
    "p,K,d,degree", [(2, 6, 2, 1), (3, 5, 2, 2), (5, 4, 1, 3), (3, 4, 3, 1)]
)
def test_fit_recovers_the_planted_field(p, K, d, degree):
    rng = np.random.default_rng(p + d)
    planted = _planted(rng, p, K, d, degree)
    x = rng.integers(0, p**K, (60 * p**d, d)).astype(object)
    y = layer.apply(p, K, planted, 1, x)
    assert layer.fit(p, K, x, y, degree) == planted


def test_inverse_and_uniformity_on_balls():
    p, K, d = 3, 4, 2
    field = _planted(np.random.default_rng(0), p, K, d, 2)
    x = np.array(list(itertools.product(range(p**3), repeat=d)), dtype=object)
    y = layer.apply(p, K, field, 1, x)
    assert np.array_equal(layer.apply(p, K, field, -1, y), x)
    # x mod p^k determines phi_1(x) mod p^k, and every ball gets one point back.
    for k in (1, 2, 3):
        images = {tuple(v % p**k for v in row) for row in y}
        assert len(images) == p ** (k * d)


def test_stops_on_pairs_from_no_field():
    p, K = 3, 4
    x = np.arange(p**K, dtype=object).reshape(-1, 1)
    with pytest.raises(ValueError, match="no time-one map"):
        layer.fit(p, K, x, (x + 1) % p**K, 2)  # x + 1 is not the identity mod p
