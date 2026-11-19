import pytest

from galrax.padic import IntegrableSystem


def _mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            e = tuple(i + j for i, j in zip(ea, eb))
            out[e] = out.get(e, 0) + ca * cb
    return {e: c for e, c in out.items() if c}


def _compose(f, maps):
    """`f(maps)` in exact integers."""
    d = len(next(iter(maps[0])))
    out = {}
    for e, c in f.items():
        term = {(0,) * d: c}
        for j, k in enumerate(e):
            for _ in range(k):
                term = _mul(term, maps[j])
        for g, v in term.items():
            out[g] = out.get(g, 0) + v
    return {e: c for e, c in out.items() if c}


QUADRATIC = {  # variables (x1, x2, y1, y2)
    "split": [{(1, 0, 1, 0): 1, (0, 1, 0, 1): 1}, {(1, 0, 1, 0): 1, (0, 1, 0, 1): -2}],
    "elliptic": [{(2, 0, 0, 0): 1, (0, 0, 2, 0): 1}, {(0, 2, 0, 0): 1, (0, 0, 0, 2): 1}],
    "focus-focus": [{(1, 0, 1, 0): 1, (0, 1, 0, 1): 1}, {(1, 0, 0, 1): 1, (0, 1, 1, 0): -1}],
}
G0 = [{(1, 0): 1, (2, 0): 1, (1, 1): -1, (3, 0): 1}, {(0, 1): 1, (0, 2): 2}]


def _system(kind):
    """`F = G0(q o S)` with `S` the composition of two polynomial symplectic shears,
    `y += grad A(x)` and `x += grad B(y)`: an exactly integrable polynomial system
    whose normal form is known."""
    shear_y = [{(1, 0, 0, 0): 1}, {(0, 1, 0, 0): 1},
               {(0, 0, 1, 0): 1, (2, 0, 0, 0): 3, (0, 2, 0, 0): 2},  # A = x1^3 + 2 x1 x2^2
               {(0, 0, 0, 1): 1, (1, 1, 0, 0): 4}]
    shear_x = [{(1, 0, 0, 0): 1, (0, 0, 1, 1): 2}, {(0, 1, 0, 0): 1, (0, 0, 2, 0): 1},  # B = y1^2 y2
               {(0, 0, 1, 0): 1}, {(0, 0, 0, 1): 1}]
    S = [_compose(s, shear_x) for s in shear_y]
    q = [_compose(f, S) for f in QUADRATIC[kind]]
    return [_compose(g, q) for g in G0]


@pytest.mark.parametrize("kind", list(QUADRATIC))
@pytest.mark.parametrize("p,K", [(2, 6), (3, 5), (5, 4)])
def test_eliasson_vey_recovers_the_normal_form(kind, p, K):
    form = IntegrableSystem(p, _system(kind)).normal_form(precision=K)
    assert form.certificate.verify()
    m, s = p**K, p ** (2 * form.radius)  # G~(u) = p^{-2 rho} G0(p^{2 rho} u)
    expected = [{g: c * s ** (sum(g) - 1) % m for g, c in f.items() if c * s ** (sum(g) - 1) % m}
                for f in G0]
    assert form.G == expected


def test_eliasson_vey_certificate_rejects_a_broken_transformation():
    form = IntegrableSystem(3, _system("split")).normal_form(precision=4)
    form.transformation[0][(1, 1, 0, 0)] = (form.transformation[0].get((1, 1, 0, 0), 0) + 3) % 81
    assert not form.certificate.verify()


def test_integrable_system_checks_its_input():
    with pytest.raises(ValueError, match="not integrable"):
        IntegrableSystem(3, [{(1, 0, 1, 0): 1}, {(2, 0, 0, 0): 1, (0, 0, 0, 2): 1}])
    with pytest.raises(ValueError, match="critical point"):
        IntegrableSystem(3, [{(1, 0, 0, 0): 1}, {(0, 1, 0, 1): 1}])
