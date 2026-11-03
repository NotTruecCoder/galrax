import diffrax
import diffrax.group_algebra as ga
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate
import scipy.linalg
from diffrax.group_algebra import (
    convolution,
    exact_linear_solution,
    FastFourierTransform,
    FourierTransform,
    Group,
    GroupODE,
    Lawson,
    solve_group_ode,
    subgroup_closure,
)


def _from_permutations(generators):
    generators = [tuple(g) for g in generators]
    identity = tuple(range(len(generators[0])))
    elements = [identity]
    index = {identity: 0}
    i = 0
    while i < len(elements):
        for s in generators:
            p = tuple(elements[i][k] for k in s)
            if p not in index:
                index[p] = len(elements)
                elements.append(p)
        i += 1
    n = len(elements)
    table = np.empty((n, n), dtype=np.int64)
    for a, pa in enumerate(elements):
        for b, pb in enumerate(elements):
            table[a, b] = index[tuple(pa[k] for k in pb)]
    return table, index


def _quaternion_group():
    def mul(p, q):
        a1, b1, c1, d1 = p
        a2, b2, c2, d2 = q
        return (
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        )

    units = [tuple(s * (k == i) for k in range(4)) for i in range(4) for s in (1, -1)]
    index = {u: i for i, u in enumerate(units)}
    return np.array([[index[mul(p, q)] for q in units] for p in units])


def _cyclic(n):
    return (np.arange(n)[:, None] + np.arange(n)[None, :]) % n


def _relabel(table, seed):
    perm = np.random.default_rng(seed).permutation(len(table))
    inverse = np.argsort(perm)
    return perm[table[inverse][:, inverse]]


_TABLES = {
    "Z1": _cyclic(1),
    "Z6": _cyclic(6),
    "Z2xZ2": _from_permutations([(1, 0, 2, 3), (0, 1, 3, 2)])[0],
    "S3": _from_permutations([(1, 0, 2), (1, 2, 0)])[0],
    "D4": _from_permutations([(1, 2, 3, 0), (3, 2, 1, 0)])[0],
    "Q8": _quaternion_group(),
    "A4": _from_permutations([(1, 2, 0, 3), (1, 0, 3, 2)])[0],
    "S4": _from_permutations([(1, 0, 2, 3), (1, 2, 3, 0)])[0],
    "S3 relabelled": _relabel(_from_permutations([(1, 0, 2), (1, 2, 0)])[0], 1),
}
_GROUPS = {name: Group(table) for name, table in _TABLES.items()}
_NONTRIVIAL = [name for name in _GROUPS if name != "Z1"]


def _linear_matrix(problem):
    n = problem.group.n
    columns = [np.asarray(problem.linear(np.eye(n)[k])) for k in range(n)]
    return np.stack(columns, axis=1)


def _reference(problem, ts):
    linear = _linear_matrix(problem)

    def rhs(t, y):
        out = linear @ y
        if problem.nonlinearity is not None:
            out = out + np.asarray(
                problem.nonlinearity(t, jnp.asarray(y), problem.args)
            )
        return out

    sol = scipy.integrate.solve_ivp(
        rhs,
        (problem.t0, problem.t1),
        problem.y0,
        method="DOP853",
        t_eval=ts,
        rtol=1e-13,
        atol=1e-13,
    )
    assert sol.success
    return sol.y.T


def _nonlinear_problem(group, seed=0, t1=1.0):
    rng = np.random.default_rng(seed)
    n = group.n
    left = rng.standard_normal(n) / n
    right = rng.standard_normal(n) / n
    right[group.ed] -= 2.0
    y0 = rng.standard_normal(n)
    return GroupODE(
        group,
        y0,
        0.0,
        t1,
        left_kernel=left,
        right_kernel=right,
        nonlinearity=lambda t, p, a: a * jnp.cos(t) * jnp.tanh(p),
        args=0.7,
    )


#
# Group
#


@pytest.mark.parametrize(
    "name, degrees, indicators",
    [
        ("Z6", [1] * 6, [1, 1, 0, 0, 0, 0]),
        ("S3", [1, 1, 2], [1, 1, 1]),
        ("D4", [1, 1, 1, 1, 2], [1, 1, 1, 1, 1]),
        ("Q8", [1, 1, 1, 1, 2], [1, 1, 1, 1, -1]),
        ("A4", [1, 1, 1, 3], [1, 0, 0, 1]),
        ("S4", [1, 1, 2, 3, 3], [1, 1, 1, 1, 1]),
    ],
)
def test_known_groups(name, degrees, indicators):
    group = _GROUPS[name]
    assert group.degrees().tolist() == degrees
    assert sorted(group.frobenius_schur().tolist()) == sorted(indicators)
    assert len(group.conjugate()) == len(degrees)


@pytest.mark.parametrize("name", list(_GROUPS))
def test_group_structure(name):
    group = _GROUPS[name]
    n = group.n
    assert (group.kaili[group.ed] == np.arange(n)).all()
    assert (group.kaili[np.arange(n), group.inv] == group.ed).all()
    classes = group.conjugate()
    assert np.array_equal(classes[0], [group.ed])
    assert np.array_equal(np.sort(np.concatenate(classes)), np.arange(n))
    for c in classes:
        for g in c:
            conjugates = group.kaili[group.kaili[:, g], group.inv]
            assert set(conjugates.tolist()) == set(c.tolist())
    table = group.character_table()
    sizes = group.class_sizes
    assert np.allclose((table * sizes) @ table.conj().T, n * np.eye(len(sizes)))
    assert np.allclose(table[0], 1)
    assert (group.degrees() ** 2).sum() == n


@pytest.mark.parametrize("name", list(_GROUPS))
def test_irreducible_representations(name):
    group = _GROUPS[name]
    for chi, rep in zip(group.characters(), group.irreducible_representations()):
        assert group.check_representation(rep)
        assert np.allclose(np.trace(rep, axis1=1, axis2=2), chi)


def test_class_constants():
    group = _GROUPS["S3"]
    a = group.class_constants
    classes = group.conjugate()
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            products = group.kaili[np.ix_(ci, cj)].reshape(-1)
            for k, ck in enumerate(classes):
                assert a[i, j, k] == np.sum(products == ck[0])


@pytest.mark.parametrize(
    "table",
    [
        [[0, 1], [0, 1]],
        [[0, 1, 2], [1, 0, 2], [2, 2, 0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0, 1], [1, 2]],
        np.zeros((2, 3), dtype=int),
    ],
)
def test_group_rejects_invalid_tables(table):
    with pytest.raises(ValueError):
        Group(table)


def test_group_rejects_nonassociative_table():
    table = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 2],
            [2, 4, 0, 1, 3],
            [3, 2, 4, 0, 1],
            [4, 3, 1, 2, 0],
        ]
    )
    with pytest.raises(ValueError, match="associative"):
        Group(table)


#
# Fourier transform
#


@pytest.mark.parametrize("name", list(_GROUPS))
def test_fourier_transform(name):
    group = _GROUPS[name]
    n = group.n
    rng = np.random.default_rng(0)
    f = rng.standard_normal((2, 3, n)) + 1j * rng.standard_normal((2, 3, n))
    h = rng.standard_normal(n)
    fft = FastFourierTransform(group)
    for transform in (FourierTransform(group), fft):
        fhat = transform.forward(f)
        assert [a.shape for a in fhat] == [(2, 3, d, d) for d in transform.degrees]
        assert np.allclose(transform.inverse(fhat), f)
        assert np.allclose(transform.convolve(f, h), convolution(group, f, h))
        energy = sum(
            d * (np.abs(a) ** 2).sum(axis=(-2, -1))
            for d, a in zip(transform.degrees, fhat)
        )
        assert np.allclose(energy / n, (np.abs(f) ** 2).sum(axis=-1))
    for rep in fft.representations:
        assert group.check_representation(rep)
    dense = FourierTransform(group, fft.representations)
    for a, b in zip(fft.forward(f), dense.forward(f)):
        assert np.allclose(a, b)


@pytest.mark.parametrize("name", _NONTRIVIAL)
def test_fast_fourier_adapted_basis(name):
    group = _GROUPS[name]
    fft = FastFourierTransform(group)
    for level, subgroup in zip(fft.levels, fft.chain[:-1]):
        local = np.full(group.n, -1)
        local[level.elements] = np.arange(len(level.elements))
        for rep, blocks in zip(level.representations, level.blocks):
            restricted = rep[local[subgroup]]
            mask = np.zeros(rep.shape[1:], dtype=bool)
            for _, offset, e in blocks:
                mask[offset : offset + e, offset : offset + e] = True
            assert np.allclose(restricted[:, ~mask], 0)
    assert fft.operation_count() <= group.n**2


def test_fast_fourier_custom_chain():
    table, index = _from_permutations([(1, 0, 2, 3), (1, 2, 3, 0)])
    group = Group(table)
    s1, s2 = index[(1, 0, 2, 3)], index[(0, 2, 1, 3)]
    chain = [
        [group.ed],
        subgroup_closure(group, [s1]),
        subgroup_closure(group, [s1, s2]),
        np.arange(24),
    ]
    fft = FastFourierTransform(group, chain)
    assert [len(c) for c in fft.chain] == [1, 2, 6, 24]
    f = np.random.default_rng(1).standard_normal(24)
    for a, b in zip(
        fft.forward(f), FourierTransform(group, fft.representations).forward(f)
    ):
        assert np.allclose(a, b)
    assert np.allclose(fft.inverse(fft.forward(f)), f)


@pytest.mark.parametrize(
    "chain",
    [
        [[0, 1, 2], np.arange(24)],
        [[0], [0, 1, 2], np.arange(24)],
        [[0], np.arange(12)],
        [[0], np.arange(24), np.arange(24)],
    ],
)
def test_fast_fourier_rejects_invalid_chains(chain):
    with pytest.raises(ValueError):
        FastFourierTransform(_GROUPS["S4"], chain)


def test_fourier_rejects_bad_shapes():
    group = _GROUPS["S3"]
    transform = FourierTransform(group)
    with pytest.raises(ValueError):
        transform.forward(np.ones(5))
    with pytest.raises(ValueError):
        transform.inverse([np.ones((1, 1))])
    with pytest.raises(ValueError):
        FourierTransform(group, group.irreducible_representations()[:2])


#
# Problem
#


def test_problem_vector_field():
    group = _GROUPS["S4"]
    problem = _nonlinear_problem(group)
    p = np.random.default_rng(3).standard_normal((2, group.n))
    expected = (
        convolution(group, problem.left_kernel, p)
        + convolution(group, p, problem.right_kernel)
        + 0.7 * np.cos(0.4) * np.tanh(p)
    )
    assert np.allclose(problem.vector_field(0.4, p), expected)
    assert not problem.is_linear


def test_heat_equation_problem():
    group = _GROUPS["S4"]
    y0 = np.random.default_rng(0).random(group.n)
    problem = GroupODE.heat_equation(group, [1, 2], y0, 0.0, 1.0, rate=2.0)
    v = problem.vector_field(0.0, y0)
    generators = np.union1d([1, 2], group.inv[[1, 2]])
    generators = generators[generators != group.ed]
    for x in range(group.n):
        neighbours = [y0[group.kaili[x, group.inv[s]]] for s in generators]
        assert np.isclose(v[x], 2.0 * (np.mean(neighbours) - y0[x]))
    assert np.isclose(v.sum(), 0)
    assert problem.is_linear
    with pytest.raises(ValueError):
        GroupODE.heat_equation(group, [group.ed], y0, 0.0, 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(y0=np.ones(3)),
        dict(y0=np.full(24, np.nan)),
        dict(t1=0.0),
        dict(t1=np.inf),
        dict(right_kernel=np.ones(5)),
        dict(left_kernel=np.ones((24, 1))),
        dict(nonlinearity=lambda t, p, a: p[..., :2]),
    ],
)
def test_problem_rejects_invalid_input(kwargs):
    arguments = dict(group=_GROUPS["S4"], y0=np.ones(24), t0=0.0, t1=1.0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        GroupODE(**arguments)


def test_problem_rejects_invalid_types():
    with pytest.raises(TypeError):
        GroupODE(_TABLES["S3"], np.ones(6), 0.0, 1.0)
    with pytest.raises(TypeError):
        GroupODE(_GROUPS["S3"], np.ones(6), 0.0, 1.0, nonlinearity=3)


def test_real_problem_rejects_complex_nonlinearity():
    with pytest.raises(ValueError, match="real"):
        GroupODE(
            _GROUPS["S3"],
            np.ones(6),
            0.0,
            1.0,
            nonlinearity=lambda t, p, a: jax.lax.complex(p, p),
        )


#
# Solvers
#


def _close(a, b, atol):
    return np.abs(np.asarray(a) - np.asarray(b)).max() < atol


def _relative_close(a, b, rtol):
    b = np.asarray(b)
    return np.abs(np.asarray(a) - b).max() < rtol * np.abs(b).max()


@pytest.mark.parametrize("name", _NONTRIVIAL)
def test_exact_linear_solution(name):
    group = _GROUPS[name]
    rng = np.random.default_rng(1)
    n = group.n
    problem = GroupODE(
        group,
        rng.standard_normal((2, n)),
        0.0,
        1.5,
        left_kernel=rng.standard_normal(n),
        right_kernel=rng.standard_normal(n),
    )
    ts = np.array([0.0, 0.3, 1.5])
    ys = exact_linear_solution(problem, ts)
    linear = _linear_matrix(problem)
    for t, y in zip(ts, ys):
        expected = problem.y0 @ scipy.linalg.expm(t * linear).T
        assert _relative_close(y, expected, 1e-10)
    assert ys.dtype == jnp.float64


@pytest.mark.parametrize("base", [diffrax.Heun(), diffrax.Bosh3(), diffrax.Tsit5()])
def test_lawson_exact_on_linear_problems(base):
    group = _GROUPS["Q8"]
    rng = np.random.default_rng(2)
    problem = GroupODE(
        group, rng.standard_normal(8), 0.0, 2.0, right_kernel=rng.standard_normal(8)
    )
    sol = solve_group_ode(
        problem,
        Lawson(base),
        dt0=0.5,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    expected = scipy.linalg.expm(2.0 * _linear_matrix(problem)) @ problem.y0
    assert _close(sol.ys[-1], expected, 1e-10)


@pytest.mark.parametrize(
    "base, order, steps",
    [
        (diffrax.Heun(), 2, (32, 64)),
        (diffrax.Midpoint(), 2, (32, 64)),
        (diffrax.Bosh3(), 3, (16, 32)),
        (diffrax.Dopri5(), 5, (4, 8)),
    ],
)
@pytest.mark.parametrize("name", ["S3", "Q8", "S4"])
def test_lawson_convergence_order(base, order, steps, name):
    problem = _nonlinear_problem(_GROUPS[name])
    reference = _reference(problem, [problem.t1])[-1]
    errors = []
    for k in steps:
        sol = solve_group_ode(
            problem,
            Lawson(base),
            dt0=problem.t1 / k,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
        errors.append(np.abs(np.asarray(sol.ys[-1]) - reference).max())
    observed = np.log2(errors[0] / errors[1])
    assert order - 0.35 < observed < order + 1


@pytest.mark.parametrize("base", [diffrax.Tsit5(), diffrax.Dopri5()])
@pytest.mark.parametrize("fast", [False, True])
def test_adaptive_lawson(base, fast):
    group = _GROUPS["S4"]
    problem = _nonlinear_problem(group, t1=3.0)
    ts = np.linspace(0.0, 3.0, 7)
    reference = _reference(problem, ts)
    transform = FastFourierTransform(group) if fast else None
    loose = solve_group_ode(
        problem, Lawson(base), ts=ts, transform=transform, rtol=1e-5, atol=1e-8
    )
    tight = solve_group_ode(
        problem, Lawson(base), ts=ts, transform=transform, rtol=1e-10, atol=1e-12
    )
    assert _close(loose.ys, reference, 1e-3)
    assert _close(tight.ys, reference, 1e-8)
    assert tight.stats["num_steps"] > loose.stats["num_steps"]
    assert _close(tight.ys[0], problem.y0, 1e-12)


@pytest.mark.parametrize(
    "solver, space",
    [
        (diffrax.Tsit5(), "fourier"),
        (diffrax.Tsit5(), "physical"),
        (diffrax.Dopri8(), "physical"),
        (diffrax.Kvaerno5(), "physical"),
    ],
)
def test_any_diffrax_solver(solver, space):
    problem = _nonlinear_problem(_GROUPS["D4"])
    reference = _reference(problem, [problem.t1])[-1]
    sol = solve_group_ode(problem, solver, space=space, rtol=1e-10, atol=1e-12)
    assert _close(sol.ys[-1], reference, 1e-7)


def test_lawson_needs_fewer_steps_on_stiff_problems():
    group = _GROUPS["S4"]
    problem = _nonlinear_problem(group)
    stiff = GroupODE(
        group,
        problem.y0,
        0.0,
        1.0,
        right_kernel=200.0 * np.asarray(problem.right_kernel),
        nonlinearity=problem.nonlinearity,
        args=problem.args,
    )
    lawson = solve_group_ode(stiff, Lawson(), rtol=1e-6, atol=1e-9)
    explicit = solve_group_ode(stiff, diffrax.Tsit5(), rtol=1e-6, atol=1e-9)
    assert _close(lawson.ys, explicit.ys, 1e-5)
    assert lawson.stats["num_steps"] * 10 < explicit.stats["num_steps"]


def test_backward_in_time():
    group = _GROUPS["S3"]
    forward = _nonlinear_problem(group, t1=1.0)
    end = solve_group_ode(forward, rtol=1e-11, atol=1e-13).ys[-1]
    backward = GroupODE(
        group,
        np.asarray(end),
        1.0,
        0.0,
        left_kernel=forward.left_kernel,
        right_kernel=forward.right_kernel,
        nonlinearity=forward.nonlinearity,
        args=forward.args,
    )
    start = solve_group_ode(backward, rtol=1e-11, atol=1e-13).ys[-1]
    assert _close(start, forward.y0, 1e-8)


def test_batch():
    group = _GROUPS["D4"]
    rng = np.random.default_rng(4)
    y0 = rng.standard_normal((3, group.n))
    kernel = rng.standard_normal(group.n)

    def cubic(t, p, a):
        return -(p**3)

    ts = np.array([0.25, 0.5, 1.0])
    batched = GroupODE(group, y0, 0.0, 1.0, right_kernel=kernel, nonlinearity=cubic)
    batched_ys = solve_group_ode(batched, ts=ts, rtol=1e-10, atol=1e-12).ys
    assert batched_ys.shape == (3, 3, group.n)
    for k in range(3):
        single = GroupODE(
            group, y0[k], 0.0, 1.0, right_kernel=kernel, nonlinearity=cubic
        )
        single_ys = solve_group_ode(single, ts=ts, rtol=1e-10, atol=1e-12).ys
        assert _close(single_ys, batched_ys[:, k], 1e-8)


def test_complex_problem():
    group = _GROUPS["A4"]
    rng = np.random.default_rng(5)
    n = group.n
    problem = GroupODE(
        group,
        rng.standard_normal(n) + 1j * rng.standard_normal(n),
        0.0,
        1.0,
        left_kernel=1j * rng.standard_normal(n),
        nonlinearity=lambda t, p, a: -1j * (jnp.abs(p) ** 2).astype(p.dtype) * p,
    )
    sol = solve_group_ode(problem, rtol=1e-10, atol=1e-12)
    assert jnp.iscomplexobj(sol.ys)
    assert _close(sol.ys[-1], _reference(problem, [1.0])[-1], 1e-7)


def test_heat_equation_converges_to_uniform():
    group = _GROUPS["S4"]
    y0 = np.random.default_rng(6).random(group.n)
    problem = GroupODE.heat_equation(group, [1, 2], y0, 0.0, 200.0)
    ts = [0.0, 1.0, 200.0]
    exact = exact_linear_solution(problem, ts)
    assert _close(exact.sum(axis=-1), np.full(3, y0.sum()), 1e-10)
    assert _close(exact[-1], np.full(group.n, y0.mean()), 1e-10)
    assert (exact[1] > 0).all()
    solved = solve_group_ode(problem, ts=ts)
    assert _close(solved.ys, exact, 1e-6)
    assert solved.stats["num_steps"] < 20


def test_transforms_under_jit_and_grad():
    group = _GROUPS["S4"]
    f = jnp.asarray(np.random.default_rng(7).standard_normal(group.n))
    for transform in (FourierTransform(group), FastFourierTransform(group)):
        coordinates = jax.jit(transform.to_coordinates)(f)
        assert _close(jax.jit(transform.from_coordinates)(coordinates).real, f, 1e-12)

        def energy(f):
            return sum(jnp.sum(block**2) for block in transform.to_coordinates(f))

        assert _close(jax.grad(energy)(f), 2 * f, 1e-12)


def test_gradient_through_solve():
    group = _GROUPS["S3"]
    problem = _nonlinear_problem(group)
    transform = FourierTransform(group)
    terms = ga.fourier_terms(problem, transform)

    def loss(y0):
        sol = diffrax.diffeqsolve(
            terms,
            Lawson(),
            0.0,
            1.0,
            None,
            transform.to_coordinates(y0),
            stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        )
        final = transform.from_coordinates(tuple(block[-1] for block in sol.ys))
        return jnp.sum(final.real**2)

    y0 = jnp.asarray(problem.y0)
    gradient = jax.grad(loss)(y0)
    direction = jnp.asarray(np.random.default_rng(8).standard_normal(group.n))
    eps = 1e-6
    finite = (loss(y0 + eps * direction) - loss(y0 - eps * direction)) / (2 * eps)
    assert abs(float(gradient @ direction) - float(finite)) < 1e-6


def test_solver_errors():
    group = _GROUPS["S3"]
    nonlinear = _nonlinear_problem(group)
    with pytest.raises((TypeError, ValueError)):
        Lawson(diffrax.Kvaerno5())
    with pytest.raises((TypeError, ValueError)):
        Lawson(diffrax.Euler())
    with pytest.raises(ValueError, match="linear"):
        exact_linear_solution(nonlinear, [0.0, 1.0])
    with pytest.raises(ValueError, match="space"):
        solve_group_ode(nonlinear, space="spectral")
    with pytest.raises(ValueError, match="fourier"):
        solve_group_ode(nonlinear, Lawson(), space="physical")
    with pytest.raises(ValueError, match="group"):
        solve_group_ode(nonlinear, transform=FourierTransform(_GROUPS["Z6"]))
    with pytest.raises(TypeError):
        solve_group_ode(nonlinear, "tsit5")
    with pytest.raises(TypeError):
        solve_group_ode("problem")
