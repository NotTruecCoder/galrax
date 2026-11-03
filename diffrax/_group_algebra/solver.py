from collections.abc import Callable
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._integrate import diffeqsolve
from .._local_interpolation import LocalLinearInterpolation
from .._saveat import SaveAt
from .._solution import RESULTS
from .._solver.base import AbstractAdaptiveSolver, AbstractSolver
from .._solver.runge_kutta import AbstractERK, ButcherTableau
from .._solver.tsit5 import Tsit5
from .._step_size_controller import (
    AbstractAdaptiveStepSizeController,
    AbstractStepSizeController,
    ClipStepSizeController,
    ConstantStepSize,
    PIDController,
)
from .._term import AbstractTerm, MultiTerm, ODETerm, WrapTerm
from .fourier import AbstractSpectralTransform, FourierTransform
from .problem import GroupODE
from .terms import AbstractExponentialTerm, to_complex, to_real


def _unwrap(term: AbstractTerm) -> AbstractTerm:
    while isinstance(term, WrapTerm):
        term = term.term
    return term


#
# Terms
#


class _CoordinateNonlinearity:
    def __init__(self, problem: GroupODE, transform: AbstractSpectralTransform):
        self.problem = problem
        self.transform = transform

    def __call__(self, t, y, args):
        problem = self.problem
        if problem.nonlinearity is None:
            return jtu.tree_map(jnp.zeros_like, y)
        p = self.transform.from_coordinates(y)
        if problem.is_real:
            p = p.real
        value = jnp.asarray(problem.nonlinearity(t, p, problem.args)).astype(p.dtype)
        return self.transform.to_coordinates(value)


def fourier_terms(
    problem: GroupODE, transform: AbstractSpectralTransform | None = None
) -> MultiTerm:
    """`MultiTerm(linear, ODETerm)`: the linear and nonlinear parts of `problem` in the
    coordinates of `transform` (by default the Fourier transform), as needed by
    `Lawson`.
    """
    transform = _check_transform(problem, transform)
    return MultiTerm(
        transform.linear_term(problem),
        ODETerm(_CoordinateNonlinearity(problem, transform)),
    )


def fourier_term(
    problem: GroupODE, transform: AbstractSpectralTransform | None = None
) -> ODETerm:
    """The whole vector field of `problem` in the coordinates of `transform`, as an
    `ODETerm` for any Diffrax solver.
    """
    linear, nonlinear = fourier_terms(problem, transform).terms

    def vector_field(t, y, args):
        return jtu.tree_map(jnp.add, linear.vf(t, y, args), nonlinear.vf(t, y, args))

    return ODETerm(vector_field)


def physical_term(problem: GroupODE) -> ODETerm:
    """The vector field of `problem` on the group itself, as an `ODETerm` for any
    Diffrax solver: an ODE in `R^n` (or `C^n`).
    """
    return ODETerm(lambda t, y, args: problem.vector_field(t, y))


#
# Lawson solver
#


class Lawson(AbstractAdaptiveSolver):
    """Lawson (integrating factor) version of an explicit Runge--Kutta method of
    Diffrax, for `dX/dt = L X + N(t, X)` with `L` given by an
    `AbstractExponentialTerm`, such as `FourierLinearTerm`:

    `U_i = e^{c_i h L} X_n + h sum_j a_ij e^{(c_i - c_j) h L} N(t_n + c_j h, U_j)`,

    `X_{n+1} = e^{h L} X_n + h sum_i b_i e^{(1 - c_i) h L} N(t_n + c_i h, U_i)`,

    with error estimate `h sum_i b_error_i e^{(1 - c_i) h L} N(t_n + c_i h, U_i)`.

    The linear part is integrated exactly, so stiffness coming from `L` does not
    restrict the step size, and the method keeps the order of the underlying
    Runge--Kutta method. Dense output is linear between steps.

    Use with the terms from `fourier_terms`.

    **Arguments:**

    - `solver`: an explicit Runge--Kutta method of Diffrax, e.g. `diffrax.Tsit5()`,
        `diffrax.Dopri5()`, `diffrax.Heun()`.
    """

    solver: AbstractERK = Tsit5()

    term_structure: ClassVar = MultiTerm[tuple[AbstractExponentialTerm, ODETerm]]
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    def __check_init__(self):
        tableau = getattr(self.solver, "tableau", None)
        if not isinstance(self.solver, AbstractERK) or not isinstance(
            tableau, ButcherTableau
        ):
            raise ValueError(
                "`Lawson` wraps an explicit Runge--Kutta solver with a single Butcher "
                "tableau, such as `diffrax.Tsit5()`."
            )

    def order(self, terms):
        return self.solver.order(terms)

    def error_order(self, terms):
        return self.solver.order(terms)

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(
        self,
        terms: MultiTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: None,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, None, RESULTS]:
        del solver_state, made_jump
        wrapped_linear, nonlinear = terms.terms
        linear = _unwrap(wrapped_linear)
        tableau = self.solver.tableau
        c = [0.0] + [float(ci) for ci in tableau.c]
        a = [[]] + [[float(aij) for aij in row] for row in tableau.a_lower]
        dt = t1 - t0
        h = wrapped_linear.contr(t0, t1)
        exponentials = {}

        def flow(tau, x):
            if tau not in exponentials:
                exponentials[tau] = linear.exponential(tau * h)
            return linear.apply(exponentials[tau], x)

        def combine(weighted):
            return [
                sum(jnp.asarray(w, x[k].dtype) * x[k] for w, x in weighted)
                for k in range(len(x0))
            ]

        x0 = to_complex(y0)
        slopes = []
        for i in range(len(c)):
            stage = [(1.0, flow(c[i], x0))]
            for j, aij in enumerate(a[i]):
                if aij != 0:
                    stage.append((h * aij, flow(c[i] - c[j], slopes[j])))
            value = nonlinear.vf(t0 + c[i] * dt, to_real(combine(stage)), args)
            slopes.append(to_complex(value))

        carried = [flow(1.0 - ci, slope) for ci, slope in zip(c, slopes)]
        x1 = combine(
            [(1.0, flow(1.0, x0))]
            + [(h * float(bi), k) for bi, k in zip(tableau.b_sol, carried)]
        )
        error = combine([(h * float(bi), k) for bi, k in zip(tableau.b_error, carried)])
        y1 = to_real(x1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, to_real(error), dense_info, None, RESULTS.successful

    def func(self, terms: MultiTerm, t0: RealScalarLike, y0: Y, args: Args) -> VF:
        return terms.vf(t0, y0, args)


#
# Solving
#


class GroupSolution:
    """The result of `solve_group_ode`.

    - `ts`: the save times.
    - `ys`: the solution on the group at those times, of shape `(len(ts),) + y0.shape`.
    - `stats`: the statistics reported by `diffrax.diffeqsolve`.
    - `solution`: the `diffrax.Solution`, in the coordinates that were solved in.
    """

    def __init__(self, ts, ys, stats, solution):
        self.ts = ts
        self.ys = ys
        self.stats = stats
        self.solution = solution


def solve_group_ode(
    problem: GroupODE,
    solver: AbstractSolver | None = None,
    *,
    ts=None,
    dt0: float | None = None,
    stepsize_controller: AbstractStepSizeController | None = None,
    transform: AbstractSpectralTransform | None = None,
    space: str = "fourier",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int | None = 4096,
    **kwargs,
) -> GroupSolution:
    """Solve `problem` with `diffrax.diffeqsolve`.

    **Arguments:**

    - `problem`: the `GroupODE`.
    - `solver`: any Diffrax solver. Defaults to `Lawson(diffrax.Tsit5())`.
    - `ts`: the save times. Defaults to `[t0, t1]`.
    - `dt0`: the initial step size; required by solvers without error estimates.
    - `stepsize_controller`: defaults to `diffrax.PIDController(rtol, atol)` for
        adaptive solvers and `diffrax.ConstantStepSize()` otherwise. With `Lawson`, an
        adaptive controller is wrapped in `diffrax.ClipStepSizeController` so that the
        solver steps exactly to `ts`: its dense output is only linear. With a constant
        step size, values at `ts` between steps are interpolated linearly.
    - `transform`: the `AbstractSpectralTransform` whose coordinates are solved for.
        Defaults to `FourierTransform`; `FastFourierTransform` is faster on large
        groups, and `diffrax.ultrametric.KozyrevTransform` diagonalises radial kernels
        on `Z/p^n`.
    - `space`: `"fourier"` to solve for the real coordinates of `transform` (required
        by `Lawson`), or `"physical"` to solve for the values on the group directly.
    - `rtol`, `atol`: tolerances of the default `PIDController`.
    - `max_steps`: passed to `diffeqsolve`.
    - `**kwargs`: any other arguments of `diffeqsolve`, such as `adjoint`.
    """
    if not isinstance(problem, GroupODE):
        raise TypeError("`problem` must be a `GroupODE`.")
    solver = Lawson() if solver is None else solver
    if not isinstance(solver, AbstractSolver):
        raise TypeError("`solver` must be a Diffrax solver.")
    if space not in ("fourier", "physical"):
        raise ValueError("`space` must be 'fourier' or 'physical'.")
    if isinstance(solver, Lawson) and space != "fourier":
        raise ValueError("`Lawson` solves in Fourier coordinates: use space='fourier'.")
    transform = _check_transform(problem, transform)
    ts = np.array([problem.t0, problem.t1]) if ts is None else np.asarray(ts, float)
    if stepsize_controller is None:
        if isinstance(solver, AbstractAdaptiveSolver):
            stepsize_controller = PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = ConstantStepSize()
    if isinstance(solver, Lawson) and isinstance(
        stepsize_controller, AbstractAdaptiveStepSizeController
    ):
        if not isinstance(stepsize_controller, ClipStepSizeController):
            stepsize_controller = ClipStepSizeController(
                stepsize_controller, step_ts=jnp.asarray(ts)
            )

    y0 = jnp.asarray(problem.y0)
    if isinstance(solver, Lawson):
        terms = fourier_terms(problem, transform)
    elif space == "fourier":
        terms = fourier_term(problem, transform)
    else:
        terms = physical_term(problem)
    if space == "fourier":
        y0 = transform.to_coordinates(y0)

    solution = diffeqsolve(
        terms,
        solver,
        problem.t0,
        problem.t1,
        dt0,
        y0,
        saveat=SaveAt(ts=jnp.asarray(ts)),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        **kwargs,
    )
    if space == "fourier":
        ys = transform.from_coordinates(
            tuple(jnp.moveaxis(block, 1, 0) for block in solution.ys)
        )
    else:
        ys = solution.ys
    if problem.is_real:
        ys = ys.real
    return GroupSolution(ts, ys, solution.stats, solution)


def exact_linear_solution(
    problem: GroupODE, ts, transform: AbstractSpectralTransform | None = None
):
    """The exact solution `p(t) = e^{(t - t0) L} y0` of a linear `problem` at the times
    `ts`, computed in the coordinates of `transform`: in Fourier coordinates
    `phat(t)(rho) = e^{(t - t0) A} phat(t0)(rho) e^{(t - t0) B}`. Returns an array of
    shape `(len(ts),) + y0.shape`.
    """
    if not problem.is_linear:
        raise ValueError("`exact_linear_solution` needs a linear problem.")
    transform = _check_transform(problem, transform)
    linear = transform.linear_term(problem)
    x0 = to_complex(transform.to_coordinates(problem.y0))

    def at(t):
        x = linear.apply(linear.exponential(t - problem.t0), x0)
        return transform.from_coordinates(to_real(x))

    ys = jax.vmap(at)(jnp.asarray(ts, dtype=float))
    return ys.real if problem.is_real else ys


def _check_transform(problem, transform):
    if transform is None:
        return FourierTransform(problem.group)
    if not isinstance(transform, AbstractSpectralTransform):
        raise TypeError("`transform` must be an `AbstractSpectralTransform`.")
    if transform.group is not problem.group:
        raise ValueError("`transform` must be built on the group of `problem`.")
    return transform
