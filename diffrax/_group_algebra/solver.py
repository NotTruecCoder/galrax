from collections.abc import Callable
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
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
from .fourier import AbstractFourierTransform, FourierTransform
from .problem import GroupODE


def _unwrap(term: AbstractTerm) -> AbstractTerm:
    while isinstance(term, WrapTerm):
        term = term.term
    return term


def _to_complex(y) -> list:
    return [jax.lax.complex(block[0], block[1]) for block in y]


def _to_real(x) -> tuple:
    return tuple(jnp.stack([block.real, block.imag]) for block in x)


#
# Terms
#


class FourierLinearTerm(AbstractTerm):
    """The linear part `L p = lambda * p + p * mu` of a `GroupODE`, in the real
    coordinates of `AbstractFourierTransform.to_coordinates`. In the block of `rho`,
    with `X` the coordinates, `A = lambdahat(rho)` and `B = muhat(rho)`,

    `L X = A X + X B`,  `e^{tau L} X = e^{tau A} X e^{tau B}`.

    **Arguments:**

    - `left`: the blocks `A`, one per representation, or `None` where `lambda` is
        absent. Each block is stored as an array of shape `(2, d, d)` holding its real
        and imaginary parts, so that the term contains no complex arrays.
    - `right`: the blocks `B`, likewise.
    """

    left: tuple
    right: tuple

    @classmethod
    def from_problem(
        cls, problem: GroupODE, transform: AbstractFourierTransform
    ) -> "FourierLinearTerm":
        def blocks(kernel):
            if kernel is None:
                return (None,) * len(transform.degrees)
            return _to_real(transform.forward(kernel))

        return cls(blocks(problem.left_kernel), blocks(problem.right_kernel))

    def _blocks(self):
        def complex_or_none(blocks):
            return [None if b is None else jax.lax.complex(b[0], b[1]) for b in blocks]

        return complex_or_none(self.left), complex_or_none(self.right)

    def vf(self, t: RealScalarLike, y: Y, args: Args) -> VF:
        out = []
        left, right = self._blocks()
        for a, b, x in zip(left, right, _to_complex(y)):
            value = jnp.zeros_like(x)
            if a is not None:
                value = value + a @ x
            if b is not None:
                value = value + x @ b
            out.append(value)
        return _to_real(out)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> RealScalarLike:
        return t1 - t0

    def prod(self, vf: VF, control: RealScalarLike) -> Y:
        return jtu.tree_map(lambda v: v * control, vf)

    def exponential(self, tau) -> list:
        """The pairs `(e^{tau A}, e^{tau B})`, one per representation (`None` where the
        kernel is absent).
        """
        left, right = self._blocks()
        return [
            (
                None if a is None else jsl.expm(jnp.asarray(tau, a.dtype) * a),
                None if b is None else jsl.expm(jnp.asarray(tau, b.dtype) * b),
            )
            for a, b in zip(left, right)
        ]

    @staticmethod
    def apply(exponential, x) -> list:
        """`e^{tau L} x` for complex blocks `x`, given `exponential(tau)`."""
        out = []
        for (left, right), block in zip(exponential, x):
            if left is not None:
                block = left @ block
            if right is not None:
                block = block @ right
            out.append(block)
        return out


class _FourierNonlinearity:
    def __init__(self, problem: GroupODE, transform: AbstractFourierTransform):
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
    problem: GroupODE, transform: AbstractFourierTransform | None = None
) -> MultiTerm:
    """`MultiTerm(FourierLinearTerm, ODETerm)`: the linear and nonlinear parts of
    `problem` in Fourier coordinates, as needed by `Lawson`.
    """
    transform = _check_transform(problem, transform)
    return MultiTerm(
        FourierLinearTerm.from_problem(problem, transform),
        ODETerm(_FourierNonlinearity(problem, transform)),
    )


def fourier_term(
    problem: GroupODE, transform: AbstractFourierTransform | None = None
) -> ODETerm:
    """The whole vector field of `problem` in Fourier coordinates, as an `ODETerm`
    for any Diffrax solver.
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
    Diffrax, for `dX/dt = L X + N(t, X)` with `L` given by a `FourierLinearTerm`:

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

    term_structure: ClassVar = MultiTerm[tuple[FourierLinearTerm, ODETerm]]
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

        x0 = _to_complex(y0)
        slopes = []
        for i in range(len(c)):
            stage = [(1.0, flow(c[i], x0))]
            for j, aij in enumerate(a[i]):
                if aij != 0:
                    stage.append((h * aij, flow(c[i] - c[j], slopes[j])))
            value = nonlinear.vf(t0 + c[i] * dt, _to_real(combine(stage)), args)
            slopes.append(_to_complex(value))

        carried = [flow(1.0 - ci, slope) for ci, slope in zip(c, slopes)]
        x1 = combine(
            [(1.0, flow(1.0, x0))]
            + [(h * float(bi), k) for bi, k in zip(tableau.b_sol, carried)]
        )
        error = combine([(h * float(bi), k) for bi, k in zip(tableau.b_error, carried)])
        y1 = _to_real(x1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, _to_real(error), dense_info, None, RESULTS.successful

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
    transform: AbstractFourierTransform | None = None,
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
    - `transform`: the Fourier transform. Defaults to `FourierTransform`.
    - `space`: `"fourier"` to solve for the real Fourier coordinates of
        `AbstractFourierTransform.to_coordinates` (required by `Lawson`), or
        `"physical"` to solve for the values on the group directly.
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
    problem: GroupODE, ts, transform: AbstractFourierTransform | None = None
):
    """The exact solution of a linear `problem` at the times `ts`:
    `phat(t)(rho) = e^{(t - t0) A} phat(t0)(rho) e^{(t - t0) B}`. Returns an array of
    shape `(len(ts),) + y0.shape`.
    """
    if not problem.is_linear:
        raise ValueError("`exact_linear_solution` needs a linear problem.")
    transform = _check_transform(problem, transform)
    linear = FourierLinearTerm.from_problem(problem, transform)
    x0 = transform.forward(problem.y0)

    def at(t):
        return transform.inverse(linear.apply(linear.exponential(t - problem.t0), x0))

    ys = jax.vmap(at)(jnp.asarray(ts, dtype=float))
    return ys.real if problem.is_real else ys


def _check_transform(problem, transform):
    if transform is None:
        return FourierTransform(problem.group)
    if not isinstance(transform, AbstractFourierTransform):
        raise TypeError("`transform` must be an `AbstractFourierTransform`.")
    if transform.group is not problem.group:
        raise ValueError("`transform` must be built on the group of `problem`.")
    return transform
