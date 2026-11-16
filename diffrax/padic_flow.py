"""Exact p-adic dynamics.

Flows of `dx/dt = f(x)` and `dx/dt = f(t, x)` with `x` in `Z_p^d`, `p`-adic time and a
polynomial or analytic `f`, computed exactly modulo `p^K` for all points and times at
once: Lie series in `(t, x)`, table-driven evaluation, Mahler interpolation of
iterates, flow-box (rectifying) coordinates by Newton's method, hitting and return
times, orbit averages, events, equilibria, linearisation and certified normal forms,
formal-group flows whose rectifying coordinate is the formal logarithm, and Koopman
operators with ultrametric diffusion. The field of a map congruent to the identity
and minimal actions of several such maps on balls are computed exactly, and so are
bounds on the small divisors of normal forms. Discrete maps `x -> T(x)` are analysed
through their cycle tree and minimal decomposition. Flows run on `int64` JAX arrays,
with any number of digits.
"""

from ._padic_flow import (
    advection_diffusion as advection_diffusion,
    Ball as Ball,
    chabauty_zeros as chabauty_zeros,
    commutator as commutator,
    companion as companion,
    Component as Component,
    Cycle as Cycle,
    disc_differentials as disc_differentials,
    EllipticCurve as EllipticCurve,
    Equilibrium as Equilibrium,
    EventSet as EventSet,
    FormalGroup as FormalGroup,
    FormalGroupFlow as FormalGroupFlow,
    height_bound as height_bound,
    KoopmanOperator as KoopmanOperator,
    lie_bracket as lie_bracket,
    Limbs as Limbs,
    linear as linear,
    linear_degree as linear_degree,
    Linearization as Linearization,
    logarithmic_exponents as logarithmic_exponents,
    matrix_power_index as matrix_power_index,
    MinimalAction as MinimalAction,
    NormalForm as NormalForm,
    Orbit as Orbit,
    PAdicEvent as PAdicEvent,
    PAdicFlow as PAdicFlow,
    PAdicMap as PAdicMap,
    plot_advection_diffusion as plot_advection_diffusion,
    plot_phase_portrait as plot_phase_portrait,
    plot_trajectory as plot_trajectory,
    PowerSeries as PowerSeries,
    recurrence_times as recurrence_times,
    Residues as Residues,
    smith_exponent_bound as smith_exponent_bound,
    time_one_field as time_one_field,
    TimeDependentFlow as TimeDependentFlow,
    tiny_integrals as tiny_integrals,
    weierstrass_omega as weierstrass_omega,
)
