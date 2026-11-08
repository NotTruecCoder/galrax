"""Exact p-adic dynamics.

Flows of `dx/dt = f(x)` and `dx/dt = f(t, x)` with `x` in `Z_p^d`, `p`-adic time and a
polynomial `f`, computed exactly modulo `p^K` for all points and times at once: Lie
series in `(t, x)`, table-driven evaluation, Mahler interpolation of iterates,
flow-box (rectifying) coordinates by Newton's method, hitting and return times, orbit
averages, events, equilibria and linearisation, formal-group flows whose rectifying
coordinate is the formal logarithm, and Koopman operators with
ultrametric diffusion. Discrete maps `x -> T(x)` are analysed through their cycle tree
and minimal decomposition. Flows run on `int64` JAX arrays, with any number of digits.
"""

from ._padic_flow import (
    advection_diffusion as advection_diffusion,
    Ball as Ball,
    Component as Component,
    Cycle as Cycle,
    EllipticCurve as EllipticCurve,
    Equilibrium as Equilibrium,
    EventSet as EventSet,
    FormalGroupFlow as FormalGroupFlow,
    KoopmanOperator as KoopmanOperator,
    Limbs as Limbs,
    Linearization as Linearization,
    logarithmic_exponents as logarithmic_exponents,
    NormalForm as NormalForm,
    PAdicEvent as PAdicEvent,
    PAdicFlow as PAdicFlow,
    PAdicMap as PAdicMap,
    plot_advection_diffusion as plot_advection_diffusion,
    plot_phase_portrait as plot_phase_portrait,
    plot_trajectory as plot_trajectory,
    PowerSeries as PowerSeries,
    Residues as Residues,
    TimeDependentFlow as TimeDependentFlow,
)
