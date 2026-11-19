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
through their cycle tree and minimal decomposition, and orbits of polynomial maps of
`Z_p^d` without attracting directions through the flow of a power of the map:
jump-ahead, return sets and zeros of linear recurrences, and zero sets of integer
orbits certified over all places by the adelic sieve ABS. Formal groups of any
dimension give logarithms as chart coordinates, and Chabauty functions are solved in
residue discs. Flows run on `int64` JAX arrays, with any number of digits.
"""

from .adelic import (
    etale_primes as etale_primes,
    henon_zeros as henon_zeros,
    map_zeros as map_zeros,
    recurrence_zeros as recurrence_zeros,
    ZeroSet as ZeroSet,
)
from .applications import (
    charpoly_to_r as charpoly_to_r,
    hypersurface_times as hypersurface_times,
    loop_exit as loop_exit,
    value_times as value_times,
)
from .arithmetic import Limbs as Limbs, Residues as Residues
from .equilibria import Equilibrium as Equilibrium, Linearization as Linearization
from .events import EventSet as EventSet, PAdicEvent as PAdicEvent
from .flow import PAdicFlow as PAdicFlow, TimeDependentFlow as TimeDependentFlow
from .formal import (
    chabauty_zeros as chabauty_zeros,
    disc_differentials as disc_differentials,
    EllipticCurve as EllipticCurve,
    FormalGroup as FormalGroup,
    FormalGroupFlow as FormalGroupFlow,
    height_bound as height_bound,
    linear_degree as linear_degree,
    logarithmic_exponents as logarithmic_exponents,
    tiny_integrals as tiny_integrals,
    weierstrass_omega as weierstrass_omega,
)
from .interpolation import (
    commutator as commutator,
    lie_bracket as lie_bracket,
    MinimalAction as MinimalAction,
    time_one_field as time_one_field,
)
from .integrable import (
    EliassonVey as EliassonVey,
    IntegrableSystem as IntegrableSystem,
)
from .koopman import (
    advection_diffusion as advection_diffusion,
    KoopmanOperator as KoopmanOperator,
)
from .maps import (
    Component as Component,
    Cycle as Cycle,
    matrix_power_index as matrix_power_index,
    PAdicMap as PAdicMap,
)
from .normal_form import (
    NormalForm as NormalForm,
    smith_exponent_bound as smith_exponent_bound,
)
from .plot import (
    plot_advection_diffusion as plot_advection_diffusion,
    plot_phase_portrait as plot_phase_portrait,
    plot_trajectory as plot_trajectory,
)
from .return_set import (
    companion as companion,
    linear as linear,
    Orbit as Orbit,
    recurrence_times as recurrence_times,
)
from .roots import Ball as Ball
from .series import PowerSeries as PowerSeries
