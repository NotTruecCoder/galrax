"""Diffusion on the p-adic integers.

Functions on `Z_p` that are constant on balls of radius `p^{-n}` (functions on
`Z/p^n`), ultrametric diffusion operators such as the Vladimirov operator, which map
these functions to themselves exactly, and the Kozyrev wavelets that diagonalise them.
Equations `du/dt = L u + N(t, u)` are `GroupODE`s on the cyclic group `Z/p^n`, solved
with `diffrax.group_algebra.solve_group_ode`.
"""

from ._ultrametric import (
    KozyrevTransform as KozyrevTransform,
    PAdicIntegers as PAdicIntegers,
    UltrametricOperator as UltrametricOperator,
)
