"""Diffusion on the p-adic integers.

Functions on `Z_p` that are constant on balls of radius `p^{-n}` (functions on
`Z/p^n`), ultrametric diffusion operators such as the Vladimirov operator, which map
these functions to themselves exactly, and the Kozyrev wavelets that diagonalise them.
The Monna map and a fractal embedding draw `Z_p` on the line and in the plane.
Equations `du/dt = L u + N(t, u)` are `GroupODE`s on the cyclic group `Z/p^n`, solved
with `diffrax.group_algebra.solve_group_ode`.
"""

from ._ultrametric import (
    fractal_embedding as fractal_embedding,
    KozyrevTransform as KozyrevTransform,
    monna as monna,
    PAdicIntegers as PAdicIntegers,
    plot_kozyrev_wavelets as plot_kozyrev_wavelets,
    UltrametricOperator as UltrametricOperator,
)
