"""Diffusion on the p-adic integers.

Functions on `Z_p` that are constant on balls of radius `p^{-n}` (functions on
`Z/p^n`), ultrametric diffusion operators such as the Vladimirov operator, which map
these functions to themselves exactly, and the Kozyrev wavelets that diagonalise them.
The Monna map and a fractal embedding draw `Z_p` on the line and in the plane.
Equations `du/dt = L u + N(t, u)` are `GroupODE`s on the cyclic group `Z/p^n`, solved
with `galrax.groups.solve_group_ode`.
"""

from .integers import PAdicIntegers as PAdicIntegers
from .operator import UltrametricOperator as UltrametricOperator
from .plot import (
    fractal_embedding as fractal_embedding,
    monna as monna,
    plot_kozyrev_wavelets as plot_kozyrev_wavelets,
)
from .wavelets import KozyrevTransform as KozyrevTransform
