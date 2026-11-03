"""Differential equations on finite groups.

Functions on a finite group `G`, convolution equations
`dp/dt = lambda * p + p * mu + N(t, p)`, Fourier transforms on `G` (direct and fast),
and solvers that integrate the convolution part exactly.
"""

from ._group_algebra import (
    AbstractExponentialTerm as AbstractExponentialTerm,
    AbstractFourierTransform as AbstractFourierTransform,
    AbstractSpectralTransform as AbstractSpectralTransform,
    convolution as convolution,
    DiagonalLinearTerm as DiagonalLinearTerm,
    exact_linear_solution as exact_linear_solution,
    FastFourierTransform as FastFourierTransform,
    fourier_term as fourier_term,
    fourier_terms as fourier_terms,
    FourierLinearTerm as FourierLinearTerm,
    FourierTransform as FourierTransform,
    Group as Group,
    GroupODE as GroupODE,
    GroupSolution as GroupSolution,
    Lawson as Lawson,
    physical_term as physical_term,
    solve_group_ode as solve_group_ode,
    subgroup_chain as subgroup_chain,
    subgroup_closure as subgroup_closure,
)
