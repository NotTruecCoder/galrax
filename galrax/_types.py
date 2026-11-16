"""The Diffrax type aliases used by `galrax.groups` (private in Diffrax)."""

from typing import Any, TYPE_CHECKING

import numpy as np
from jaxtyping import Array, ArrayLike, Bool, PyTree, Real, Shaped


if TYPE_CHECKING:
    BoolScalarLike = bool | Array | np.ndarray
    RealScalarLike = int | float | Array | np.ndarray
else:
    BoolScalarLike = Bool[ArrayLike, ""]
    RealScalarLike = Real[ArrayLike, ""]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Args = PyTree[Any]
DenseInfo = dict[str, PyTree[Array]]
