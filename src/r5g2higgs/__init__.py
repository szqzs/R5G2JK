"""Rank-5, determinant-degree-1, genus-2 pairing-matrix computation in degree 22."""

from .constants import (
    COLLAPSED_CENTRAL_PREFACTOR,
    B_LABELS,
    DEFAULT_PRIME,
    GAMMA_LABELS,
    GENUS,
    RANK,
    SOURCE_DEGREE,
    TOP_DEGREE,
    W_DEGREE,
)
from .model import InvariantExp, add_invariant_exp, zero_invariant_exp
from .arithmetic import rational_mod, sympy_rational_mod
from .sparse_mod import Alpha, SparsePoly, ZERO_ALPHA
from .delta_mod import DeltaKey, DerivOrders, ZERO_DELTA, ZERO_DERIV
from .pairing_mod import pairing_total_mod, pairing_totals_mod
from .pairing_q import pairing_total_q, pairing_totals_q
from .relation_q import C12_RELATION_INTEGER_VECTOR, relation_dot_q
from .linear_mod import LeftKernelResult

__all__ = [
    "COLLAPSED_CENTRAL_PREFACTOR",
    "C12_RELATION_INTEGER_VECTOR",
    "DEFAULT_PRIME",
    "GAMMA_LABELS",
    "GENUS",
    "InvariantExp",
    "LeftKernelResult",
    "RANK",
    "SOURCE_DEGREE",
    "Alpha",
    "B_LABELS",
    "DeltaKey",
    "DerivOrders",
    "SparsePoly",
    "TOP_DEGREE",
    "W_DEGREE",
    "ZERO_ALPHA",
    "ZERO_DELTA",
    "ZERO_DERIV",
    "add_invariant_exp",
    "pairing_total_mod",
    "pairing_total_q",
    "pairing_totals_mod",
    "pairing_totals_q",
    "rational_mod",
    "relation_dot_q",
    "sympy_rational_mod",
    "zero_invariant_exp",
]
