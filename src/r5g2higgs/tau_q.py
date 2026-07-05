"""Exact-Q rank-5 tau polynomials in residue coordinates.

This mirrors the tau layer in `tau_mod.py`, but keeps rational coefficients.
The c=12 relation verifier will use this exact layer rather than SymPy's much
heavier symbolic rational functions.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from typing import Tuple

from .model import FourTuple
from .sparse_mod import Alpha, ZERO_ALPHA
from .sparse_q import (
    SparseQPoly,
    add,
    directional_derivative,
    derivative,
    linear_poly,
    mul,
    pow_poly,
    sorted_items,
)
from .tau_mod import C_TILDE_DIRECTION_Y, SIMPLE_COROOT_DIRECTIONS, X_COORDINATE_NUMERATORS


@lru_cache(maxsize=None)
def rank5_x_polys_q() -> Tuple[Tuple[Tuple[Alpha, Fraction], ...], ...]:
    """Return the five linear x_i(Y) polynomials over Q."""

    return tuple(sorted_items(linear_poly(numerators)) for numerators in X_COORDINATE_NUMERATORS)


@lru_cache(maxsize=None)
def rank5_tau_q(r: int) -> Tuple[Tuple[Alpha, Fraction], ...]:
    """Return tau_r(Y), the r-th elementary symmetric polynomial in the x_i."""

    if r < 2 or r > 5:
        raise ValueError(f"rank-5 tau index must be 2,3,4,5; got {r}")
    x_polys = [dict(items) for items in rank5_x_polys_q()]
    total: SparseQPoly = {}
    for combo in combinations(x_polys, r):
        term: SparseQPoly = {ZERO_ALPHA: 1}
        for poly in combo:
            term = mul(term, poly)
        total = add(total, term)
    return sorted_items(total)


@lru_cache(maxsize=None)
def tau_power_q(a_exp: FourTuple) -> Tuple[Tuple[Alpha, Fraction], ...]:
    """Return the product tau2^a2 tau3^a3 tau4^a4 tau5^a5 over Q."""

    out: SparseQPoly = {ZERO_ALPHA: 1}
    for offset, exponent in enumerate(a_exp):
        if exponent:
            out = mul(out, pow_poly(dict(rank5_tau_q(offset + 2)), int(exponent)))
    return sorted_items(out)


@lru_cache(maxsize=None)
def tau_gradient_q(r: int) -> Tuple[Tuple[Tuple[Alpha, Fraction], ...], ...]:
    """Return the gradient of tau_r with respect to Y1,...,Y4 over Q."""

    tau_poly = dict(rank5_tau_q(r))
    return tuple(sorted_items(derivative(tau_poly, var_idx)) for var_idx in range(4))


@lru_cache(maxsize=None)
def tau_hessian_q(r: int) -> Tuple[Tuple[Tuple[Tuple[Alpha, Fraction], ...], ...], ...]:
    """Return the Hessian matrix of tau_r in the Y variables over Q."""

    tau_poly = dict(rank5_tau_q(r))
    rows = []
    for left_idx in range(4):
        row = []
        first = derivative(tau_poly, left_idx)
        for right_idx in range(4):
            row.append(sorted_items(derivative(first, right_idx)))
        rows.append(tuple(row))
    return tuple(rows)


@lru_cache(maxsize=None)
def c_direction_term_q(r: int) -> Tuple[Tuple[Alpha, Fraction], ...]:
    """Return the tau_r contribution to dq(c-tilde) over Q."""

    if r < 3 or r > 5:
        raise ValueError(f"expected r=3,4,5; got {r}")
    return sorted_items(directional_derivative(dict(rank5_tau_q(r)), C_TILDE_DIRECTION_Y))


@lru_cache(maxsize=None)
def b_perturbation_q(r: int, j: int) -> Tuple[Tuple[Alpha, Fraction], ...]:
    """Return -d(tau_r)(h_j) over Q."""

    if r < 3 or r > 5 or j < 1 or j > 4:
        raise ValueError(f"expected r=3,4,5 and j=1,2,3,4; got {(r, j)}")
    deriv = directional_derivative(dict(rank5_tau_q(r)), SIMPLE_COROOT_DIRECTIONS[j - 1])
    return sorted_items({alpha: -coeff for alpha, coeff in deriv.items()})
