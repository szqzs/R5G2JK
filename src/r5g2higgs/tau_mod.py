"""Rank-5 tau polynomials in the residue coordinates.

This file is the first place where the Jeffrey-Kirwan formula becomes a
finite calculation.

Mathematical role
-----------------
After passing to maximal torus coordinates, the rank-5 calculation is written
in four residue variables:

    Y1, Y2, Y3, Y4.

The five torus Chern roots x_i are linear functions of these Y variables, with
the trace-zero condition already built in.  The classes a_r are represented by
the elementary symmetric polynomials in the x_i:

    tau_r(Y) = elementary symmetric polynomial e_r(x_1(Y), ..., x_5(Y)).

The code stores each tau_r as a sparse polynomial over F_p.

Why q and delta appear later
----------------------------
The f_2 class gives the basic symplectic/exponential term.  The higher
f-classes are handled by introducing formal bookkeeping variables

    delta3, delta4, delta5

and replacing tau_2 by

    q = tau_2 + delta3 tau_3 + delta4 tau_4 + delta5 tau_5.

The rest of the code expands in these delta variables and extracts the
coefficient selected by the powers of f3, f4, and f5.  The functions here
provide the tau polynomials, their derivatives, and the special directional
derivatives used in that expansion.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Sequence, Tuple

from .model import FourTuple
from .sparse_mod import Alpha, SparsePoly, ZERO_ALPHA, add, directional_derivative, derivative, linear_poly, mul, pow_poly, sorted_items

X_COORDINATE_NUMERATORS: Tuple[Tuple[int, int, int, int], ...] = (
    (4, 3, 2, 1),
    (-1, 3, 2, 1),
    (-1, -2, 2, 1),
    (-1, -2, -3, 1),
    (-1, -2, -3, -4),
)
"""Numerators of the five x_i as linear functions of Y.

The denominator is 5.  Since the prime p is not 5, multiplication by 1/5 is
handled inside `linear_poly`.  These five rows encode the trace-zero rank-5
coordinates used throughout the residue calculation.
"""

SIMPLE_COROOT_DIRECTIONS: Tuple[Tuple[int, int, int, int], ...] = (
    (2, -1, 0, 0),
    (-1, 2, -1, 0),
    (0, -1, 2, -1),
    (0, 0, -1, 2),
)
"""Simple coroot directions h_j expressed in Y coordinates.

JK's denominator factors use directional derivatives of q in these directions.
The code uses them to build the perturbations B_j.
"""

C_TILDE_DIRECTION_Y: Tuple[int, int, int, int] = (0, 0, 0, 1)
"""The c-tilde direction in Y coordinates.

For rank 5 and determinant degree 1, dq(c-tilde) becomes differentiation in
the Y4 direction in this coordinate system.
"""


@lru_cache(maxsize=None)
def rank5_x_polys(p: int) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    """Return the five linear x_i(Y) polynomials over F_p."""

    return tuple(
        sorted_items(linear_poly(numerators, p))
        for numerators in X_COORDINATE_NUMERATORS
    )


@lru_cache(maxsize=None)
def rank5_tau(r: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    """Return tau_r(Y), the r-th elementary symmetric polynomial in the x_i."""

    if r < 2 or r > 5:
        raise ValueError(f"rank-5 tau index must be 2,3,4,5; got {r}")
    x_polys = [dict(items) for items in rank5_x_polys(p)]
    total: SparsePoly = {}
    for combo in combinations(x_polys, r):
        term: SparsePoly = {ZERO_ALPHA: 1}
        for poly in combo:
            term = mul(term, poly, p)
        total = add(total, term, p)
    return sorted_items(total)


@lru_cache(maxsize=None)
def tau_power(a_exp: FourTuple, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    """Return the product tau2^a2 tau3^a3 tau4^a4 tau5^a5."""

    out: SparsePoly = {ZERO_ALPHA: 1}
    for offset, exponent in enumerate(a_exp):
        if exponent:
            out = mul(out, pow_poly(dict(rank5_tau(offset + 2, p)), int(exponent), p), p)
    return sorted_items(out)


@lru_cache(maxsize=None)
def tau_gradient(r: int, p: int) -> Tuple[Tuple[Tuple[Alpha, int], ...], ...]:
    """Return the gradient of tau_r with respect to Y1,...,Y4."""

    tau_poly = dict(rank5_tau(r, p))
    return tuple(sorted_items(derivative(tau_poly, var_idx, p)) for var_idx in range(4))


@lru_cache(maxsize=None)
def tau_hessian(r: int, p: int) -> Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...]:
    """Return the Hessian matrix of tau_r in the Y variables."""

    tau_poly = dict(rank5_tau(r, p))
    rows = []
    for left_idx in range(4):
        row = []
        first = derivative(tau_poly, left_idx, p)
        for right_idx in range(4):
            row.append(sorted_items(derivative(first, right_idx, p)))
        rows.append(tuple(row))
    return tuple(rows)


@lru_cache(maxsize=None)
def c_direction_term(r: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    """Return the tau_r contribution to dq(c-tilde).

    Only r=3,4,5 appear because tau_2 is the undeformed base term and
    delta3, delta4, delta5 record the higher f-insertions.
    """

    if r < 3 or r > 5:
        raise ValueError(f"expected r=3,4,5; got {r}")
    return sorted_items(directional_derivative(dict(rank5_tau(r, p)), C_TILDE_DIRECTION_Y, p))


@lru_cache(maxsize=None)
def b_perturbation(r: int, j: int, p: int) -> Tuple[Tuple[Alpha, int], ...]:
    """Return the tau_r contribution to the j-th denominator perturbation.

    This is -d(tau_r)(h_j), where h_j is the j-th simple coroot direction.
    The minus sign matches the denominator factor convention used in the JK
    formula.
    """

    if r < 3 or r > 5 or j < 1 or j > 4:
        raise ValueError(f"expected r=3,4,5 and j=1,2,3,4; got {(r, j)}")
    deriv = directional_derivative(dict(rank5_tau(r, p)), SIMPLE_COROOT_DIRECTIONS[j - 1], p)
    return sorted_items({alpha: (-coeff) % p for alpha, coeff in deriv.items()})
