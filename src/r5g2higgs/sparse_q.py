"""Sparse polynomials over Q in Y1,Y2,Y3,Y4.

This mirrors `sparse_mod.py`, but coefficients are exact `Fraction` values
instead of residues modulo a prime.  It is the first building block for the
exact c=12 relation verifier.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Sequence, Tuple

from .arithmetic import rational_mod
from .sparse_mod import Alpha, ZERO_ALPHA, as_alpha

SparseQPoly = Dict[Alpha, Fraction]


def clean(poly: SparseQPoly) -> SparseQPoly:
    return {as_alpha(alpha): Fraction(coeff) for alpha, coeff in poly.items() if coeff}


def const_poly(value: int | Fraction) -> SparseQPoly:
    value = Fraction(value)
    return {} if value == 0 else {ZERO_ALPHA: value}


def add(left: SparseQPoly, right: SparseQPoly, *, scale: int | Fraction = 1) -> SparseQPoly:
    out = dict(left)
    scale = Fraction(scale)
    if not scale:
        return clean(out)
    for alpha, coeff in right.items():
        alpha = as_alpha(alpha)
        value = out.get(alpha, Fraction(0)) + scale * Fraction(coeff)
        if value:
            out[alpha] = value
        else:
            out.pop(alpha, None)
    return out


def neg(poly: SparseQPoly) -> SparseQPoly:
    return scale(poly, -1)


def sub(left: SparseQPoly, right: SparseQPoly) -> SparseQPoly:
    return add(left, right, scale=-1)


def scale(poly: SparseQPoly, scalar: int | Fraction) -> SparseQPoly:
    scalar = Fraction(scalar)
    if not scalar:
        return {}
    out: SparseQPoly = {}
    for alpha, coeff in poly.items():
        value = Fraction(coeff) * scalar
        if value:
            out[alpha] = value
    return out


def mul(left: SparseQPoly, right: SparseQPoly) -> SparseQPoly:
    if not left or not right:
        return {}
    out: SparseQPoly = {}
    for alpha_left, coeff_left in left.items():
        coeff_left_q = Fraction(coeff_left)
        for alpha_right, coeff_right in right.items():
            alpha = (
                alpha_left[0] + alpha_right[0],
                alpha_left[1] + alpha_right[1],
                alpha_left[2] + alpha_right[2],
                alpha_left[3] + alpha_right[3],
            )
            out[alpha] = out.get(alpha, Fraction(0)) + coeff_left_q * Fraction(coeff_right)
    return clean(out)


def prod(polys: Iterable[SparseQPoly]) -> SparseQPoly:
    out: SparseQPoly = {ZERO_ALPHA: Fraction(1)}
    for poly in polys:
        out = mul(out, poly)
        if not out:
            break
    return out


def pow_poly(base: SparseQPoly, exponent: int) -> SparseQPoly:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("negative polynomial powers are not supported")
    out: SparseQPoly = {ZERO_ALPHA: Fraction(1)}
    cur = clean(base)
    while exponent:
        if exponent & 1:
            out = mul(out, cur)
        exponent >>= 1
        if exponent:
            cur = mul(cur, cur)
    return out


def derivative(poly: SparseQPoly, var_idx: int) -> SparseQPoly:
    if var_idx < 0 or var_idx >= 4:
        raise ValueError(f"var_idx must be 0,1,2,3; got {var_idx}")
    out: SparseQPoly = {}
    for alpha, coeff in poly.items():
        alpha = as_alpha(alpha)
        power = alpha[var_idx]
        if not power:
            continue
        next_alpha = list(alpha)
        next_alpha[var_idx] -= 1
        next_alpha_t = as_alpha(next_alpha)
        out[next_alpha_t] = out.get(next_alpha_t, Fraction(0)) + Fraction(coeff) * power
    return clean(out)


def directional_derivative(poly: SparseQPoly, direction: Sequence[int]) -> SparseQPoly:
    if len(direction) != 4:
        raise ValueError("direction must have length 4")
    out: SparseQPoly = {}
    for var_idx, coeff in enumerate(direction):
        if coeff:
            out = add(out, derivative(poly, var_idx), scale=int(coeff))
    return out


def linear_poly(coeffs_over_5: Sequence[int]) -> SparseQPoly:
    """Build (c1 Y1 + ... + c4 Y4)/5 over Q."""

    if len(coeffs_over_5) != 4:
        raise ValueError("linear_poly expects four coefficients")
    out: SparseQPoly = {}
    for idx, coeff in enumerate(coeffs_over_5):
        coeff_q = Fraction(int(coeff), 5)
        if not coeff_q:
            continue
        alpha = [0, 0, 0, 0]
        alpha[idx] = 1
        out[as_alpha(alpha)] = coeff_q
    return out


def sorted_items(poly: SparseQPoly) -> Tuple[Tuple[Alpha, Fraction], ...]:
    return tuple(sorted(clean(poly).items()))


def reduce_mod_p(poly: SparseQPoly, p: int) -> Dict[Alpha, int]:
    """Reduce an exact sparse polynomial modulo p."""

    return {
        alpha: rational_mod(coeff, p)
        for alpha, coeff in clean(poly).items()
        if rational_mod(coeff, p)
    }
