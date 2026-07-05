r"""Sparse modular polynomials in Y1,Y2,Y3,Y4.

A polynomial is represented as:

```text
{(e1,e2,e3,e4): coefficient_mod_p}
```

for the monomial \(Y_1^{e1}Y_2^{e2}Y_3^{e3}Y_4^{e4}\).
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .arithmetic import sympy_rational_mod

Alpha = Tuple[int, int, int, int]
SparsePoly = Dict[Alpha, int]

ZERO_ALPHA: Alpha = (0, 0, 0, 0)


def mod_inv(value: int, p: int) -> int:
    value %= p
    if value == 0:
        raise ZeroDivisionError(f"denominator is zero modulo {p}")
    return pow(value, -1, p)


def as_alpha(values: Sequence[int]) -> Alpha:
    if len(values) != 4:
        raise ValueError(f"alpha must have length 4, got {len(values)}")
    out = tuple(int(value) for value in values)
    if any(value < 0 for value in out):
        raise ValueError(f"alpha exponents must be nonnegative: {out}")
    return out  # type: ignore[return-value]


def clean(poly: SparsePoly, p: int) -> SparsePoly:
    return {as_alpha(alpha): int(coeff) % p for alpha, coeff in poly.items() if int(coeff) % p}


def clean_known(poly: SparsePoly, p: int) -> SparsePoly:
    """Clean a polynomial whose alpha keys are already known valid."""

    return {alpha: int(coeff) % p for alpha, coeff in poly.items() if int(coeff) % p}


def const_poly(value: int, p: int) -> SparsePoly:
    value %= p
    return {} if value == 0 else {ZERO_ALPHA: value}


def add(left: SparsePoly, right: SparsePoly, p: int, *, scale: int = 1) -> SparsePoly:
    out = dict(left)
    scale %= p
    if not scale:
        return clean(out, p)
    for alpha, coeff in right.items():
        alpha = as_alpha(alpha)
        value = (out.get(alpha, 0) + scale * int(coeff)) % p
        if value:
            out[alpha] = value
        else:
            out.pop(alpha, None)
    return out


def neg(poly: SparsePoly, p: int) -> SparsePoly:
    return scale(poly, -1, p)


def sub(left: SparsePoly, right: SparsePoly, p: int) -> SparsePoly:
    return add(left, right, p, scale=-1)


def scale(poly: SparsePoly, scalar: int, p: int) -> SparsePoly:
    scalar %= p
    if not scalar:
        return {}
    out: SparsePoly = {}
    for alpha, coeff in poly.items():
        value = int(coeff) * scalar % p
        if value:
            out[alpha] = value
    return out


def mul(left: SparsePoly, right: SparsePoly, p: int) -> SparsePoly:
    if not left or not right:
        return {}
    out: SparsePoly = {}
    for alpha_left, coeff_left in left.items():
        coeff_left_int = int(coeff_left)
        for alpha_right, coeff_right in right.items():
            alpha = (
                alpha_left[0] + alpha_right[0],
                alpha_left[1] + alpha_right[1],
                alpha_left[2] + alpha_right[2],
                alpha_left[3] + alpha_right[3],
            )
            out[alpha] = out.get(alpha, 0) + coeff_left_int * int(coeff_right)
    return clean_known(out, p)


def prod(polys: Iterable[SparsePoly], p: int) -> SparsePoly:
    out: SparsePoly = {ZERO_ALPHA: 1}
    for poly in polys:
        out = mul(out, poly, p)
        if not out:
            break
    return out


def pow_poly(base: SparsePoly, exponent: int, p: int) -> SparsePoly:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("negative polynomial powers are not supported")
    out: SparsePoly = {ZERO_ALPHA: 1}
    cur = clean(base, p)
    while exponent:
        if exponent & 1:
            out = mul(out, cur, p)
        exponent >>= 1
        if exponent:
            cur = mul(cur, cur, p)
    return out


def derivative(poly: SparsePoly, var_idx: int, p: int) -> SparsePoly:
    if var_idx < 0 or var_idx >= 4:
        raise ValueError(f"var_idx must be 0,1,2,3; got {var_idx}")
    out: SparsePoly = {}
    for alpha, coeff in poly.items():
        alpha = as_alpha(alpha)
        power = alpha[var_idx]
        if not power:
            continue
        next_alpha = list(alpha)
        next_alpha[var_idx] -= 1
        next_alpha_t = as_alpha(next_alpha)
        out[next_alpha_t] = (out.get(next_alpha_t, 0) + int(coeff) * power) % p
    return clean(out, p)


def directional_derivative(poly: SparsePoly, direction: Sequence[int], p: int) -> SparsePoly:
    if len(direction) != 4:
        raise ValueError("direction must have length 4")
    out: SparsePoly = {}
    for var_idx, coeff in enumerate(direction):
        if coeff:
            out = add(out, derivative(poly, var_idx, p), p, scale=int(coeff))
    return out


def linear_poly(coeffs_over_5: Sequence[int], p: int) -> SparsePoly:
    r"""Build \((c1 Y1 + ... + c4 Y4)/5\) modulo p."""

    if len(coeffs_over_5) != 4:
        raise ValueError("linear_poly expects four coefficients")
    inv5 = mod_inv(5, p)
    out: SparsePoly = {}
    for idx, coeff in enumerate(coeffs_over_5):
        coeff_mod = int(coeff) * inv5 % p
        if not coeff_mod:
            continue
        alpha = [0, 0, 0, 0]
        alpha[idx] = 1
        out[as_alpha(alpha)] = coeff_mod
    return out


def sorted_items(poly: SparsePoly) -> Tuple[Tuple[Alpha, int], ...]:
    return tuple(sorted(poly.items()))


def from_sympy_poly(expr, symbols: Sequence, p: int) -> SparsePoly:
    """Convert a SymPy polynomial in Y-symbols to SparsePoly modulo p."""

    import sympy as sp

    poly = sp.Poly(expr, *symbols)
    out: SparsePoly = {}
    for monom, coeff in poly.terms():
        out[as_alpha(monom)] = sympy_rational_mod(sp.Rational(coeff), p)
    return clean(out, p)
