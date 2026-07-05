"""Small exact/modular arithmetic helpers."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Any

import sympy as sp


def rational_mod(value: Any, p: int) -> int:
    """Reduce an exact rational value modulo the prime p."""

    rational = Fraction(value)
    numerator = rational.numerator % p
    denominator = rational.denominator % p
    if denominator == 0:
        raise ZeroDivisionError(f"denominator is zero modulo {p}")
    return numerator * pow(denominator, -1, p) % p


def sympy_rational_mod(value: sp.Expr, p: int) -> int:
    """Reduce a SymPy integer/rational expression modulo p."""

    value = sp.factor(value)
    if not value.is_Rational:
        raise TypeError(f"expected a rational SymPy value, got {value!r}")
    return rational_mod(Fraction(int(value.p), int(value.q)), p)


@lru_cache(maxsize=None)
def mod_factorial(n: int, p: int) -> int:
    if n < 0:
        raise ValueError("factorial input must be nonnegative")
    out = 1
    for value in range(2, int(n) + 1):
        out = out * value % p
    return out


@lru_cache(maxsize=None)
def f_factorial_scale_mod(f_exp: tuple[int, int, int, int], p: int) -> int:
    scale = 1
    for exponent in f_exp:
        scale = scale * mod_factorial(int(exponent), p) % p
    return scale
