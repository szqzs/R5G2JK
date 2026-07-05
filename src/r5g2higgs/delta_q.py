"""Truncated delta-polynomial arithmetic over Q."""

from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Dict, List, Sequence, Tuple

from .delta_mod import DELTA_UNITS, ZERO_DELTA, ZERO_DERIV, DeltaKey, DerivOrders, delta_add, delta_leq
from .sparse_mod import ZERO_ALPHA
from .sparse_q import SparseQPoly, add as sparse_add, clean as sparse_clean, mul as sparse_mul, scale as sparse_scale, sorted_items

DeltaQPoly = Dict[DeltaKey, SparseQPoly]
KernelQTerms = Dict[Tuple[DeltaKey, DerivOrders], SparseQPoly]


def clean(poly: DeltaQPoly) -> DeltaQPoly:
    out: DeltaQPoly = {}
    for delta, value in poly.items():
        value = sparse_clean(value)
        if value:
            out[tuple(int(item) for item in delta)] = value  # type: ignore[assignment]
    return out


def add(left: DeltaQPoly, right: DeltaQPoly, *, scale: int | Fraction = 1) -> DeltaQPoly:
    out: DeltaQPoly = {delta: dict(poly) for delta, poly in left.items()}
    for delta, poly in right.items():
        out[delta] = sparse_add(out.get(delta, {}), poly, scale=scale)
        if not out[delta]:
            del out[delta]
    return out


def scale(poly: DeltaQPoly, scalar: int | Fraction) -> DeltaQPoly:
    out: DeltaQPoly = {}
    for delta, value in poly.items():
        scaled = sparse_scale(value, scalar)
        if scaled:
            out[delta] = scaled
    return out


def mul(left: DeltaQPoly, right: DeltaQPoly, max_delta: DeltaKey) -> DeltaQPoly:
    out: DeltaQPoly = {}
    for d1, p1 in left.items():
        for d2, p2 in right.items():
            delta = delta_add(d1, d2)
            if not delta_leq(delta, max_delta):
                continue
            product = sparse_mul(p1, p2)
            if product:
                out[delta] = sparse_add(out.get(delta, {}), product)
    return out


def pow_delta(base: DeltaQPoly, exponent: int, max_delta: DeltaKey) -> DeltaQPoly:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("negative delta powers are not supported")
    out: DeltaQPoly = {ZERO_DELTA: {ZERO_ALPHA: Fraction(1)}}
    cur = clean(base)
    while exponent:
        if exponent & 1:
            out = mul(out, cur, max_delta)
        exponent >>= 1
        if exponent:
            cur = mul(cur, cur, max_delta)
    return out


def exp_linear(linear: Dict[DeltaKey, SparseQPoly], max_delta: DeltaKey) -> DeltaQPoly:
    """Return the truncated exp(sum delta_i * poly_i) over Q."""

    out: DeltaQPoly = {}
    powers: Dict[DeltaKey, Tuple[SparseQPoly, ...]] = {}
    for unit in DELTA_UNITS:
        unit_index = unit.index(1)
        poly = linear.get(unit, {})
        unit_powers: List[SparseQPoly] = [{ZERO_ALPHA: Fraction(1)}]
        for _ in range(1, max_delta[unit_index] + 1):
            unit_powers.append(sparse_mul(unit_powers[-1], poly))
        powers[unit] = tuple(unit_powers)

    inv_factorials = {n: Fraction(1, factorial(n)) for n in range(sum(max_delta) + 1)}
    for e3 in range(max_delta[0] + 1):
        for e4 in range(max_delta[1] + 1):
            for e5 in range(max_delta[2] + 1):
                term: SparseQPoly = {ZERO_ALPHA: Fraction(1)}
                for exponent, unit in ((e3, (1, 0, 0)), (e4, (0, 1, 0)), (e5, (0, 0, 1))):
                    if exponent:
                        term = sparse_mul(term, powers[unit][exponent])
                        term = sparse_scale(term, inv_factorials[exponent])
                if term:
                    out[(e3, e4, e5)] = term
    return out


def kernel_terms_mul_delta(terms: KernelQTerms, poly: DeltaQPoly, max_delta: DeltaKey) -> KernelQTerms:
    """Multiply kernel terms by a delta polynomial without changing derivatives."""

    out: KernelQTerms = {}
    for (term_delta, deriv), term_poly in terms.items():
        for poly_delta, poly_value in poly.items():
            next_delta = delta_add(term_delta, poly_delta)
            if not delta_leq(next_delta, max_delta):
                continue
            product = sparse_mul(term_poly, poly_value)
            if product:
                key = (next_delta, deriv)
                out[key] = sparse_add(out.get(key, {}), product)
    return {key: value for key, value in out.items() if value}


def matrix_identity(size: int) -> Tuple[Tuple[DeltaQPoly, ...], ...]:
    return tuple(
        tuple({ZERO_DELTA: {ZERO_ALPHA: Fraction(1)}} if row == col else {} for col in range(size))
        for row in range(size)
    )


def matrix_mul(
    left: Sequence[Sequence[DeltaQPoly]],
    right: Sequence[Sequence[DeltaQPoly]],
    max_delta: DeltaKey,
) -> Tuple[Tuple[DeltaQPoly, ...], ...]:
    row_count = len(left)
    col_count = len(right[0])
    inner_count = len(right)
    out: List[List[DeltaQPoly]] = [[{} for _ in range(col_count)] for _ in range(row_count)]
    for row in range(row_count):
        for col in range(col_count):
            accum: DeltaQPoly = {}
            for inner in range(inner_count):
                accum = add(accum, mul(left[row][inner], right[inner][col], max_delta))
            out[row][col] = accum
    return tuple(tuple(row) for row in out)


def sorted_delta_items(poly: DeltaQPoly) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, int, int, int], Fraction], ...]], ...]:
    return tuple(sorted((delta, sorted_items(value)) for delta, value in poly.items() if value))
