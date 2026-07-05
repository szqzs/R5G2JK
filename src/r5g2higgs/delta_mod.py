"""Truncated delta-polynomial arithmetic.

The variables

    delta3, delta4, delta5

are bookkeeping variables for the higher f-insertions.  A delta monomial is
stored as a triple:

    (power of delta3, power of delta4, power of delta5).

The coefficient of each delta monomial is itself a sparse polynomial in the
residue variables Y1,...,Y4.

All multiplication is truncated by a `max_delta`.  This is exact for
coefficient extraction: if we only need one target delta coefficient, higher
delta powers can never contribute to it.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .arithmetic import mod_factorial
from .sparse_mod import SparsePoly, ZERO_ALPHA, add as sparse_add, clean as sparse_clean, mod_inv, mul as sparse_mul, scale as sparse_scale, sorted_items

DeltaKey = Tuple[int, int, int]
DerivOrders = Tuple[int, int, int, int]
DeltaPoly = Dict[DeltaKey, SparsePoly]
KernelTerms = Dict[Tuple[DeltaKey, DerivOrders], SparsePoly]

ZERO_DELTA: DeltaKey = (0, 0, 0)
ZERO_DERIV: DerivOrders = (0, 0, 0, 0)
DELTA_UNITS: Tuple[DeltaKey, DeltaKey, DeltaKey] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def delta_leq(left: DeltaKey, right: DeltaKey) -> bool:
    return all(int(left[idx]) <= int(right[idx]) for idx in range(3))


def delta_add(left: DeltaKey, right: DeltaKey) -> DeltaKey:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def delta_sub(left: DeltaKey, right: DeltaKey) -> DeltaKey | None:
    out = (left[0] - right[0], left[1] - right[1], left[2] - right[2])
    if min(out) < 0:
        return None
    return out


def clean(poly: DeltaPoly, p: int) -> DeltaPoly:
    out: DeltaPoly = {}
    for delta, value in poly.items():
        value = sparse_clean(value, p)
        if value:
            out[tuple(int(item) for item in delta)] = value  # type: ignore[assignment]
    return out


def add(left: DeltaPoly, right: DeltaPoly, p: int, *, scale: int = 1) -> DeltaPoly:
    out: DeltaPoly = {delta: dict(poly) for delta, poly in left.items()}
    for delta, poly in right.items():
        out[delta] = sparse_add(out.get(delta, {}), poly, p, scale=scale)
        if not out[delta]:
            del out[delta]
    return out


def scale(poly: DeltaPoly, scalar: int, p: int) -> DeltaPoly:
    out: DeltaPoly = {}
    for delta, value in poly.items():
        scaled = sparse_scale(value, scalar, p)
        if scaled:
            out[delta] = scaled
    return out


def mul(left: DeltaPoly, right: DeltaPoly, max_delta: DeltaKey, p: int) -> DeltaPoly:
    out: DeltaPoly = {}
    for d1, p1 in left.items():
        for d2, p2 in right.items():
            delta = delta_add(d1, d2)
            if not delta_leq(delta, max_delta):
                continue
            product = sparse_mul(p1, p2, p)
            if product:
                out[delta] = sparse_add(out.get(delta, {}), product, p)
    return out


def pow_delta(base: DeltaPoly, exponent: int, max_delta: DeltaKey, p: int) -> DeltaPoly:
    exponent = int(exponent)
    if exponent < 0:
        raise ValueError("negative delta powers are not supported")
    out: DeltaPoly = {ZERO_DELTA: {ZERO_ALPHA: 1}}
    cur = clean(base, p)
    while exponent:
        if exponent & 1:
            out = mul(out, cur, max_delta, p)
        exponent >>= 1
        if exponent:
            cur = mul(cur, cur, max_delta, p)
    return out


def exp_linear(linear: Dict[DeltaKey, SparsePoly], max_delta: DeltaKey, p: int) -> DeltaPoly:
    """Return the truncated exp(sum delta_i * poly_i).

    The three delta variables are independent, so the exponential factors into
    one-variable exponential series.  We only build terms up to `max_delta`.
    """

    out: DeltaPoly = {}
    powers: Dict[DeltaKey, Tuple[SparsePoly, ...]] = {}
    for unit in DELTA_UNITS:
        unit_index = unit.index(1)
        poly = linear.get(unit, {})
        unit_powers: List[SparsePoly] = [{ZERO_ALPHA: 1}]
        for _ in range(1, max_delta[unit_index] + 1):
            unit_powers.append(sparse_mul(unit_powers[-1], poly, p))
        powers[unit] = tuple(unit_powers)

    inv_factorials = {
        n: mod_inv(mod_factorial(n, p), p)
        for n in range(sum(max_delta) + 1)
    }
    for e3 in range(max_delta[0] + 1):
        for e4 in range(max_delta[1] + 1):
            for e5 in range(max_delta[2] + 1):
                term: SparsePoly = {ZERO_ALPHA: 1}
                for exponent, unit in ((e3, (1, 0, 0)), (e4, (0, 1, 0)), (e5, (0, 0, 1))):
                    if exponent:
                        term = sparse_mul(term, powers[unit][exponent], p)
                        term = sparse_scale(term, inv_factorials[exponent], p)
                if term:
                    out[(e3, e4, e5)] = term
    return out


def kernel_terms_mul_delta(terms: KernelTerms, poly: DeltaPoly, max_delta: DeltaKey, p: int) -> KernelTerms:
    """Multiply kernel terms by a delta polynomial without changing derivatives."""

    out: KernelTerms = {}
    for (term_delta, deriv), term_poly in terms.items():
        for poly_delta, poly_value in poly.items():
            next_delta = delta_add(term_delta, poly_delta)
            if not delta_leq(next_delta, max_delta):
                continue
            product = sparse_mul(term_poly, poly_value, p)
            if product:
                key = (next_delta, deriv)
                out[key] = sparse_add(out.get(key, {}), product, p)
    return {key: value for key, value in out.items() if value}


def matrix_identity(size: int) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    return tuple(
        tuple({ZERO_DELTA: {ZERO_ALPHA: 1}} if row == col else {} for col in range(size))
        for row in range(size)
    )


def matrix_mul(
    left: Sequence[Sequence[DeltaPoly]],
    right: Sequence[Sequence[DeltaPoly]],
    max_delta: DeltaKey,
    p: int,
) -> Tuple[Tuple[DeltaPoly, ...], ...]:
    row_count = len(left)
    col_count = len(right[0])
    inner_count = len(right)
    out: List[List[DeltaPoly]] = [[{} for _ in range(col_count)] for _ in range(row_count)]
    for row in range(row_count):
        for col in range(col_count):
            accum: DeltaPoly = {}
            for inner in range(inner_count):
                accum = add(accum, mul(left[row][inner], right[inner][col], max_delta, p), p)
            out[row][col] = accum
    return tuple(tuple(row) for row in out)


def sorted_delta_items(poly: DeltaPoly) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Tuple[int, int, int, int], int], ...]], ...]:
    return tuple(sorted((delta, sorted_items(value)) for delta, value in poly.items() if value))
