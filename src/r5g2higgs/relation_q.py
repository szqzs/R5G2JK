"""Exact-Q checks for the lifted c=12 relation."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import Dict, Sequence, Tuple

from .basis_mod import add_exp
from .constants import COLLAPSED_CENTRAL_PREFACTOR, SOURCE_DEGREE, W_DEGREE
from .delta_mod import DeltaKey
from .matrix_mod import source_basis, target_basis
from .model import FourTuple
from .pairing_q import f_factorial_scale_q, pairing_kernel_gamma_products_q
from .residue_q import residue_poly_q
from .sparse_q import SparseQPoly, add as sparse_add, mul as sparse_mul, scale as sparse_scale
from .tau_q import tau_power_q

C12_RELATION_INTEGER_VECTOR: Tuple[int, ...] = (
    32,
    -48,
    -16,
    -208,
    -192,
    0,
    4,
    -312,
    240,
    92,
    256,
    96,
    240,
    40,
    16,
    288,
    87,
    576,
    480,
    60,
    12,
    256,
    -64,
    -48,
    -250,
    -220,
    -35,
    108,
    288,
    -18,
    -18,
    -450,
    -168,
    -450,
    -150,
    -90,
    -255,
    -64,
    -450,
    -30,
    0,
    -125,
    50,
    150,
)


RelationBucketKey = Tuple[FourTuple, Tuple[int, ...]]


def relation_column_buckets_q(
    w_index: int,
    relation_vector: Sequence[int] = C12_RELATION_INTEGER_VECTOR,
    *,
    chern_degree: int = 12,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Dict[RelationBucketKey, SparseQPoly]:
    """Group the exact polynomial contribution of R5 times one target basis element.

    The key is `(f_exp, gamma_exp)`.  For each key, the value is the already
    coefficient-weighted sum of tau-power polynomials from the source rows.
    """

    source = source_basis(chern_degree, source_degree)
    target = target_basis(w_degree)
    if len(relation_vector) != len(source):
        raise ValueError(f"relation vector length {len(relation_vector)} does not match source dimension {len(source)}")
    if w_index < 0 or w_index >= len(target):
        raise IndexError(f"W index {w_index} outside W dimension {len(target)}")

    buckets: Dict[RelationBucketKey, SparseQPoly] = {}
    target_exp = target[int(w_index)].exp
    for row_index, coefficient in enumerate(relation_vector):
        coefficient = int(coefficient)
        if not coefficient:
            continue
        total = add_exp(source[row_index].exp, target_exp)
        key = (total.f, total.gamma)
        row_poly = sparse_scale(dict(tau_power_q(total.a)), coefficient * f_factorial_scale_q(total.f))
        if row_poly:
            buckets[key] = sparse_add(buckets.get(key, {}), row_poly)
            if not buckets[key]:
                del buckets[key]
    return buckets


@lru_cache(maxsize=None)
def relation_dot_q(w_index: int) -> Fraction:
    """Return <R5 * W[w_index]> exactly over Q."""

    value = Fraction(0)
    for (f_exp, gamma_exp), combined_a_poly in relation_column_buckets_q(w_index).items():
        target_delta: DeltaKey = (f_exp[1], f_exp[2], f_exp[3])
        for deriv_orders, shared_items in pairing_kernel_gamma_products_q(target_delta, gamma_exp):
            full_poly = sparse_mul(combined_a_poly, dict(shared_items))
            if full_poly:
                value += residue_poly_q(full_poly, deriv_orders)
    return COLLAPSED_CENTRAL_PREFACTOR * value
