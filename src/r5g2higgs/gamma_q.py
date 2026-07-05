"""Exact-Q odd/gamma layer for the JK evaluator."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Dict, Optional, Sequence, Tuple

from . import delta_q as dq
from .constants import B_INDEX, B_LABELS, GAMMA_LABELS
from .delta_mod import DeltaKey
from .kernel_q import hat_pair_delta_q
from .reference_sympy import gamma_product_to_b_terms
from .sparse_mod import Alpha, ZERO_ALPHA
from .sparse_q import scale as sparse_scale

BLabel = Tuple[int, int]


def bit_for_label(label: BLabel) -> int:
    return 1 << B_INDEX[label]


def labels_from_mask(mask: int) -> Tuple[BLabel, ...]:
    return tuple(label for idx, label in enumerate(B_LABELS) if mask & (1 << idx))


def wedge_masks(left: int, right: int) -> Optional[Tuple[int, int]]:
    if left & right:
        return None
    inversions = 0
    for idx in range(len(B_LABELS)):
        if not (left & (1 << idx)):
            continue
        inversions += sum(1 for lower in range(idx) if right & (1 << lower))
    return (-1 if inversions % 2 else 1, left | right)


@lru_cache(maxsize=None)
def gamma_mask_expansion_q(gamma_exp: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    """Expand a gamma monomial into exterior b-masks with integer coefficients."""

    if len(gamma_exp) != len(GAMMA_LABELS):
        raise ValueError(f"gamma exponent tuple must have length {len(GAMMA_LABELS)}")
    out = []
    for coeff, labels in gamma_product_to_b_terms(gamma_exp):
        mask = 0
        for label in labels:
            mask |= bit_for_label(label)
        if coeff:
            out.append((mask, int(coeff)))
    return tuple(sorted(out))


def ext_delta_mul_pruned_q(
    left: Dict[int, dq.DeltaQPoly],
    right: Dict[int, dq.DeltaQPoly],
    max_delta: DeltaKey,
    target_mask: int,
    target_len: int,
) -> Dict[int, dq.DeltaQPoly]:
    """Multiply exterior/delta expressions while discarding impossible masks."""

    out: Dict[int, dq.DeltaQPoly] = {}
    for left_mask, left_delta in left.items():
        for right_mask, right_delta in right.items():
            wedge = wedge_masks(left_mask, right_mask)
            if wedge is None:
                continue
            sign, mask = wedge
            if mask.bit_count() > target_len or (mask | target_mask) != target_mask:
                continue
            product = dq.mul(left_delta, right_delta, max_delta)
            if product:
                out[mask] = dq.add(out.get(mask, {}), product, scale=sign)
    return {mask: poly for mask, poly in out.items() if poly}


@lru_cache(maxsize=None)
def b_hat_mask_q(
    mask: int,
    max_delta: DeltaKey,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Contract one exterior b-monomial over Q."""

    target = labels_from_mask(mask)
    if len(target) % 2:
        return ()
    if not target:
        return ((dq.ZERO_DELTA, ((ZERO_ALPHA, Fraction(1)),)),)

    pair_terms: Dict[int, dq.DeltaQPoly] = {}
    target_set = set(target)
    for left_side, right_side in ((1, 3), (2, 4)):
        left_labels = [label for label in target if label[1] == left_side]
        right_labels = [label for label in target if label[1] == right_side]
        for left_label in left_labels:
            for right_label in right_labels:
                pair_mask = bit_for_label(left_label) | bit_for_label(right_label)
                if any(label not in target_set for label in labels_from_mask(pair_mask)):
                    continue
                wedge = wedge_masks(bit_for_label(left_label), bit_for_label(right_label))
                if wedge is None:
                    continue
                sign, odd_mask = wedge
                coeff: dq.DeltaQPoly = {
                    delta: dict(items)
                    for delta, items in hat_pair_delta_q(left_label[0], right_label[0], max_delta)
                }
                pair_terms[odd_mask] = dq.add(pair_terms.get(odd_mask, {}), coeff, scale=sign)

    pair_count = len(target) // 2
    power: Dict[int, dq.DeltaQPoly] = {0: {dq.ZERO_DELTA: {ZERO_ALPHA: Fraction(1)}}}
    for _ in range(pair_count):
        power = ext_delta_mul_pruned_q(power, pair_terms, max_delta, mask, len(target))
        if not power:
            return ()

    result = {
        delta: sparse_scale(poly, Fraction(1, factorial(pair_count)))
        for delta, poly in power.get(mask, {}).items()
    }
    return dq.sorted_delta_items(result)


@lru_cache(maxsize=None)
def gamma_hat_q(
    gamma_exp: Tuple[int, ...],
    target_delta: DeltaKey,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Return the contracted gamma contribution up to `target_delta` over Q."""

    out: dq.DeltaQPoly = {}
    for mask, coeff in gamma_mask_expansion_q(gamma_exp):
        b_delta = {
            delta: dict(poly_items)
            for delta, poly_items in b_hat_mask_q(mask, target_delta)
        }
        out = dq.add(out, b_delta, scale=coeff)
    return dq.sorted_delta_items(out)
