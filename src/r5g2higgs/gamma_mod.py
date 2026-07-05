"""Odd/gamma layer for the modular JK evaluator.

This file handles the odd cohomology classes, but it does so by finite
exterior algebra rather than by any physics-style notation.

The invariant generators gamma_rs are shorthand for symplectic contractions of
the odd generators b_r^j.  The code first expands every gamma product into
exterior monomials in the b variables.  A b-monomial is stored as a bitmask:

    bit i is 1  <=>  the i-th b variable appears.

Exterior multiplication is then easy to check:

    repeated bit  ->  product is zero
    disjoint bits ->  product survives with the wedge sign

After expanding in b variables, the JK odd integral becomes an algebraic
contraction rule.  Pairing two odd variables with ranks r and s contributes

    - grad(tau_r)^T Hessian(q)^(-1) grad(tau_s).

The function `gamma_hat` applies this contraction rule to a gamma monomial and
returns the resulting delta/Y-polynomial contribution.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial
from typing import Dict, Optional, Sequence, Tuple

from .constants import B_INDEX, B_LABELS, GAMMA_LABELS
from . import delta_mod as dm
from .kernel_mod import hat_pair_delta
from .sparse_mod import Alpha, ZERO_ALPHA, mod_inv, scale as sparse_scale
from .reference_sympy import gamma_product_to_b_terms

BLabel = Tuple[int, int]


def bit_for_label(label: BLabel) -> int:
    """Return the exterior bit representing one b_r^j variable."""

    return 1 << B_INDEX[label]


def labels_from_mask(mask: int) -> Tuple[BLabel, ...]:
    """Decode an exterior bitmask back into ordered b labels."""

    return tuple(label for idx, label in enumerate(B_LABELS) if mask & (1 << idx))


def wedge_masks(left: int, right: int) -> Optional[Tuple[int, int]]:
    """Multiply two exterior monomial masks.

    The product is zero if the masks overlap.  Otherwise the sign is the parity
    of the permutation needed to move the right variables past the left
    variables into the fixed global b-order.
    """

    if left & right:
        return None
    inversions = 0
    for idx in range(len(B_LABELS)):
        if not (left & (1 << idx)):
            continue
        inversions += sum(1 for lower in range(idx) if right & (1 << lower))
    return (-1 if inversions % 2 else 1, left | right)


@lru_cache(maxsize=None)
def gamma_mask_expansion(gamma_exp: Tuple[int, ...], p: int) -> Tuple[Tuple[int, int], ...]:
    """Expand a gamma monomial into exterior b-masks."""

    if len(gamma_exp) != len(GAMMA_LABELS):
        raise ValueError(f"gamma exponent tuple must have length {len(GAMMA_LABELS)}")
    out = []
    for coeff, labels in gamma_product_to_b_terms(gamma_exp):
        mask = 0
        for label in labels:
            mask |= bit_for_label(label)
        coeff_mod = int(coeff) % p
        if coeff_mod:
            out.append((mask, coeff_mod))
    return tuple(sorted(out))


def ext_delta_mul_pruned(
    left: Dict[int, dm.DeltaPoly],
    right: Dict[int, dm.DeltaPoly],
    max_delta: dm.DeltaKey,
    target_mask: int,
    target_len: int,
    p: int,
) -> Dict[int, dm.DeltaPoly]:
    """Multiply exterior/delta expressions while discarding impossible masks."""

    out: Dict[int, dm.DeltaPoly] = {}
    for left_mask, left_delta in left.items():
        for right_mask, right_delta in right.items():
            wedge = wedge_masks(left_mask, right_mask)
            if wedge is None:
                continue
            sign, mask = wedge
            if mask.bit_count() > target_len or (mask | target_mask) != target_mask:
                continue
            product = dm.mul(left_delta, right_delta, max_delta, p)
            if product:
                out[mask] = dm.add(out.get(mask, {}), product, p, scale=sign)
    return {mask: poly for mask, poly in out.items() if poly}


@lru_cache(maxsize=None)
def b_hat_mask(
    mask: int,
    max_delta: dm.DeltaKey,
    p: int,
) -> Tuple[Tuple[dm.DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Contract one exterior b-monomial.

    Only even-length b-monomials can survive.  We build all allowed pair
    contractions using the two symplectic pairings (1,3) and (2,4), multiply
    them in the exterior algebra, and divide by pair_count! because the
    exponential expansion counts unordered pairings with that factorial.
    """

    target = labels_from_mask(mask)
    if len(target) % 2:
        return ()
    if not target:
        return ((dm.ZERO_DELTA, ((ZERO_ALPHA, 1),)),)

    pair_terms: Dict[int, dm.DeltaPoly] = {}
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
                coeff: dm.DeltaPoly = {
                    delta: dict(items)
                    for delta, items in hat_pair_delta(left_label[0], right_label[0], max_delta, p)
                }
                pair_terms[odd_mask] = dm.add(pair_terms.get(odd_mask, {}), coeff, p, scale=sign)

    pair_count = len(target) // 2
    power: Dict[int, dm.DeltaPoly] = {0: {dm.ZERO_DELTA: {ZERO_ALPHA: 1}}}
    for _ in range(pair_count):
        power = ext_delta_mul_pruned(power, pair_terms, max_delta, mask, len(target), p)
        if not power:
            return ()

    scale = mod_inv(factorial(pair_count), p)
    result = {
        delta: sparse_scale(poly, scale, p)
        for delta, poly in power.get(mask, {}).items()
    }
    return dm.sorted_delta_items(result)


@lru_cache(maxsize=None)
def gamma_hat(
    gamma_exp: Tuple[int, ...],
    target_delta: dm.DeltaKey,
    p: int,
) -> Tuple[Tuple[dm.DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Return the contracted gamma contribution up to `target_delta`."""

    out: dm.DeltaPoly = {}
    for mask, coeff in gamma_mask_expansion(gamma_exp, p):
        b_delta = {
            delta: dict(poly_items)
            for delta, poly_items in b_hat_mask(mask, target_delta, p)
        }
        out = dm.add(out, b_delta, p, scale=coeff)
    return dm.sorted_delta_items(out)
