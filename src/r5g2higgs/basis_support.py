"""Small exterior-algebra helpers used by basis enumeration."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .constants import B_INDEX, B_LABELS, GAMMA_LABELS
from .gamma_mod import wedge_masks
from .reference_sympy import gamma_product_to_b_terms

BLabel = Tuple[int, int]


def mask_for_b_label(label: BLabel) -> int:
    return 1 << B_INDEX[label]


__all__ = [
    "BLabel",
    "B_LABELS",
    "GAMMA_LABELS",
    "gamma_product_to_b_terms",
    "mask_for_b_label",
    "wedge_masks",
]
