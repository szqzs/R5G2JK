"""Fixed conventions for the rank-5 genus-2 computation."""

from __future__ import annotations

RANK = 5
GENUS = 2
SOURCE_DEGREE = 22
W_DEGREE = 26
TOP_DEGREE = SOURCE_DEGREE + W_DEGREE

DEFAULT_PRIME = 2305843009213693951
COLLAPSED_CENTRAL_PREFACTOR = 5

GENERATOR_LABELS = (2, 3, 4, 5)
B_LABELS = tuple((r, j) for r in range(2, 6) for j in range(1, 5))
B_INDEX = {label: idx for idx, label in enumerate(B_LABELS)}
GAMMA_LABELS = tuple((r, s) for r in range(2, 6) for s in range(r, 6))
GAMMA_INDEX = {label: idx for idx, label in enumerate(GAMMA_LABELS)}
