"""Typed exponent data model.

This layer intentionally contains no JK formula implementation. It only defines
the small immutable objects that later formula and matrix layers will share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .constants import GAMMA_LABELS

FourTuple = Tuple[int, int, int, int]
GammaTuple = Tuple[int, ...]


def _as_four_tuple(values: Tuple[int, ...], field_name: str) -> FourTuple:
    if len(values) != 4:
        raise ValueError(f"{field_name} must have length 4, got {len(values)}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _as_gamma_tuple(values: Tuple[int, ...]) -> GammaTuple:
    if len(values) != len(GAMMA_LABELS):
        raise ValueError(f"gamma must have length {len(GAMMA_LABELS)}, got {len(values)}")
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class InvariantExp:
    """Exponents of a monomial in a, f, and gamma generators.

    The a tuple means powers of (a2, a3, a4, a5).
    The f tuple means powers of (f2, f3, f4, f5).
    The gamma tuple follows constants.GAMMA_LABELS.
    """

    a: FourTuple = (0, 0, 0, 0)
    f: FourTuple = (0, 0, 0, 0)
    gamma: GammaTuple = (0,) * len(GAMMA_LABELS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _as_four_tuple(tuple(self.a), "a"))
        object.__setattr__(self, "f", _as_four_tuple(tuple(self.f), "f"))
        object.__setattr__(self, "gamma", _as_gamma_tuple(tuple(self.gamma)))
        if any(value < 0 for value in (*self.a, *self.f, *self.gamma)):
            raise ValueError("InvariantExp exponents must be nonnegative")

    @property
    def target_delta(self) -> Tuple[int, int, int]:
        """Delta degree selected by f3, f4, f5 powers."""

        return (self.f[1], self.f[2], self.f[3])


def zero_invariant_exp() -> InvariantExp:
    return InvariantExp()


def add_invariant_exp(left: InvariantExp, right: InvariantExp) -> InvariantExp:
    return InvariantExp(
        tuple(left.a[idx] + right.a[idx] for idx in range(4)),  # type: ignore[arg-type]
        tuple(left.f[idx] + right.f[idx] for idx in range(4)),  # type: ignore[arg-type]
        tuple(left.gamma[idx] + right.gamma[idx] for idx in range(len(GAMMA_LABELS))),
    )

