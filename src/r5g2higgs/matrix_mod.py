"""Pairing-matrix assembly.

This file is deliberately less mathematical than `pairing_mod.py`.

The bases are already chosen by `basis_mod.py`, and one pairing value is
already implemented by `pairing_mod.py`.  The job here is to assemble the
finite matrices:

    row    = source basis element in degree 22 and Chern degree c
    column = target basis element in degree 26
    entry  = pairing(row * column)

The matrix is never conceptually different from the pairing formula.  It is
just the tabulation of that formula in the chosen bases.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from .basis_mod import add_exp, independent_basis_by_chern, independent_invariant_basis
from .constants import DEFAULT_PRIME, SOURCE_DEGREE, W_DEGREE
from .model import InvariantExp
from .pairing_mod import pairing_total_mod_from_prepared


@dataclass(frozen=True)
class PairingMatrixShape:
    """Dimensions of one Chern-degree pairing matrix."""

    chern_degree: int
    source_dimension: int
    w_dimension: int


@lru_cache(maxsize=None)
def source_basis(
    chern_degree: int,
    source_degree: int = SOURCE_DEGREE,
):
    """Return the degree-22 source basis for one Chern degree."""

    basis_by_chern, _raw_counts, _meta = independent_basis_by_chern(
        5,
        int(source_degree),
        (int(chern_degree),),
    )
    return basis_by_chern.get(int(chern_degree), ())


@lru_cache(maxsize=None)
def target_basis(w_degree: int = W_DEGREE):
    """Return the target basis in ordinary degree `w_degree`, here 26."""

    basis, _meta = independent_invariant_basis(5, int(w_degree))
    return basis


@contextmanager
def selected_residue_backend(backend: str | None):
    """Temporarily select the residue backend used by pairing_mod."""

    if backend is None:
        yield
        return

    prior_backend = os.environ.get("R5G2HIGGS_RESIDUE_BACKEND")
    os.environ["R5G2HIGGS_RESIDUE_BACKEND"] = backend
    try:
        yield
    finally:
        if prior_backend is None:
            os.environ.pop("R5G2HIGGS_RESIDUE_BACKEND", None)
        else:
            os.environ["R5G2HIGGS_RESIDUE_BACKEND"] = prior_backend


def matrix_shape(
    chern_degree: int,
    *,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> PairingMatrixShape:
    source = source_basis(chern_degree, source_degree)
    target = target_basis(w_degree)
    return PairingMatrixShape(
        chern_degree=int(chern_degree),
        source_dimension=len(source),
        w_dimension=len(target),
    )


def matrix_basis_names(
    chern_degree: int,
    *,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return `(source_basis_names, w_basis_names)` for a matrix."""

    source = source_basis(chern_degree, source_degree)
    target = target_basis(w_degree)
    return (
        tuple(item.name for item in source),
        tuple(item.name for item in target),
    )


def entry_total_exponent(
    chern_degree: int,
    row_index: int,
    w_index: int,
    *,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> InvariantExp:
    """Return the invariant exponent for row * column.

    Pairing matrix entries are pairings of products.  Since basis elements are
    stored by exponent vectors, multiplying row and column means adding their
    exponent vectors.
    """

    source = source_basis(chern_degree, source_degree)
    target = target_basis(w_degree)
    return add_exp(source[int(row_index)].exp, target[int(w_index)].exp)


def basis_total_grid(
    chern_degree: int,
    row_indices: Tuple[int, ...],
    w_indices: Tuple[int, ...],
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> List[Tuple[int, int, InvariantExp]]:
    """Return `(row_index, w_index, total_exponent)` over a rectangular block."""

    source = source_basis(chern_degree, source_degree)
    target = target_basis(w_degree)
    out: List[Tuple[int, int, InvariantExp]] = []
    for row_index in row_indices:
        source_exp = source[int(row_index)].exp
        for w_index in w_indices:
            out.append((int(row_index), int(w_index), add_exp(source_exp, target[int(w_index)].exp)))
    return out


def pairing_entry(
    chern_degree: int,
    row_index: int,
    w_index: int,
    *,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> int:
    """Compute one pairing-matrix entry."""

    total = entry_total_exponent(
        chern_degree,
        row_index,
        w_index,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    with selected_residue_backend(backend):
        return int(pairing_total_mod_from_prepared(total, p))


def all_row_indices(shape: PairingMatrixShape) -> Tuple[int, ...]:
    return tuple(range(shape.source_dimension))


def selected_row_indices(
    chern_degree: int,
    row_indices: Sequence[int] | None,
    *,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[int, ...]:
    shape = matrix_shape(chern_degree, source_degree=source_degree, w_degree=w_degree)
    rows = all_row_indices(shape) if row_indices is None else tuple(int(i) for i in row_indices)
    if len(set(rows)) != len(rows):
        raise ValueError("row_indices contains duplicates")
    for row_index in rows:
        if row_index < 0 or row_index >= shape.source_dimension:
            raise IndexError(
                f"row index {row_index} outside source dimension {shape.source_dimension}"
            )
    return rows


def selected_w_indices(
    chern_degree: int,
    w_indices: Sequence[int],
    *,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[int, ...]:
    shape = matrix_shape(chern_degree, source_degree=source_degree, w_degree=w_degree)
    cols = tuple(int(i) for i in w_indices)
    if len(set(cols)) != len(cols):
        raise ValueError("w_indices contains duplicates")
    for w_index in cols:
        if w_index < 0 or w_index >= shape.w_dimension:
            raise IndexError(f"W index {w_index} outside W dimension {shape.w_dimension}")
    return cols


def column_total_exponents(
    chern_degree: int,
    w_index: int,
    *,
    row_indices: Sequence[int] | None = None,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[Tuple[int, InvariantExp], ...]:
    """Return `(row_index, total_exponent)` data for one column."""

    rows = selected_row_indices(
        chern_degree,
        row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    cols = selected_w_indices(
        chern_degree,
        (w_index,),
        source_degree=source_degree,
        w_degree=w_degree,
    )
    return tuple(
        (row_index, total)
        for row_index, _w_index, total in basis_total_grid(
            chern_degree,
            rows,
            cols,
            source_degree,
            w_degree,
        )
    )


def pairing_column(
    chern_degree: int,
    w_index: int,
    *,
    row_indices: Sequence[int] | None = None,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[int, ...]:
    """Compute one W-column of a Chern-degree pairing matrix."""

    entries = column_total_exponents(
        chern_degree,
        w_index,
        row_indices=row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    with selected_residue_backend(backend):
        return tuple(
            int(pairing_total_mod_from_prepared(total, p))
            for _row_index, total in entries
        )


def pairing_columns(
    chern_degree: int,
    w_indices: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[Tuple[int, ...], ...]:
    """Compute columns and return them as a tuple of column vectors."""

    rows = selected_row_indices(
        chern_degree,
        row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    cols = selected_w_indices(
        chern_degree,
        w_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    grid = basis_total_grid(chern_degree, rows, cols, source_degree, w_degree)
    by_col = {w_index: [] for w_index in cols}
    with selected_residue_backend(backend):
        for _row_index, w_index, total in grid:
            by_col[w_index].append(int(pairing_total_mod_from_prepared(total, p)))
    return tuple(tuple(by_col[w_index]) for w_index in cols)


def pairing_column_with_rows(
    chern_degree: int,
    w_index: int,
    *,
    row_indices: Sequence[int] | None = None,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[Tuple[int, int], ...]:
    """Compute one column as `(row_index, value)` pairs."""

    rows = selected_row_indices(
        chern_degree,
        row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    values = pairing_column(
        chern_degree,
        w_index,
        row_indices=rows,
        p=p,
        backend=backend,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    return tuple(zip(rows, values))


def pairing_submatrix(
    chern_degree: int,
    row_indices: Sequence[int],
    w_indices: Sequence[int],
    *,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[Tuple[int, ...], ...]:
    """Compute a row-major submatrix."""

    rows = selected_row_indices(
        chern_degree,
        row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    cols = selected_w_indices(
        chern_degree,
        w_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    grid = basis_total_grid(chern_degree, rows, cols, source_degree, w_degree)
    values = {}
    with selected_residue_backend(backend):
        for row_index, w_index, total in grid:
            values[(row_index, w_index)] = int(pairing_total_mod_from_prepared(total, p))
    return tuple(
        tuple(values[(row_index, w_index)] for w_index in cols)
        for row_index in rows
    )


def rank_mod_p(rows: Iterable[Sequence[int]], p: int = DEFAULT_PRIME) -> int:
    """Return the rank of a finite matrix over F_p."""

    matrix: List[List[int]] = [
        [int(value) % p for value in row]
        for row in rows
    ]
    if not matrix:
        return 0
    width = max((len(row) for row in matrix), default=0)
    if width == 0:
        return 0
    for row in matrix:
        if len(row) != width:
            raise ValueError("rank_mod_p requires a rectangular matrix")

    rank = 0
    for col in range(width):
        pivot = None
        for row_idx in range(rank, len(matrix)):
            if matrix[row_idx][col] % p:
                pivot = row_idx
                break
        if pivot is None:
            continue

        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        inv = pow(matrix[rank][col], -1, p)
        matrix[rank] = [(value * inv) % p for value in matrix[rank]]

        for row_idx in range(len(matrix)):
            if row_idx == rank:
                continue
            factor = matrix[row_idx][col] % p
            if not factor:
                continue
            matrix[row_idx] = [
                (value - factor * pivot_value) % p
                for value, pivot_value in zip(matrix[row_idx], matrix[rank])
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank
