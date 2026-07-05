"""Finite-field linear algebra for certificates.

For `c != 12`, the main question is simply whether the matrix has full row
rank over F_p.

For `c = 12`, the matrix has 44 source rows and rank 43.  The relation line is
therefore a one-dimensional left kernel.  This file contains the elementary
finite-field linear algebra used to extract one normalized left-kernel vector
and to verify that it annihilates the matrix columns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from .constants import DEFAULT_PRIME
from .matrix_mod import rank_mod_p


Matrix = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class LeftKernelResult:
    """A left-kernel vector extracted from one nonsingular square minor."""

    vector: Tuple[int, ...]
    selected_rows: Tuple[int, ...]
    selected_columns: Tuple[int, ...]
    omitted_row: int
    minor_determinant: int
    vector_sha256: str


def canonical_matrix(rows: Iterable[Sequence[int]], p: int = DEFAULT_PRIME) -> Matrix:
    """Return a rectangular row-major matrix with all entries reduced modulo p."""

    matrix = tuple(tuple(int(value) % p for value in row) for row in rows)
    if not matrix:
        return ()
    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError("matrix must be rectangular")
    return matrix


def matrix_dimensions(rows: Iterable[Sequence[int]], p: int = DEFAULT_PRIME) -> Tuple[int, int]:
    matrix = canonical_matrix(rows, p)
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def transpose(rows: Iterable[Sequence[int]], p: int = DEFAULT_PRIME) -> Matrix:
    matrix = canonical_matrix(rows, p)
    if not matrix:
        return ()
    return tuple(tuple(row[col] for row in matrix) for col in range(len(matrix[0])))


def vector_dot_mod_p(
    left: Sequence[int],
    right: Sequence[int],
    p: int = DEFAULT_PRIME,
) -> int:
    if len(left) != len(right):
        raise ValueError("dot product requires vectors of equal length")
    return sum((int(a) % p) * (int(b) % p) for a, b in zip(left, right)) % p


def vector_digest(vector: Sequence[int], p: int = DEFAULT_PRIME) -> str:
    digest = hashlib.sha256()
    digest.update(f"p={int(p)}\n".encode("ascii"))
    digest.update(f"len={len(vector)}\n".encode("ascii"))
    for value in vector:
        digest.update(f"{int(value) % p}\n".encode("ascii"))
    return digest.hexdigest()


def normalize_vector_mod_p(
    vector: Sequence[int],
    p: int = DEFAULT_PRIME,
    *,
    preferred_index: int | None = None,
) -> Tuple[int, ...]:
    """Scale a nonzero vector so a chosen nonzero coordinate becomes 1."""

    normalized = tuple(int(value) % p for value in vector)
    if preferred_index is not None:
        if preferred_index < 0 or preferred_index >= len(normalized):
            raise IndexError("preferred_index is outside the vector")
        if normalized[preferred_index]:
            pivot_index = preferred_index
        else:
            pivot_index = None
    else:
        pivot_index = None

    if pivot_index is None:
        for idx, value in enumerate(normalized):
            if value:
                pivot_index = idx
                break
    if pivot_index is None:
        raise ValueError("cannot normalize the zero vector")

    scale = pow(normalized[pivot_index], -1, p)
    return tuple((value * scale) % p for value in normalized)


def determinant_mod_p(rows: Iterable[Sequence[int]], p: int = DEFAULT_PRIME) -> int:
    """Return the determinant of a square matrix over F_p."""

    matrix = [list(row) for row in canonical_matrix(rows, p)]
    n = len(matrix)
    if n == 0:
        return 1
    if any(len(row) != n for row in matrix):
        raise ValueError("determinant requires a square matrix")

    det = 1
    for col in range(n):
        pivot = None
        for row_idx in range(col, n):
            if matrix[row_idx][col] % p:
                pivot = row_idx
                break
        if pivot is None:
            return 0
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            det = (-det) % p

        pivot_value = matrix[col][col] % p
        det = (det * pivot_value) % p
        inv_pivot = pow(pivot_value, -1, p)
        for row_idx in range(col + 1, n):
            factor = (matrix[row_idx][col] * inv_pivot) % p
            if not factor:
                continue
            for update_col in range(col, n):
                matrix[row_idx][update_col] = (
                    matrix[row_idx][update_col] - factor * matrix[col][update_col]
                ) % p
    return det % p


def solve_square_mod_p(
    rows: Iterable[Sequence[int]],
    rhs: Sequence[int],
    p: int = DEFAULT_PRIME,
) -> Tuple[int, ...]:
    """Solve A x = rhs over F_p for square nonsingular A."""

    matrix = [list(row) for row in canonical_matrix(rows, p)]
    n = len(matrix)
    if n == 0:
        if rhs:
            raise ValueError("empty matrix can only solve an empty right-hand side")
        return ()
    if any(len(row) != n for row in matrix):
        raise ValueError("solve_square_mod_p requires a square matrix")
    if len(rhs) != n:
        raise ValueError("right-hand side length must equal matrix size")

    augmented = [
        row + [int(rhs_value) % p]
        for row, rhs_value in zip(matrix, rhs)
    ]
    for col in range(n):
        pivot = None
        for row_idx in range(col, n):
            if augmented[row_idx][col] % p:
                pivot = row_idx
                break
        if pivot is None:
            raise ValueError("matrix is singular")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        inv_pivot = pow(augmented[col][col], -1, p)
        augmented[col] = [(value * inv_pivot) % p for value in augmented[col]]
        for row_idx in range(n):
            if row_idx == col:
                continue
            factor = augmented[row_idx][col] % p
            if not factor:
                continue
            augmented[row_idx] = [
                (value - factor * pivot_value) % p
                for value, pivot_value in zip(augmented[row_idx], augmented[col])
            ]
    return tuple(row[-1] % p for row in augmented)


def submatrix(
    rows: Iterable[Sequence[int]],
    row_indices: Sequence[int],
    column_indices: Sequence[int],
    p: int = DEFAULT_PRIME,
) -> Matrix:
    matrix = canonical_matrix(rows, p)
    return tuple(
        tuple(matrix[row_idx][col_idx] for col_idx in column_indices)
        for row_idx in row_indices
    )


def matrix_rank_with_selected_columns(
    rows: Iterable[Sequence[int]],
    column_indices: Sequence[int],
    p: int = DEFAULT_PRIME,
) -> int:
    matrix = canonical_matrix(rows, p)
    return rank_mod_p(
        (
            tuple(row[col_idx] for col_idx in column_indices)
            for row in matrix
        ),
        p,
    )


def select_independent_columns(
    rows: Iterable[Sequence[int]],
    p: int = DEFAULT_PRIME,
    *,
    target_rank: int | None = None,
) -> Tuple[int, ...]:
    """Greedily select columns whose span has the requested rank.

    This is not a randomized certificate.  It scans columns in order and keeps
    a column exactly when it raises the rank.
    """

    matrix = canonical_matrix(rows, p)
    if not matrix:
        if target_rank not in (None, 0):
            raise ValueError("empty matrix cannot reach a positive target rank")
        return ()
    width = len(matrix[0])
    selected = []
    current_rank = 0
    for col_idx in range(width):
        trial = selected + [col_idx]
        trial_rank = matrix_rank_with_selected_columns(matrix, trial, p)
        if trial_rank > current_rank:
            selected.append(col_idx)
            current_rank = trial_rank
            if target_rank is not None and current_rank == target_rank:
                return tuple(selected)
    if target_rank is not None and current_rank != target_rank:
        raise ValueError(f"could only find rank {current_rank}, not target rank {target_rank}")
    return tuple(selected)


def select_independent_rows(
    rows: Iterable[Sequence[int]],
    p: int = DEFAULT_PRIME,
    *,
    target_rank: int | None = None,
) -> Tuple[int, ...]:
    """Greedily select source rows whose span has the requested rank."""

    matrix = canonical_matrix(rows, p)
    selected = []
    current_rank = 0
    for row_idx in range(len(matrix)):
        trial_indices = selected + [row_idx]
        trial_rank = rank_mod_p((matrix[idx] for idx in trial_indices), p)
        if trial_rank > current_rank:
            selected.append(row_idx)
            current_rank = trial_rank
            if target_rank is not None and current_rank == target_rank:
                return tuple(selected)
    if target_rank is not None and current_rank != target_rank:
        raise ValueError(f"could only find rank {current_rank}, not target rank {target_rank}")
    return tuple(selected)


def column_dot_values(
    rows: Iterable[Sequence[int]],
    vector: Sequence[int],
    p: int = DEFAULT_PRIME,
) -> Tuple[int, ...]:
    """Return λ^T M column by column for row vector λ and matrix M."""

    matrix = canonical_matrix(rows, p)
    if not matrix:
        return ()
    if len(vector) != len(matrix):
        raise ValueError("left vector length must equal matrix row count")
    return tuple(
        vector_dot_mod_p(vector, tuple(row[col_idx] for row in matrix), p)
        for col_idx in range(len(matrix[0]))
    )


def left_kernel_from_minor(
    rows: Iterable[Sequence[int]],
    selected_rows: Sequence[int],
    selected_columns: Sequence[int],
    p: int = DEFAULT_PRIME,
    *,
    omitted_row: int | None = None,
    normalization_index: int | None = None,
) -> LeftKernelResult:
    """Build a left-kernel vector from one nonsingular rank-(m-1) minor.

    The selected square minor uses all rows except `omitted_row`. We set the
    omitted coordinate to 1 and solve A^T lambda = -b, where A is the selected minor
    and b is the omitted row restricted to the selected columns.
    """

    matrix = canonical_matrix(rows, p)
    row_count = len(matrix)
    selected_rows_tuple = tuple(int(row) for row in selected_rows)
    selected_columns_tuple = tuple(int(col) for col in selected_columns)
    if len(selected_rows_tuple) != len(selected_columns_tuple):
        raise ValueError("selected minor must be square")
    if row_count != len(selected_rows_tuple) + 1:
        raise ValueError("left_kernel_from_minor expects exactly one omitted row")
    if omitted_row is None:
        omitted = tuple(sorted(set(range(row_count)) - set(selected_rows_tuple)))
        if len(omitted) != 1:
            raise ValueError("could not infer a unique omitted row")
        omitted_row = omitted[0]
    omitted_row = int(omitted_row)
    if omitted_row in selected_rows_tuple:
        raise ValueError("omitted_row must not be one of selected_rows")

    minor = submatrix(matrix, selected_rows_tuple, selected_columns_tuple, p)
    minor_det = determinant_mod_p(minor, p)
    if minor_det == 0:
        raise ValueError("selected minor is singular")

    omitted_values = tuple((-matrix[omitted_row][col_idx]) % p for col_idx in selected_columns_tuple)
    selected_solution = solve_square_mod_p(transpose(minor, p), omitted_values, p)
    vector = [0] * row_count
    for row_idx, coefficient in zip(selected_rows_tuple, selected_solution):
        vector[row_idx] = coefficient
    vector[omitted_row] = 1
    vector_tuple = normalize_vector_mod_p(
        vector,
        p,
        preferred_index=omitted_row if normalization_index is None else normalization_index,
    )

    selected_dots = column_dot_values(
        submatrix(matrix, range(row_count), selected_columns_tuple, p),
        vector_tuple,
        p,
    )
    if any(selected_dots):
        raise ValueError("constructed vector does not annihilate the selected columns")

    return LeftKernelResult(
        vector=vector_tuple,
        selected_rows=selected_rows_tuple,
        selected_columns=selected_columns_tuple,
        omitted_row=omitted_row,
        minor_determinant=minor_det,
        vector_sha256=vector_digest(vector_tuple, p),
    )
