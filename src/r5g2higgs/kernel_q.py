"""Exact-Q even-kernel delta expansions for the JK evaluator."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from math import factorial
from typing import List, Sequence, Tuple

from . import delta_q as dq
from .constants import GENUS
from .delta_mod import DELTA_UNITS, ZERO_DELTA, ZERO_DERIV, DeltaKey, DerivOrders, delta_add, delta_leq, delta_sub
from .sparse_mod import Alpha, ZERO_ALPHA
from .sparse_q import SparseQPoly, add as sparse_add, mul as sparse_mul, scale as sparse_scale, sorted_items
from .tau_q import b_perturbation_q, c_direction_term_q, tau_gradient_q, tau_hessian_q


def constant_sparse_value_q(poly_items: Tuple[Tuple[Alpha, Fraction], ...]) -> Fraction:
    poly = dict(poly_items)
    if any(alpha != ZERO_ALPHA and coeff for alpha, coeff in poly.items()):
        raise ValueError("expected a constant sparse polynomial")
    return Fraction(poly.get(ZERO_ALPHA, 0))


def invert_const_matrix_q(matrix: Sequence[Sequence[Fraction]]) -> Tuple[Tuple[Fraction, ...], ...]:
    size = len(matrix)
    aug = [
        [Fraction(matrix[row][col]) for col in range(size)]
        + [Fraction(1 if row == col else 0) for col in range(size)]
        for row in range(size)
    ]
    for col in range(size):
        pivot = None
        for row in range(col, size):
            if aug[row][col]:
                pivot = row
                break
        if pivot is None:
            raise ZeroDivisionError("matrix is singular over Q")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = Fraction(1, 1) / aug[col][col]
        aug[col] = [value * inv for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            coeff = aug[row][col]
            if coeff:
                aug[row] = [aug[row][idx] - coeff * aug[col][idx] for idx in range(2 * size)]
    return tuple(tuple(row[size:]) for row in aug)


@lru_cache(maxsize=None)
def hessian_tau2_inverse_const_q() -> Tuple[Tuple[Fraction, ...], ...]:
    """Return the inverse of Hessian(tau2) over Q."""

    matrix = [
        [constant_sparse_value_q(tau_hessian_q(2)[row][col]) for col in range(4)]
        for row in range(4)
    ]
    return invert_const_matrix_q(matrix)


def sparse_matrix_left_const_mul_q(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Tuple[Tuple[Alpha, Fraction], ...]]],
) -> Tuple[Tuple[Tuple[Tuple[Alpha, Fraction], ...], ...], ...]:
    rows = []
    for row in range(4):
        out_row = []
        for col in range(4):
            accum: SparseQPoly = {}
            for inner in range(4):
                if left[row][inner]:
                    accum = sparse_add(accum, dict(right[inner][col]), scale=left[row][inner])
            out_row.append(sorted_items(accum))
        rows.append(tuple(out_row))
    return tuple(rows)


@lru_cache(maxsize=None)
def hessian_perturbation_q() -> Tuple[Tuple[Tuple[Tuple[Tuple[Alpha, Fraction], ...], ...], ...], ...]:
    """Return Hessian(tau2)^(-1) Hessian(tau_r) for r=3,4,5 over Q."""

    h0_inv = hessian_tau2_inverse_const_q()
    return tuple(sparse_matrix_left_const_mul_q(h0_inv, tau_hessian_q(r)) for r in (3, 4, 5))


@lru_cache(maxsize=None)
def hessian_inverse_delta_q(
    max_delta: DeltaKey,
) -> Tuple[Tuple[Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, Fraction], ...]], ...], ...], ...]:
    """Return the truncated delta expansion of Hessian(q)^(-1) over Q."""

    perturbation = hessian_perturbation_q()
    a_matrix: List[List[dq.DeltaQPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            entry: dq.DeltaQPoly = {}
            for unit, matrix in zip(DELTA_UNITS, perturbation):
                if delta_leq(unit, max_delta):
                    poly = dict(matrix[row][col])
                    if poly:
                        entry[unit] = sparse_add(entry.get(unit, {}), poly)
            a_matrix[row][col] = entry

    series = dq.matrix_identity(4)
    power = dq.matrix_identity(4)
    for order in range(1, sum(max_delta) + 1):
        power = dq.matrix_mul(power, a_matrix, max_delta)
        sign = -1 if order % 2 else 1
        series = tuple(
            tuple(dq.add(series[row][col], power[row][col], scale=sign) for col in range(4))
            for row in range(4)
        )

    h0_inv = hessian_tau2_inverse_const_q()
    out: List[List[dq.DeltaQPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            accum: dq.DeltaQPoly = {}
            for inner in range(4):
                if h0_inv[inner][col]:
                    accum = dq.add(accum, dq.scale(series[row][inner], h0_inv[inner][col]))
            out[row][col] = accum

    return tuple(
        tuple(dq.sorted_delta_items(cell) for cell in row)
        for row in out
    )


def hessian_inverse_cell_q(max_delta: DeltaKey, row: int, col: int) -> dq.DeltaQPoly:
    return {delta: dict(poly_items) for delta, poly_items in hessian_inverse_delta_q(max_delta)[row][col]}


@lru_cache(maxsize=None)
def hat_pair_delta_q(
    r: int,
    s: int,
    max_delta: DeltaKey,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Return -grad(tau_r)^T Hessian(q)^(-1) grad(tau_s) over Q."""

    grad_r = tau_gradient_q(r)
    grad_s = tau_gradient_q(s)
    accum: dq.DeltaQPoly = {}
    for row in range(4):
        for col in range(4):
            cell = hessian_inverse_cell_q(max_delta, row, col)
            if not cell:
                continue
            coeff_poly = sparse_mul(dict(grad_r[row]), dict(grad_s[col]))
            coeff_poly = sparse_scale(coeff_poly, -1)
            for delta, poly in cell.items():
                product = sparse_mul(coeff_poly, poly)
                if product:
                    accum[delta] = sparse_add(accum.get(delta, {}), product)
    return dq.sorted_delta_items(accum)


@lru_cache(maxsize=None)
def det_ratio_delta_power_q(
    max_delta: DeltaKey,
    power: int,
) -> Tuple[Tuple[DeltaKey, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Return det(Hessian(q))/det(Hessian(tau2)), raised to `power`, over Q."""

    perturbation = hessian_perturbation_q()
    matrix: List[List[dq.DeltaQPoly]] = []
    for row in range(4):
        out_row: List[dq.DeltaQPoly] = []
        for col in range(4):
            entry: dq.DeltaQPoly = {ZERO_DELTA: {ZERO_ALPHA: Fraction(1)}} if row == col else {}
            for unit, perturb_matrix in zip(DELTA_UNITS, perturbation):
                if delta_leq(unit, max_delta):
                    poly = dict(perturb_matrix[row][col])
                    if poly:
                        entry[unit] = sparse_add(entry.get(unit, {}), poly)
            out_row.append(entry)
        matrix.append(out_row)

    det_poly: dq.DeltaQPoly = {}
    for perm in permutations(range(4)):
        inversions = sum(1 for left in range(4) for right in range(left + 1, 4) if perm[left] > perm[right])
        term: dq.DeltaQPoly = {ZERO_DELTA: {ZERO_ALPHA: Fraction(-1 if inversions % 2 else 1)}}
        for row, col in enumerate(perm):
            term = dq.mul(term, matrix[row][col], max_delta)
            if not term:
                break
        det_poly = dq.add(det_poly, term)
    return dq.sorted_delta_items(dq.pow_delta(det_poly, power, max_delta))


def denominator_taylor_terms_q(max_delta: DeltaKey) -> dq.KernelQTerms:
    """Expand the four denominator perturbation factors over Q."""

    if max_delta == ZERO_DELTA:
        return {(ZERO_DELTA, ZERO_DERIV): {ZERO_ALPHA: Fraction(1)}}
    terms: dq.KernelQTerms = {(ZERO_DELTA, ZERO_DERIV): {ZERO_ALPHA: Fraction(1)}}
    max_order = sum(max_delta)
    unit_to_rank = {(1, 0, 0): 3, (0, 1, 0): 4, (0, 0, 1): 5}
    for j in range(1, 5):
        eps: dq.DeltaQPoly = {
            unit: dict(b_perturbation_q(rank, j))
            for unit, rank in unit_to_rank.items()
            if delta_leq(unit, max_delta)
        }
        factor: dq.KernelQTerms = {}
        for order in range(max_order + 1):
            eps_power = dq.pow_delta(eps, order, max_delta)
            if not eps_power:
                continue
            deriv = [0, 0, 0, 0]
            deriv[j - 1] = order
            deriv_tuple = tuple(deriv)  # type: ignore[assignment]
            scale = Fraction(1, factorial(order))
            for delta, poly in eps_power.items():
                if poly:
                    factor[(delta, deriv_tuple)] = sparse_add(
                        factor.get((delta, deriv_tuple), {}),
                        poly,
                        scale=scale,
                    )

        next_terms: dq.KernelQTerms = {}
        for (d1, der1), poly1 in terms.items():
            for (d2, der2), poly2 in factor.items():
                next_delta = delta_add(d1, d2)
                if not delta_leq(next_delta, max_delta):
                    continue
                next_deriv = tuple(der1[idx] + der2[idx] for idx in range(4))  # type: ignore[assignment]
                product = sparse_mul(poly1, poly2)
                if product:
                    next_terms[(next_delta, next_deriv)] = sparse_add(
                        next_terms.get((next_delta, next_deriv), {}),
                        product,
                    )
        terms = {key: value for key, value in next_terms.items() if value}
    return terms


@lru_cache(maxsize=None)
def even_kernel_terms_q(
    target_delta: DeltaKey,
) -> Tuple[Tuple[DeltaKey, DerivOrders, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Return the exact even JK kernel terms up to the requested delta degree."""

    linear = {
        (1, 0, 0): dict(c_direction_term_q(3)),
        (0, 1, 0): dict(c_direction_term_q(4)),
        (0, 0, 1): dict(c_direction_term_q(5)),
    }
    exp_delta = dq.exp_linear(linear, target_delta)
    det_delta = {delta: dict(poly_items) for delta, poly_items in det_ratio_delta_power_q(target_delta, GENUS)}
    terms = denominator_taylor_terms_q(target_delta)
    terms = dq.kernel_terms_mul_delta(terms, exp_delta, target_delta)
    terms = dq.kernel_terms_mul_delta(terms, det_delta, target_delta)
    return tuple(
        (delta, deriv, sorted_items(poly))
        for (delta, deriv), poly in sorted(terms.items())
        if poly
    )
