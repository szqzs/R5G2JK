"""Even-kernel delta expansions for the modular JK evaluator.

This file implements the even part of the Jeffrey-Kirwan integrand after the
specialization to rank 5, determinant degree 1, and genus 2.

The object expanded here is built from:

    q = tau2 + delta3 tau3 + delta4 tau4 + delta5 tau5,

the Hessian of q, the determinant factor, the exponential factor coming from
dq(c-tilde), and the four denominator factors involving B_j.

The delta variables are bookkeeping variables.  They let us compute all
f3/f4/f5 insertions by coefficient extraction.  Every expansion in this file is
truncated at `target_delta`; this is exact for the requested coefficient and is
not a numerical approximation.

Output shape
------------
The main function `even_kernel_terms(target_delta, p)` returns terms

    (delta exponent, Y-derivative orders, Y-polynomial coefficient).

The derivative orders tell the residue layer which Y-derivatives came from the
Taylor expansion of the denominator factors.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import List, Sequence, Tuple

from . import delta_mod as dm
from .arithmetic import mod_factorial
from .constants import GENUS
from .sparse_mod import Alpha, SparsePoly, ZERO_ALPHA, add as sparse_add, mod_inv, mul as sparse_mul, scale as sparse_scale, sorted_items
from .tau_mod import b_perturbation, c_direction_term, rank5_tau, tau_gradient, tau_hessian


def constant_sparse_value(poly_items: Tuple[Tuple[Alpha, int], ...], p: int) -> int:
    poly = dict(poly_items)
    if any(alpha != ZERO_ALPHA and coeff % p for alpha, coeff in poly.items()):
        raise ValueError("expected a constant sparse polynomial")
    return poly.get(ZERO_ALPHA, 0) % p


def invert_const_matrix(matrix: Sequence[Sequence[int]], p: int) -> Tuple[Tuple[int, ...], ...]:
    size = len(matrix)
    aug = [
        [int(matrix[row][col]) % p for col in range(size)]
        + [1 if row == col else 0 for col in range(size)]
        for row in range(size)
    ]
    for col in range(size):
        pivot = None
        for row in range(col, size):
            if aug[row][col] % p:
                pivot = row
                break
        if pivot is None:
            raise ZeroDivisionError("matrix is singular modulo p")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = mod_inv(aug[col][col], p)
        aug[col] = [value * inv % p for value in aug[col]]
        for row in range(size):
            if row == col:
                continue
            coeff = aug[row][col] % p
            if coeff:
                aug[row] = [(aug[row][idx] - coeff * aug[col][idx]) % p for idx in range(2 * size)]
    return tuple(tuple(row[size:]) for row in aug)


@lru_cache(maxsize=None)
def hessian_tau2_inverse_const(p: int) -> Tuple[Tuple[int, ...], ...]:
    """Return the inverse of Hessian(tau2).

    Hessian(tau2) is constant in the Y coordinates.  This constant inverse is
    the base matrix used to expand Hessian(q)^(-1).
    """

    matrix = [
        [constant_sparse_value(tau_hessian(2, p)[row][col], p) for col in range(4)]
        for row in range(4)
    ]
    return invert_const_matrix(matrix, p)


def sparse_matrix_left_const_mul(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[Tuple[Tuple[Alpha, int], ...]]],
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...]:
    rows = []
    for row in range(4):
        out_row = []
        for col in range(4):
            accum: SparsePoly = {}
            for inner in range(4):
                if left[row][inner] % p:
                    accum = sparse_add(accum, dict(right[inner][col]), p, scale=left[row][inner])
            out_row.append(sorted_items(accum))
        rows.append(tuple(out_row))
    return tuple(rows)


@lru_cache(maxsize=None)
def hessian_perturbation(p: int) -> Tuple[Tuple[Tuple[Tuple[Tuple[Alpha, int], ...], ...], ...], ...]:
    """Return Hessian(tau2)^(-1) Hessian(tau_r) for r=3,4,5."""

    h0_inv = hessian_tau2_inverse_const(p)
    return tuple(sparse_matrix_left_const_mul(h0_inv, tau_hessian(r, p), p) for r in (3, 4, 5))


@lru_cache(maxsize=None)
def hessian_inverse_delta(
    max_delta: dm.DeltaKey,
    p: int,
) -> Tuple[Tuple[Tuple[Tuple[dm.DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...], ...], ...]:
    """Return the truncated delta expansion of Hessian(q)^(-1).

    Write

        Hessian(q) = H0 * (I + A),

    where H0 = Hessian(tau2) and A is linear in delta3, delta4, delta5.
    Then

        Hessian(q)^(-1) = (I - A + A^2 - A^3 + ...) * H0^(-1).

    The series is finite after truncation to `max_delta`.
    """

    perturbation = hessian_perturbation(p)
    a_matrix: List[List[dm.DeltaPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            entry: dm.DeltaPoly = {}
            for unit, matrix in zip(dm.DELTA_UNITS, perturbation):
                if dm.delta_leq(unit, max_delta):
                    poly = dict(matrix[row][col])
                    if poly:
                        entry[unit] = sparse_add(entry.get(unit, {}), poly, p)
            a_matrix[row][col] = entry

    series = dm.matrix_identity(4)
    power = dm.matrix_identity(4)
    for order in range(1, sum(max_delta) + 1):
        power = dm.matrix_mul(power, a_matrix, max_delta, p)
        sign = -1 if order % 2 else 1
        series = tuple(
            tuple(dm.add(series[row][col], power[row][col], p, scale=sign) for col in range(4))
            for row in range(4)
        )

    h0_inv = hessian_tau2_inverse_const(p)
    out: List[List[dm.DeltaPoly]] = [[{} for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            accum: dm.DeltaPoly = {}
            for inner in range(4):
                if h0_inv[inner][col] % p:
                    accum = dm.add(accum, dm.scale(series[row][inner], h0_inv[inner][col], p), p)
            out[row][col] = accum

    return tuple(
        tuple(dm.sorted_delta_items(cell) for cell in row)
        for row in out
    )


def hessian_inverse_cell(max_delta: dm.DeltaKey, row: int, col: int, p: int) -> dm.DeltaPoly:
    return {delta: dict(poly_items) for delta, poly_items in hessian_inverse_delta(max_delta, p)[row][col]}


@lru_cache(maxsize=None)
def hat_pair_delta(
    r: int,
    s: int,
    max_delta: dm.DeltaKey,
    p: int,
) -> Tuple[Tuple[dm.DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Return the contraction coefficient for a pair of odd insertions.

    A pair with ranks r and s contributes

        - grad(tau_r)^T Hessian(q)^(-1) grad(tau_s).

    This is the algebraic replacement for the corresponding exterior integral
    in the JK formula.
    """

    grad_r = tau_gradient(r, p)
    grad_s = tau_gradient(s, p)
    accum: dm.DeltaPoly = {}
    for row in range(4):
        for col in range(4):
            cell = hessian_inverse_cell(max_delta, row, col, p)
            if not cell:
                continue
            coeff_poly = sparse_mul(dict(grad_r[row]), dict(grad_s[col]), p)
            coeff_poly = sparse_scale(coeff_poly, -1, p)
            for delta, poly in cell.items():
                product = sparse_mul(coeff_poly, poly, p)
                if product:
                    accum[delta] = sparse_add(accum.get(delta, {}), product, p)
    return dm.sorted_delta_items(accum)


@lru_cache(maxsize=None)
def det_ratio_delta_power(
    max_delta: dm.DeltaKey,
    power: int,
    p: int,
) -> Tuple[Tuple[dm.DeltaKey, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Return det(Hessian(q))/det(Hessian(tau2)), raised to `power`.

    The determinant is normalized by the tau2 determinant so that the result is
    independent of the chosen Y-coordinate volume normalization.  For genus 2
    the power used by the pairing formula is 2.
    """

    perturbation = hessian_perturbation(p)
    matrix: List[List[dm.DeltaPoly]] = []
    for row in range(4):
        out_row: List[dm.DeltaPoly] = []
        for col in range(4):
            entry: dm.DeltaPoly = {dm.ZERO_DELTA: {ZERO_ALPHA: 1}} if row == col else {}
            for unit, perturb_matrix in zip(dm.DELTA_UNITS, perturbation):
                if dm.delta_leq(unit, max_delta):
                    poly = dict(perturb_matrix[row][col])
                    if poly:
                        entry[unit] = sparse_add(entry.get(unit, {}), poly, p)
            out_row.append(entry)
        matrix.append(out_row)

    det_poly: dm.DeltaPoly = {}
    for perm in permutations(range(4)):
        inversions = sum(1 for left in range(4) for right in range(left + 1, 4) if perm[left] > perm[right])
        term: dm.DeltaPoly = {dm.ZERO_DELTA: {ZERO_ALPHA: (-1 if inversions % 2 else 1) % p}}
        for row, col in enumerate(perm):
            term = dm.mul(term, matrix[row][col], max_delta, p)
            if not term:
                break
        det_poly = dm.add(det_poly, term, p)
    return dm.sorted_delta_items(dm.pow_delta(det_poly, power, max_delta, p))


def denominator_taylor_terms(max_delta: dm.DeltaKey, p: int) -> dm.KernelTerms:
    """Expand the four denominator factors involving B_j.

    The JK denominator has factors of the form 1 - exp(-B_j).  In the residue
    coordinates, the base part supplies the residue pole and the delta
    perturbation is expanded by Taylor series.  The Taylor order becomes a
    derivative order in the residue layer.
    """

    if max_delta == dm.ZERO_DELTA:
        return {(dm.ZERO_DELTA, dm.ZERO_DERIV): {ZERO_ALPHA: 1}}
    terms: dm.KernelTerms = {(dm.ZERO_DELTA, dm.ZERO_DERIV): {ZERO_ALPHA: 1}}
    max_order = sum(max_delta)
    unit_to_rank = {(1, 0, 0): 3, (0, 1, 0): 4, (0, 0, 1): 5}
    for j in range(1, 5):
        eps: dm.DeltaPoly = {
            unit: dict(b_perturbation(rank, j, p))
            for unit, rank in unit_to_rank.items()
            if dm.delta_leq(unit, max_delta)
        }
        factor: dm.KernelTerms = {}
        for order in range(max_order + 1):
            eps_power = dm.pow_delta(eps, order, max_delta, p)
            if not eps_power:
                continue
            deriv = [0, 0, 0, 0]
            deriv[j - 1] = order
            deriv_tuple = tuple(deriv)  # type: ignore[assignment]
            scale = mod_inv(mod_factorial(order, p), p)
            for delta, poly in eps_power.items():
                if poly:
                    factor[(delta, deriv_tuple)] = sparse_add(
                        factor.get((delta, deriv_tuple), {}),
                        poly,
                        p,
                        scale=scale,
                    )

        next_terms: dm.KernelTerms = {}
        for (d1, der1), poly1 in terms.items():
            for (d2, der2), poly2 in factor.items():
                next_delta = dm.delta_add(d1, d2)
                if not dm.delta_leq(next_delta, max_delta):
                    continue
                next_deriv = tuple(der1[idx] + der2[idx] for idx in range(4))  # type: ignore[assignment]
                product = sparse_mul(poly1, poly2, p)
                if product:
                    next_terms[(next_delta, next_deriv)] = sparse_add(
                        next_terms.get((next_delta, next_deriv), {}),
                        product,
                        p,
                    )
        terms = {key: value for key, value in next_terms.items() if value}
    return terms


@lru_cache(maxsize=None)
def even_kernel_terms(
    target_delta: dm.DeltaKey,
    p: int,
) -> Tuple[Tuple[dm.DeltaKey, dm.DerivOrders, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Return the even JK kernel terms up to the requested delta degree."""

    linear = {
        (1, 0, 0): dict(c_direction_term(3, p)),
        (0, 1, 0): dict(c_direction_term(4, p)),
        (0, 0, 1): dict(c_direction_term(5, p)),
    }
    exp_delta = dm.exp_linear(linear, target_delta, p)
    det_delta = {delta: dict(poly_items) for delta, poly_items in det_ratio_delta_power(target_delta, GENUS, p)}
    terms = denominator_taylor_terms(target_delta, p)
    terms = dm.kernel_terms_mul_delta(terms, exp_delta, target_delta, p)
    terms = dm.kernel_terms_mul_delta(terms, det_delta, target_delta, p)
    return tuple(
        (delta, deriv, sorted_items(poly))
        for (delta, deriv), poly in sorted(terms.items())
        if poly
    )
