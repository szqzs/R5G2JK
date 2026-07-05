"""Readable SymPy reference for the rank-5 genus-2 JK formula.

This module favors mathematical transparency over speed. It is meant to be a
reference and test oracle for the future fast implementation, not the production
rank engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

import sympy as sp

from .constants import COLLAPSED_CENTRAL_PREFACTOR, GAMMA_LABELS, GENUS
from .model import InvariantExp

Y = sp.symbols("Y1:5")
DELTA = sp.symbols("d3 d4 d5")

Alpha = Tuple[int, int, int, int]
DeltaKey = Tuple[int, int, int]
BLabel = Tuple[int, int]
MaskPoly = Dict[int, sp.Expr]

B_LABELS: Tuple[BLabel, ...] = tuple((r, j) for r in range(2, 6) for j in range(1, 5))
B_INDEX = {label: idx for idx, label in enumerate(B_LABELS)}

SIMPLE_COROOT_DIRECTIONS: Tuple[Alpha, ...] = (
    (2, -1, 0, 0),
    (-1, 2, -1, 0),
    (0, -1, 2, -1),
    (0, 0, -1, 2),
)


@dataclass(frozen=True)
class PaperBMonomial:
    """A monomial with raw JK b-labels."""

    a: Tuple[int, int, int, int] = (0, 0, 0, 0)
    f: Tuple[int, int, int, int] = (0, 0, 0, 0)
    b: Tuple[BLabel, ...] = ()

    def __post_init__(self) -> None:
        if len(self.a) != 4 or len(self.f) != 4:
            raise ValueError("a and f exponent tuples must have length 4")
        for r, j in self.b:
            if r < 2 or r > 5 or j < 1 or j > 4:
                raise ValueError(f"bad JK b-label {(r, j)}")


def x_coordinates() -> Tuple[sp.Expr, ...]:
    """Recover trace-zero x-coordinates from simple-difference Y-coordinates."""

    y1, y2, y3, y4 = Y
    return (
        sp.Rational(1, 5) * (4 * y1 + 3 * y2 + 2 * y3 + y4),
        sp.Rational(1, 5) * (-y1 + 3 * y2 + 2 * y3 + y4),
        sp.Rational(1, 5) * (-y1 - 2 * y2 + 2 * y3 + y4),
        sp.Rational(1, 5) * (-y1 - 2 * y2 - 3 * y3 + y4),
        sp.Rational(1, 5) * (-y1 - 2 * y2 - 3 * y3 - 4 * y4),
    )


@lru_cache(maxsize=None)
def tau(r: int) -> sp.Expr:
    """Elementary symmetric polynomial τ_r in the trace-zero x-coordinates."""

    if r < 2 or r > 5:
        raise ValueError(f"tau index must be 2,3,4,5, got {r}")
    total = sp.Integer(0)
    for combo in combinations(x_coordinates(), r):
        term = sp.prod(combo)
        total += term
    return sp.expand(total)


@lru_cache(maxsize=None)
def tau_gradient_y(r: int) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    tr = tau(r)
    return tuple(sp.expand(sp.diff(tr, y)) for y in Y)


def directional_derivative(expr: sp.Expr, direction_y: Sequence[int | sp.Expr]) -> sp.Expr:
    if len(direction_y) != 4:
        raise ValueError("direction_y must have length 4")
    return sp.expand(sum(direction_y[idx] * sp.diff(expr, Y[idx]) for idx in range(4)))


def q_polynomial() -> sp.Expr:
    d3, d4, d5 = DELTA
    return sp.expand(tau(2) + d3 * tau(3) + d4 * tau(4) + d5 * tau(5))


def c_tilde_direction_y() -> Alpha:
    """The direction of c-tilde in Y-coordinates.

    For c-tilde = (1/5,1/5,1/5,1/5,-4/5), only Y4 changes.
    """

    return (0, 0, 0, 1)


def c_tilde_exponent(q: sp.Expr) -> sp.Expr:
    return directional_derivative(q, c_tilde_direction_y())


def capital_b_components(q: sp.Expr) -> Tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """The four denominator terms B_j = -dq(h_j)."""

    return tuple(
        sp.expand(-directional_derivative(q, direction))
        for direction in SIMPLE_COROOT_DIRECTIONS
    )


@lru_cache(maxsize=None)
def b_perturbation(r: int, j: int) -> sp.Expr:
    """The δ_r coefficient in B_j, for r=3,4,5 and j=1,2,3,4."""

    if r < 3 or r > 5 or j < 1 or j > 4:
        raise ValueError(f"expected r=3,4,5 and j=1,2,3,4; got {(r, j)}")
    return sp.expand(-directional_derivative(tau(r), SIMPLE_COROOT_DIRECTIONS[j - 1]))


@lru_cache(maxsize=None)
def c_direction_term(r: int) -> sp.Expr:
    """The coefficient dτ_r(c-tilde) in the moment-map exponential."""

    if r < 3 or r > 5:
        raise ValueError(f"expected r=3,4,5; got {r}")
    return directional_derivative(tau(r), c_tilde_direction_y())


def positive_roots() -> Tuple[sp.Expr, ...]:
    """Positive roots in the Y-basis: Y_i + ... + Y_j."""

    roots: List[sp.Expr] = []
    for start in range(4):
        running = sp.Integer(0)
        for end in range(start, 4):
            running += Y[end]
            roots.append(sp.expand(running))
    return tuple(roots)


def denominator_root_product_squared() -> sp.Expr:
    product = sp.Integer(1)
    for root in positive_roots():
        product *= root**2
    return sp.expand(product)


def hessian_y(expr: sp.Expr) -> sp.Matrix:
    return sp.Matrix([[sp.diff(expr, left, right) for right in Y] for left in Y])


@lru_cache(maxsize=None)
def tau_hessian_y(r: int) -> sp.Matrix:
    return hessian_y(tau(r))


@lru_cache(maxsize=None)
def tau2_hessian_inverse_y() -> sp.Matrix:
    return tau_hessian_y(2).inv()


def det_hessian_ratio(q: sp.Expr) -> sp.Expr:
    """The normalized Hessian determinant det(H_q)/det(H_tau2)."""

    perturbation = hessian_y(q) - tau_hessian_y(2)
    matrix = sp.eye(4) + tau2_hessian_inverse_y() * perturbation
    return sp.expand(det_4x4_by_permutations(matrix))


def det_4x4_by_permutations(matrix: sp.Matrix) -> sp.Expr:
    """Small explicit determinant for 4x4 symbolic matrices."""

    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {matrix.shape}")
    total = sp.Integer(0)
    for perm in permutations(range(4)):
        inversions = sum(
            1
            for left in range(4)
            for right in range(left + 1, 4)
            if perm[left] > perm[right]
        )
        sign = -1 if inversions % 2 else 1
        term = sp.Integer(sign)
        for row, col in enumerate(perm):
            term *= matrix[row, col]
        total += term
    return total


def hessian_inverse_y(q: sp.Expr) -> sp.Matrix:
    """Inverse Hessian using the tau2 perturbation form."""

    perturbation = hessian_y(q) - tau_hessian_y(2)
    return (sp.eye(4) + tau2_hessian_inverse_y() * perturbation).inv() * tau2_hessian_inverse_y()


def hat_pair_coefficient(q: sp.Expr, r: int, s: int) -> sp.Expr:
    """Pair coefficient T_rs = -grad(τ_r)^T H_q^{-1} grad(τ_s)."""

    h_inv = hessian_inverse_y(q)
    grad_r = sp.Matrix(tau_gradient_y(r))
    grad_s = sp.Matrix(tau_gradient_y(s))
    return sp.factor(-(grad_r.T * h_inv * grad_s)[0])


def mask_for_b_label(label: BLabel) -> int:
    return 1 << B_INDEX[label]


def wedge_masks(left: int, right: int) -> Optional[Tuple[int, int]]:
    """Multiply two exterior monomial masks.

    Returns None if a b-variable repeats. Otherwise returns (sign, mask).
    """

    if left & right:
        return None
    inversions = 0
    for left_idx in range(len(B_LABELS)):
        if not (left & (1 << left_idx)):
            continue
        inversions += sum(1 for right_idx in range(left_idx) if right & (1 << right_idx))
    return (-1 if inversions % 2 else 1, left | right)


def exterior_mul(left: MaskPoly, right: MaskPoly) -> MaskPoly:
    out: MaskPoly = {}
    for left_mask, left_coeff in left.items():
        for right_mask, right_coeff in right.items():
            wedge = wedge_masks(left_mask, right_mask)
            if wedge is None:
                continue
            sign, mask = wedge
            out[mask] = sp.expand(out.get(mask, sp.Integer(0)) + sign * left_coeff * right_coeff)
    return {mask: coeff for mask, coeff in out.items() if coeff != 0}


def b_product_to_mask(labels: Sequence[BLabel]) -> Optional[Tuple[int, int]]:
    mask = 0
    sign = 1
    for label in labels:
        wedge = wedge_masks(mask, mask_for_b_label(label))
        if wedge is None:
            return None
        step_sign, mask = wedge
        sign *= step_sign
    return sign, mask


def hat_tau_exterior_quadratic(q: sp.Expr) -> MaskPoly:
    """The quadratic form whose exponential encodes b-insertions."""

    out: MaskPoly = {}
    for left_handle, right_handle in ((1, 3), (2, 4)):
        for r in range(2, 6):
            for s in range(2, 6):
                wedge = wedge_masks(
                    mask_for_b_label((r, left_handle)),
                    mask_for_b_label((s, right_handle)),
                )
                if wedge is None:
                    continue
                sign, mask = wedge
                coeff = sp.expand(sign * hat_pair_coefficient(q, r, s))
                out[mask] = sp.expand(out.get(mask, sp.Integer(0)) + coeff)
    return {mask: coeff for mask, coeff in out.items() if coeff != 0}


def b_insertion_factor(q: sp.Expr, labels: Sequence[BLabel]) -> sp.Expr:
    """Coefficient of a raw b-monomial in exp(hat_tau quadratic)."""

    target = b_product_to_mask(labels)
    if target is None:
        return sp.Integer(0)
    input_sign, target_mask = target
    if target_mask == 0:
        return sp.Integer(1)
    if target_mask.bit_count() % 2:
        return sp.Integer(0)

    quadratic = hat_tau_exterior_quadratic(q)
    pair_count = target_mask.bit_count() // 2
    power: MaskPoly = {0: sp.Integer(1)}
    for _ in range(pair_count):
        power = exterior_mul(power, quadratic)
        if not power:
            return sp.Integer(0)
    return sp.expand(input_sign * power.get(target_mask, sp.Integer(0)) / factorial(pair_count))


def gamma_b_terms(r: int, s: int) -> Tuple[Tuple[int, Tuple[BLabel, ...]], ...]:
    """Sp-invariant gamma_rs expanded in raw b-variables."""

    if r < 2 or r > 5 or s < 2 or s > 5:
        raise ValueError(f"gamma labels must be between 2 and 5, got {(r, s)}")
    return (
        (1, ((r, 1), (s, 3))),
        (-1, ((r, 3), (s, 1))),
        (1, ((r, 2), (s, 4))),
        (-1, ((r, 4), (s, 2))),
    )


def gamma_product_to_b_terms(gamma_exp: Sequence[int]) -> Tuple[Tuple[int, Tuple[BLabel, ...]], ...]:
    """Expand a gamma monomial into signed canonical b-monomials."""

    if len(gamma_exp) != len(GAMMA_LABELS):
        raise ValueError(f"expected {len(GAMMA_LABELS)} gamma exponents, got {len(gamma_exp)}")
    terms: Dict[Tuple[BLabel, ...], int] = {(): 1}
    for idx, exponent in enumerate(gamma_exp):
        if not exponent:
            continue
        factor = gamma_b_terms(*GAMMA_LABELS[idx])
        for _ in range(int(exponent)):
            next_terms: Dict[Tuple[BLabel, ...], int] = {}
            for labels, coeff in terms.items():
                for factor_coeff, factor_labels in factor:
                    target = b_product_to_mask(labels + factor_labels)
                    if target is None:
                        continue
                    sign, mask = target
                    canonical = tuple(label for label in B_LABELS if mask & mask_for_b_label(label))
                    next_terms[canonical] = next_terms.get(canonical, 0) + coeff * factor_coeff * sign
            terms = {labels: coeff for labels, coeff in next_terms.items() if coeff}
    return tuple(sorted((coeff, labels) for labels, coeff in terms.items() if coeff))


def gamma_insertion_factor(q: sp.Expr, gamma_exp: Sequence[int]) -> sp.Expr:
    total = sp.Integer(0)
    for coeff, labels in gamma_product_to_b_terms(gamma_exp):
        total += coeff * b_insertion_factor(q, labels)
    return sp.expand(total)


def a_monomial_factor(exponents: Sequence[int]) -> sp.Expr:
    if len(exponents) != 4:
        raise ValueError("a exponent tuple must have length 4")
    product = sp.Integer(1)
    for idx, exponent in enumerate(exponents):
        if exponent:
            product *= tau(idx + 2) ** int(exponent)
    return sp.expand(product)


def f_factorial_scale(exponents: Sequence[int]) -> int:
    if len(exponents) != 4:
        raise ValueError("f exponent tuple must have length 4")
    scale = 1
    for exponent in exponents:
        scale *= factorial(int(exponent))
    return scale


def delta_coefficient(expr: sp.Expr, orders: Sequence[int]) -> sp.Expr:
    """Coefficient of d3^orders[0] d4^orders[1] d5^orders[2]."""

    if len(orders) != 3:
        raise ValueError("delta order tuple must have length 3")
    out = expr
    scale = 1
    for delta_symbol, order in zip(DELTA, orders):
        order = int(order)
        if order:
            out = sp.diff(out, delta_symbol, order)
            scale *= factorial(order)
    out = out.subs({delta_symbol: 0 for delta_symbol in DELTA})
    return sp.factor(out / scale)


def generated_integrand_from_b(monomial: PaperBMonomial) -> sp.Expr:
    """JK generating integrand before extracting delta coefficients."""

    q = q_polynomial()
    denominator = denominator_root_product_squared()
    for component in capital_b_components(q):
        denominator *= 1 - sp.exp(-component)

    numerator = sp.exp(c_tilde_exponent(q))
    numerator *= a_monomial_factor(monomial.a)
    numerator *= det_hessian_ratio(q) ** GENUS
    numerator *= b_insertion_factor(q, monomial.b)
    return COLLAPSED_CENTRAL_PREFACTOR * numerator / denominator


def generated_integrand(exp: InvariantExp) -> sp.Expr:
    """JK generating integrand for an a/f/gamma exponent vector."""

    q = q_polynomial()
    denominator = denominator_root_product_squared()
    for component in capital_b_components(q):
        denominator *= 1 - sp.exp(-component)

    numerator = sp.exp(c_tilde_exponent(q))
    numerator *= a_monomial_factor(exp.a)
    numerator *= det_hessian_ratio(q) ** GENUS
    numerator *= gamma_insertion_factor(q, exp.gamma)
    return COLLAPSED_CENTRAL_PREFACTOR * numerator / denominator


def delta_extracted_integrand(exp: InvariantExp) -> sp.Expr:
    coeff = delta_coefficient(generated_integrand(exp), exp.target_delta)
    return f_factorial_scale(exp.f) * coeff


def one_variable_residue(expr: sp.Expr, var: sp.Symbol, series_order: int) -> sp.Expr:
    return sp.expand(sp.series(expr, var, 0, series_order).removeO().coeff(var, -1))


def iterated_residue(expr: sp.Expr, series_order: int = 80) -> sp.Expr:
    out = expr
    for var in reversed(Y):
        out = one_variable_residue(out, var, series_order)
    return out


def intersection_number(exp: InvariantExp, series_order: int = 80) -> sp.Expr:
    """Naive exact JK residue.

    This is mathematically direct but usually too slow for real JK entries.
    Keep it for experiments on very small controlled expressions; the
    production path should use a coefficient-truncated residue algorithm.
    """

    return iterated_residue(delta_extracted_integrand(exp), series_order=series_order)
