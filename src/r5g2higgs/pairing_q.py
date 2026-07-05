"""Exact-Q assembly of one JK pairing value."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial
from typing import Iterable, List, Tuple

from .constants import COLLAPSED_CENTRAL_PREFACTOR
from .delta_mod import DeltaKey, DerivOrders, delta_sub
from .gamma_q import gamma_hat_q
from .kernel_q import even_kernel_terms_q
from .model import InvariantExp
from .residue_q import residue_poly_q
from .sparse_mod import Alpha
from .sparse_q import mul as sparse_mul, sorted_items
from .tau_q import tau_power_q


def f_factorial_scale_q(f_exp: tuple[int, int, int, int]) -> int:
    scale = 1
    for exponent in f_exp:
        scale *= factorial(int(exponent))
    return scale


@lru_cache(maxsize=None)
def pairing_kernel_gamma_products_q(
    target_delta: DeltaKey,
    gamma_exp: Tuple[int, ...],
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Alpha, Fraction], ...]], ...]:
    """Precompute exact kernel-gamma factors shared by entries with the same f/gamma."""

    gamma_delta = {
        delta: dict(poly_items)
        for delta, poly_items in gamma_hat_q(gamma_exp, target_delta)
    }
    if not gamma_delta:
        return ()

    out: List[Tuple[DerivOrders, Tuple[Tuple[Alpha, Fraction], ...]]] = []
    for kernel_delta, deriv_orders, kernel_items in even_kernel_terms_q(target_delta):
        gamma_needed = delta_sub(target_delta, kernel_delta)
        if gamma_needed is None or gamma_needed not in gamma_delta:
            continue
        shared_poly = sparse_mul(dict(kernel_items), gamma_delta[gamma_needed])
        if shared_poly:
            out.append((deriv_orders, sorted_items(shared_poly)))
    return tuple(out)


def pairing_total_q_from_prepared(total: InvariantExp) -> Fraction:
    """Evaluate the exact pairing for one invariant monomial."""

    target_delta: DeltaKey = total.target_delta
    shared_terms = pairing_kernel_gamma_products_q(target_delta, total.gamma)
    if not shared_terms:
        return Fraction(0)

    a_poly = dict(tau_power_q(total.a))
    value = Fraction(0)
    for deriv_orders, shared_items in shared_terms:
        shared_poly = dict(shared_items)
        full_poly = sparse_mul(a_poly, shared_poly)
        if full_poly:
            value += residue_poly_q(full_poly, deriv_orders)

    scale = COLLAPSED_CENTRAL_PREFACTOR * f_factorial_scale_q(total.f)
    return scale * value


@lru_cache(maxsize=None)
def pairing_total_q(total: InvariantExp) -> Fraction:
    return pairing_total_q_from_prepared(total)


def pairing_totals_q(totals: Iterable[InvariantExp]) -> List[Fraction]:
    return [pairing_total_q_from_prepared(total) for total in totals]
