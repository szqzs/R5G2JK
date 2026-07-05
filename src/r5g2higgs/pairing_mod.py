"""Assembly of one modular pairing value.

This file is where the pieces of the JK formula meet.

Input
-----
An `InvariantExp` recording powers of

    a2,a3,a4,a5; f2,f3,f4,f5; gamma_rs.

Output
------
One element of F_p: the pairing of the corresponding class with the
fundamental class.

The computation does the following:

1. The powers of a_r become a product of tau_r(Y).
2. The powers of f3,f4,f5 select a coefficient in delta3,delta4,delta5.
3. The even kernel supplies the Hessian, determinant, exponential, and
   denominator pieces.
4. The gamma part supplies the odd contraction contribution.
5. The remaining Y-polynomial is sent to the residue layer.
6. The fixed scalar prefactor and f-factorials are applied.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Tuple

from .arithmetic import f_factorial_scale_mod
from .constants import COLLAPSED_CENTRAL_PREFACTOR, DEFAULT_PRIME
from .delta_mod import DeltaKey, DerivOrders, delta_sub
from .gamma_mod import gamma_hat
from .kernel_mod import even_kernel_terms
from .model import InvariantExp
from .residue_mod import residue_poly
from .sparse_mod import Alpha, mul as sparse_mul, sorted_items
from .tau_mod import tau_power


@lru_cache(maxsize=None)
def pairing_kernel_gamma_products(
    target_delta: DeltaKey,
    gamma_exp: Tuple[int, ...],
    p: int,
) -> Tuple[Tuple[DerivOrders, Tuple[Tuple[Alpha, int], ...]], ...]:
    """Precompute kernel-gamma factors shared by entries with the same f/gamma.

    The even kernel and the gamma contraction both produce delta expansions.
    We multiply only combinations whose delta exponents add to the requested
    coefficient selected by the f-exponents.
    """

    gamma_delta = {
        delta: dict(poly_items)
        for delta, poly_items in gamma_hat(gamma_exp, target_delta, p)
    }
    if not gamma_delta:
        return ()

    out: List[Tuple[DerivOrders, Tuple[Tuple[Alpha, int], ...]]] = []
    for kernel_delta, deriv_orders, kernel_items in even_kernel_terms(target_delta, p):
        gamma_needed = delta_sub(target_delta, kernel_delta)
        if gamma_needed is None or gamma_needed not in gamma_delta:
            continue
        shared_poly = sparse_mul(dict(kernel_items), gamma_delta[gamma_needed], p)
        if shared_poly:
            out.append((deriv_orders, sorted_items(shared_poly)))
    return tuple(out)


def pairing_total_mod_from_prepared(total: InvariantExp, p: int = DEFAULT_PRIME) -> int:
    """Evaluate the pairing for one invariant monomial."""

    target_delta: DeltaKey = total.target_delta
    shared_terms = pairing_kernel_gamma_products(target_delta, total.gamma, p)
    if not shared_terms:
        return 0

    a_poly = dict(tau_power(total.a, p))
    value = 0
    backend = os.environ.get("R5G2HIGGS_RESIDUE_BACKEND", "python").strip().lower()
    if backend in ("native", "auto"):
        from .residue_backend import native_available, residue_products_sum_native

        if backend == "native" or native_available():
            value = residue_products_sum_native(
                a_poly,
                [(deriv_orders, dict(shared_items)) for deriv_orders, shared_items in shared_terms],
                p,
            )
        else:
            backend = "python"

    if backend not in ("native", "auto"):
        for deriv_orders, shared_items in shared_terms:
            shared_poly = dict(shared_items)
            full_poly = sparse_mul(a_poly, shared_poly, p)
            if full_poly:
                value = (value + residue_poly(full_poly, deriv_orders, p)) % p

    # The scalar prefactor is fixed for this rank/determinant/genus
    # specialization.  The factorial scale accounts for repeated f-insertions
    # in the exponential generating function.
    scale = COLLAPSED_CENTRAL_PREFACTOR % p
    scale = scale * f_factorial_scale_mod(total.f, p) % p
    return scale * value % p


def pairing_totals_mod(totals: Iterable[InvariantExp], p: int = DEFAULT_PRIME) -> List[int]:
    return [pairing_total_mod_from_prepared(total, p) for total in totals]


@lru_cache(maxsize=None)
def pairing_total_mod(total: InvariantExp, p: int = DEFAULT_PRIME) -> int:
    return pairing_total_mod_from_prepared(total, p)
