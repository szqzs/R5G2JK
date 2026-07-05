"""Exact-Q iterated residue transition for the rank-5 denominator.

This mirrors `residue_mod.py`, but the coefficients are exact rational numbers.
It is intentionally finite and coefficient-truncated; it does not ask SymPy to
expand the full rational JK integrand.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial
from typing import Dict, Tuple

from .delta_mod import DerivOrders
from .residue_mod import (
    RANK5_BASE_LAMBDA_NUMS,
    RANK5_D_ROOT_POWERS,
    RANK5_ROOT_INDEX,
    RANK5_ZERO_DENOM,
    max_survivable_y_exp,
    root_transition_schedule,
)
from .sparse_mod import Alpha, ZERO_ALPHA
from .sparse_q import SparseQPoly


@lru_cache(maxsize=None)
def h_coeffs_q(nmax: int) -> Tuple[Fraction, ...]:
    """Coefficients of h(y)=(1-exp(-y))/y over Q."""

    return tuple(Fraction(-1 if n % 2 else 1, factorial(n + 1)) for n in range(nmax + 1))


@lru_cache(maxsize=None)
def poly_power_coeffs_q(base: Tuple[Fraction, ...], power: int, nmax: int) -> Tuple[Fraction, ...]:
    coeffs = [Fraction(0) for _ in range(nmax + 1)]
    coeffs[0] = Fraction(1)
    for _ in range(power):
        next_coeffs = [Fraction(0) for _ in range(nmax + 1)]
        for left_idx, left_coeff in enumerate(coeffs):
            if not left_coeff:
                continue
            for right_idx, right_coeff in enumerate(base[: nmax + 1 - left_idx]):
                if right_coeff:
                    next_coeffs[left_idx + right_idx] += left_coeff * right_coeff
        coeffs = next_coeffs
    return tuple(coeffs)


@lru_cache(maxsize=None)
def exp_coeffs_q(lam_num: int, nmax: int) -> Tuple[Fraction, ...]:
    lam = Fraction(lam_num, 5)
    out = []
    fact = 1
    lam_power = Fraction(1)
    for n in range(nmax + 1):
        if n:
            fact *= n
            lam_power *= lam
        out.append(lam_power / fact)
    return tuple(out)


@lru_cache(maxsize=None)
def special_series_q(power: int, lam_num: int, cutoff: int) -> Tuple[Tuple[int, Fraction], ...]:
    """Series used when one simple-root denominator is eliminated."""

    nmax = cutoff + power
    if nmax < 0:
        return ()
    if power == 0:
        return tuple((n, coeff) for n, coeff in enumerate(exp_coeffs_q(lam_num, cutoff)) if coeff)
    h_power = poly_power_coeffs_q(h_coeffs_q(nmax), power, nmax)
    e_coeffs = exp_coeffs_q(lam_num, nmax)
    quotient = [Fraction(0) for _ in range(nmax + 1)]
    for n in range(nmax + 1):
        coeff = e_coeffs[n]
        for idx in range(1, n + 1):
            coeff -= h_power[idx] * quotient[n - idx]
        quotient[n] = coeff
    return tuple((n - power, coeff) for n, coeff in enumerate(quotient) if coeff)


@lru_cache(maxsize=None)
def stirling2(nmax: int) -> Tuple[Tuple[int, ...], ...]:
    table = [[0 for _ in range(nmax + 1)] for _ in range(nmax + 1)]
    table[0][0] = 1
    for n in range(1, nmax + 1):
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + k * table[n - 1][k]
    return tuple(tuple(row) for row in table)


@lru_cache(maxsize=None)
def special_derivative_dict_q(order: int, lam_num: int, cutoff: int) -> Dict[int, Fraction]:
    """Return exact coefficients after applying the requested Y-derivative order."""

    if order == 0:
        return dict(special_series_q(1, lam_num, cutoff))
    accum: Dict[int, Fraction] = {}
    sign = -1 if order % 2 else 1
    st = stirling2(order)
    for k in range(1, order + 1):
        s2 = st[order][k]
        if not s2:
            continue
        scale = sign * factorial(k) * s2
        for exponent, coeff in special_series_q(k + 1, lam_num - 5 * k, cutoff):
            accum[exponent] = accum.get(exponent, Fraction(0)) + scale * coeff
    return {exponent: coeff for exponent, coeff in accum.items() if coeff}


@lru_cache(maxsize=None)
def binomial_series_q(root_power: int, cutoff: int) -> Tuple[Fraction, ...]:
    return tuple(Fraction(((-1) ** m) * comb(root_power + m - 1, m)) for m in range(cutoff + 1))


@lru_cache(maxsize=None)
def variable_transition_q(
    var_idx: int,
    deriv_orders: DerivOrders,
    y_exp: int,
    denom_powers: Tuple[int, ...],
) -> Tuple[Tuple[Tuple[int, ...], Fraction], ...]:
    """Eliminate one Y variable from one monomial/root-denominator state over Q."""

    states: Dict[Tuple[int, Tuple[int, ...]], Fraction] = {(int(y_exp), denom_powers): Fraction(1)}
    for pos, lower_pos in root_transition_schedule()[var_idx]:
        next_states: Dict[Tuple[int, Tuple[int, ...]], Fraction] = {}
        for (cur_y_exp, dtuple), state_coeff in states.items():
            root_power = int(dtuple[pos])
            if not root_power:
                key = (cur_y_exp, dtuple)
                next_states[key] = next_states.get(key, Fraction(0)) + state_coeff
                continue
            base_den = list(dtuple)
            base_den[pos] = 0
            base_den_tuple = tuple(base_den)
            y_bound = max_survivable_y_exp(var_idx, deriv_orders, base_den_tuple, pos)
            if lower_pos < 0:
                next_y_exp = cur_y_exp - root_power
                if next_y_exp > y_bound:
                    continue
                key = (next_y_exp, base_den_tuple)
                next_states[key] = next_states.get(key, Fraction(0)) + state_coeff
                continue
            max_m = y_bound - cur_y_exp
            if max_m < 0:
                continue
            binoms = binomial_series_q(root_power, max_m)
            for m in range(max_m + 1):
                expanded_den = list(base_den_tuple)
                expanded_den[lower_pos] += root_power + m
                key = (cur_y_exp + m, tuple(expanded_den))
                next_states[key] = next_states.get(key, Fraction(0)) + state_coeff * binoms[m]
        states = {key: value for key, value in next_states.items() if value}
        if not states:
            return ()

    needed_cutoff = max(max(0, -1 - cur_y_exp) for cur_y_exp, _dtuple in states)
    special = special_derivative_dict_q(
        deriv_orders[var_idx],
        RANK5_BASE_LAMBDA_NUMS[var_idx],
        needed_cutoff,
    )
    out: Dict[Tuple[int, ...], Fraction] = {}
    for (cur_y_exp, dtuple), state_coeff in states.items():
        special_coeff = special.get(-1 - cur_y_exp)
        if special_coeff:
            out[dtuple] = out.get(dtuple, Fraction(0)) + state_coeff * special_coeff
    return tuple(sorted((dtuple, coeff) for dtuple, coeff in out.items() if coeff))


@lru_cache(maxsize=None)
def residue_monomial_q(alpha: Alpha, deriv_orders: DerivOrders) -> Fraction:
    """Return the exact JK residue of one Y monomial."""

    terms: Dict[Tuple[Alpha, Tuple[int, ...]], Fraction] = {(alpha, RANK5_D_ROOT_POWERS): Fraction(1)}
    for var_idx in reversed(range(4)):
        new_terms: Dict[Tuple[Alpha, Tuple[int, ...]], Fraction] = {}
        for (cur_alpha, denom_powers), coeff in terms.items():
            next_alpha = list(cur_alpha)
            next_alpha[var_idx] = 0
            next_alpha_t = tuple(next_alpha)  # type: ignore[assignment]
            for dtuple, transition_coeff in variable_transition_q(
                var_idx,
                deriv_orders,
                cur_alpha[var_idx],
                denom_powers,
            ):
                key = (next_alpha_t, dtuple)
                new_terms[key] = new_terms.get(key, Fraction(0)) + coeff * transition_coeff
        terms = {key: value for key, value in new_terms.items() if value}
        if not terms:
            return Fraction(0)
    total = Fraction(0)
    for (cur_alpha, denom_powers), coeff in terms.items():
        if cur_alpha == ZERO_ALPHA and denom_powers == RANK5_ZERO_DENOM:
            total += coeff
    return total


def residue_poly_termwise_q(poly: SparseQPoly, deriv_orders: DerivOrders) -> Fraction:
    """Evaluate an exact polynomial by summing monomial residues one by one."""

    total = Fraction(0)
    for alpha, coeff in poly.items():
        res = residue_monomial_q(alpha, deriv_orders)
        if res:
            total += Fraction(coeff) * res
    return total


def residue_poly_batch_q(poly: SparseQPoly, deriv_orders: DerivOrders) -> Fraction:
    """Specialized rank-5 batch residue over Q."""

    terms3: Dict[Tuple[int, int, int, Tuple[int, ...]], Fraction] = {}
    for (a0, a1, a2, a3), coeff in poly.items():
        coeff_q = Fraction(coeff)
        if not coeff_q:
            continue
        for dtuple, transition_coeff in variable_transition_q(3, deriv_orders, a3, RANK5_D_ROOT_POWERS):
            key = (a0, a1, a2, dtuple)
            value = terms3.get(key, Fraction(0)) + coeff_q * transition_coeff
            if value:
                terms3[key] = value
            else:
                terms3.pop(key, None)
    if not terms3:
        return Fraction(0)

    terms2: Dict[Tuple[int, int, Tuple[int, ...]], Fraction] = {}
    for (a0, a1, a2, denom_powers), coeff in terms3.items():
        for dtuple, transition_coeff in variable_transition_q(2, deriv_orders, a2, denom_powers):
            key = (a0, a1, dtuple)
            value = terms2.get(key, Fraction(0)) + coeff * transition_coeff
            if value:
                terms2[key] = value
            else:
                terms2.pop(key, None)
    if not terms2:
        return Fraction(0)

    terms1: Dict[Tuple[int, Tuple[int, ...]], Fraction] = {}
    for (a0, a1, denom_powers), coeff in terms2.items():
        for dtuple, transition_coeff in variable_transition_q(1, deriv_orders, a1, denom_powers):
            key = (a0, dtuple)
            value = terms1.get(key, Fraction(0)) + coeff * transition_coeff
            if value:
                terms1[key] = value
            else:
                terms1.pop(key, None)
    if not terms1:
        return Fraction(0)

    total = Fraction(0)
    for (a0, denom_powers), coeff in terms1.items():
        for dtuple, transition_coeff in variable_transition_q(0, deriv_orders, a0, denom_powers):
            if dtuple == RANK5_ZERO_DENOM:
                total += coeff * transition_coeff
    return total


def residue_poly_q(poly: SparseQPoly, deriv_orders: DerivOrders) -> Fraction:
    """Public exact residue entry point."""

    return residue_poly_batch_q(poly, deriv_orders)
