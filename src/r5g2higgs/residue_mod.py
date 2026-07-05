"""Iterated residue transition for the rank-5 denominator.

At this point in the computation, the even kernel, the gamma contraction, and
the a-class tau powers have all been multiplied into a sparse polynomial in

    Y1, Y2, Y3, Y4.

The remaining operation is the Jeffrey-Kirwan iterated residue.  In this
rank-5 specialization, the denominator consists of the ten positive roots
e_i - e_j.  The code tracks the powers of these ten root factors while
eliminating the variables in the order

    Y4, Y3, Y2, Y1.

This file is the readable Python implementation.  The optional Rust backend in
`native/residue_kernel/` implements the same transition faster.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import comb, factorial
from typing import Dict, Tuple

from .delta_mod import DerivOrders
from .sparse_mod import Alpha, SparsePoly, ZERO_ALPHA, mod_inv

RANK5_ROOT_INTERVALS = tuple((i, j) for i in range(4) for j in range(i + 1, 5))
RANK5_ROOT_INDEX = {interval: idx for idx, interval in enumerate(RANK5_ROOT_INTERVALS)}
RANK5_D_ROOT_POWERS = tuple(2 for _ in RANK5_ROOT_INTERVALS)
RANK5_ZERO_DENOM = tuple(0 for _ in RANK5_ROOT_INTERVALS)
RANK5_BASE_LAMBDA_NUMS = (-1, -2, -3, -4)
"""Numerators of the exponential linear terms for the four Y variables.

The common denominator is 5.  These constants come from the rank-5
determinant-degree-1 specialization.
"""


@lru_cache(maxsize=None)
def h_coeffs(nmax: int, p: int) -> Tuple[int, ...]:
    """Coefficients of h(y)=(1-exp(-y))/y."""

    out = []
    fact = 1
    for n in range(nmax + 1):
        fact = fact * (n + 1) % p
        coeff = mod_inv(fact, p)
        if n % 2:
            coeff = (-coeff) % p
        out.append(coeff)
    return tuple(out)


@lru_cache(maxsize=None)
def poly_power_coeffs(base: Tuple[int, ...], power: int, nmax: int, p: int) -> Tuple[int, ...]:
    coeffs = [0 for _ in range(nmax + 1)]
    coeffs[0] = 1
    for _ in range(power):
        next_coeffs = [0 for _ in range(nmax + 1)]
        for left_idx, left_coeff in enumerate(coeffs):
            if not left_coeff:
                continue
            for right_idx, right_coeff in enumerate(base[: nmax + 1 - left_idx]):
                if right_coeff:
                    next_coeffs[left_idx + right_idx] = (
                        next_coeffs[left_idx + right_idx] + left_coeff * right_coeff
                    ) % p
        coeffs = next_coeffs
    return tuple(coeffs)


@lru_cache(maxsize=None)
def exp_coeffs(lam_num: int, nmax: int, p: int) -> Tuple[int, ...]:
    lam = lam_num * mod_inv(5, p) % p
    out = []
    fact = 1
    lam_power = 1
    for n in range(nmax + 1):
        if n:
            fact = fact * n % p
            lam_power = lam_power * lam % p
        out.append(lam_power * mod_inv(fact, p) % p)
    return tuple(out)


@lru_cache(maxsize=None)
def special_series(power: int, lam_num: int, cutoff: int, p: int) -> Tuple[Tuple[int, int], ...]:
    """Series needed when the simple-root denominator is eliminated.

    This packages the expansion of the exponential factor divided by powers of
    h(y) = (1 - exp(-y))/y.  The output is truncated to the range that can
    contribute to the residue coefficient.
    """

    nmax = cutoff + power
    if nmax < 0:
        return ()
    if power == 0:
        return tuple((n, coeff) for n, coeff in enumerate(exp_coeffs(lam_num, cutoff, p)) if coeff)
    h_power = poly_power_coeffs(h_coeffs(nmax, p), power, nmax, p)
    e_coeffs = exp_coeffs(lam_num, nmax, p)
    quotient = [0 for _ in range(nmax + 1)]
    for n in range(nmax + 1):
        coeff = e_coeffs[n]
        for i in range(1, n + 1):
            coeff = (coeff - h_power[i] * quotient[n - i]) % p
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
def special_derivative_dict(order: int, lam_num: int, cutoff: int, p: int) -> Dict[int, int]:
    """Return coefficients after applying the requested Y-derivative order."""

    if order == 0:
        return dict(special_series(1, lam_num, cutoff, p))
    accum: Dict[int, int] = {}
    sign = -1 if order % 2 else 1
    st = stirling2(order)
    for k in range(1, order + 1):
        s2 = st[order][k]
        if not s2:
            continue
        scale = sign * factorial(k) * s2
        for exponent, coeff in special_series(k + 1, lam_num - 5 * k, cutoff, p):
            accum[exponent] = (accum.get(exponent, 0) + scale * coeff) % p
    return {exponent: coeff for exponent, coeff in accum.items() if coeff % p}


@lru_cache(maxsize=None)
def binomial_series(root_power: int, cutoff: int, p: int) -> Tuple[int, ...]:
    return tuple(((-1) ** m * comb(root_power + m - 1, m)) % p for m in range(cutoff + 1))


@lru_cache(maxsize=None)
def root_transition_schedule() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """For each Y variable, list root factors affected by eliminating it."""

    by_var = [[] for _ in range(4)]
    for var_idx in range(4):
        for interval, pos in RANK5_ROOT_INDEX.items():
            if interval[1] != var_idx + 1:
                continue
            lower_pos = -1 if interval[0] == var_idx else RANK5_ROOT_INDEX[(interval[0], var_idx)]
            by_var[var_idx].append((pos, lower_pos))
    return tuple(tuple(items) for items in by_var)


def max_survivable_y_exp(
    var_idx: int,
    deriv_orders: DerivOrders,
    denom_powers: Tuple[int, ...],
    current_root_pos: int,
) -> int:
    simple_pos = RANK5_ROOT_INDEX[(var_idx, var_idx + 1)]
    simple_drop = int(denom_powers[simple_pos]) if current_root_pos < simple_pos else 0
    return int(deriv_orders[var_idx]) + simple_drop


@lru_cache(maxsize=None)
def variable_transition(
    var_idx: int,
    deriv_orders: DerivOrders,
    y_exp: int,
    denom_powers: Tuple[int, ...],
    p: int,
) -> Tuple[Tuple[Tuple[int, ...], int], ...]:
    """Eliminate one Y variable from one monomial/root-denominator state.

    The state consists of:

    - the current exponent of this Y variable;
    - the ten denominator powers for the positive roots.

    The output is a finite list of new denominator-power states and
    coefficients after taking the residue in this Y variable.
    """

    states: Dict[Tuple[int, Tuple[int, ...]], int] = {(int(y_exp), denom_powers): 1}
    for pos, lower_pos in root_transition_schedule()[var_idx]:
        next_states: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        for (cur_y_exp, dtuple), state_coeff in states.items():
            root_power = int(dtuple[pos])
            if not root_power:
                key = (cur_y_exp, dtuple)
                next_states[key] = (next_states.get(key, 0) + state_coeff) % p
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
                next_states[key] = (next_states.get(key, 0) + state_coeff) % p
                continue
            max_m = y_bound - cur_y_exp
            if max_m < 0:
                continue
            binoms = binomial_series(root_power, max_m, p)
            for m in range(max_m + 1):
                expanded_den = list(base_den_tuple)
                expanded_den[lower_pos] += root_power + m
                key = (cur_y_exp + m, tuple(expanded_den))
                next_states[key] = (next_states.get(key, 0) + state_coeff * binoms[m]) % p
        states = {key: value for key, value in next_states.items() if value % p}
        if not states:
            return ()

    needed_cutoff = max(max(0, -1 - cur_y_exp) for cur_y_exp, _dtuple in states)
    special = special_derivative_dict(
        deriv_orders[var_idx],
        RANK5_BASE_LAMBDA_NUMS[var_idx],
        needed_cutoff,
        p,
    )
    out: Dict[Tuple[int, ...], int] = {}
    for (cur_y_exp, dtuple), state_coeff in states.items():
        special_coeff = special.get(-1 - cur_y_exp)
        if special_coeff:
            out[dtuple] = (out.get(dtuple, 0) + state_coeff * special_coeff) % p
    return tuple(sorted((dtuple, coeff) for dtuple, coeff in out.items() if coeff % p))


@lru_cache(maxsize=None)
def residue_monomial(alpha: Alpha, deriv_orders: DerivOrders, p: int) -> int:
    """Return the JK residue of one Y monomial."""

    terms: Dict[Tuple[Alpha, Tuple[int, ...]], int] = {(alpha, RANK5_D_ROOT_POWERS): 1}
    for var_idx in reversed(range(4)):
        new_terms: Dict[Tuple[Alpha, Tuple[int, ...]], int] = {}
        for (cur_alpha, denom_powers), coeff in terms.items():
            next_alpha = list(cur_alpha)
            next_alpha[var_idx] = 0
            next_alpha_t = tuple(next_alpha)  # type: ignore[assignment]
            for dtuple, transition_coeff in variable_transition(
                var_idx,
                deriv_orders,
                cur_alpha[var_idx],
                denom_powers,
                p,
            ):
                key = (next_alpha_t, dtuple)
                new_terms[key] = (new_terms.get(key, 0) + coeff * transition_coeff) % p
        terms = {key: value for key, value in new_terms.items() if value % p}
        if not terms:
            return 0
    total = 0
    for (cur_alpha, denom_powers), coeff in terms.items():
        if cur_alpha == ZERO_ALPHA and denom_powers == RANK5_ZERO_DENOM:
            total = (total + coeff) % p
    return total


def residue_poly_termwise(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    """Evaluate a polynomial by summing monomial residues one by one."""

    total = 0
    for alpha, coeff in poly.items():
        res = residue_monomial(alpha, deriv_orders, p)
        if res:
            total = (total + int(coeff) * res) % p
    return total


def residue_poly_batch_python(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    # Specialized rank-5 batch elimination.  After eliminating Y4, then Y3,
    # then Y2, then Y1, the eliminated alpha coordinates are known to be zero;
    # keeping them in every state costs a lot of tuple/list allocation.
    terms3: Dict[Tuple[int, int, int, Tuple[int, ...]], int] = {}
    for (a0, a1, a2, a3), coeff in poly.items():
        coeff_mod = int(coeff) % p
        if not coeff_mod:
            continue
        for dtuple, transition_coeff in variable_transition(
            3,
            deriv_orders,
            a3,
            RANK5_D_ROOT_POWERS,
            p,
        ):
            key = (a0, a1, a2, dtuple)
            value = (terms3.get(key, 0) + coeff_mod * transition_coeff) % p
            if value:
                terms3[key] = value
            else:
                terms3.pop(key, None)
    if not terms3:
        return 0

    terms2: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    for (a0, a1, a2, denom_powers), coeff in terms3.items():
        for dtuple, transition_coeff in variable_transition(
            2,
            deriv_orders,
            a2,
            denom_powers,
            p,
        ):
            key = (a0, a1, dtuple)
            value = (terms2.get(key, 0) + coeff * transition_coeff) % p
            if value:
                terms2[key] = value
            else:
                terms2.pop(key, None)
    if not terms2:
        return 0

    terms1: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    for (a0, a1, denom_powers), coeff in terms2.items():
        for dtuple, transition_coeff in variable_transition(
            1,
            deriv_orders,
            a1,
            denom_powers,
            p,
        ):
            key = (a0, dtuple)
            value = (terms1.get(key, 0) + coeff * transition_coeff) % p
            if value:
                terms1[key] = value
            else:
                terms1.pop(key, None)
    if not terms1:
        return 0

    total = 0
    for (a0, denom_powers), coeff in terms1.items():
        for dtuple, transition_coeff in variable_transition(
            0,
            deriv_orders,
            a0,
            denom_powers,
            p,
        ):
            if dtuple == RANK5_ZERO_DENOM:
                total = (total + coeff * transition_coeff) % p
    return total


def residue_poly(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    """Public residue entry point used by the pairing evaluator."""

    return residue_poly_batch(poly, deriv_orders, p)


def residue_poly_batch(poly: SparsePoly, deriv_orders: DerivOrders, p: int) -> int:
    """Evaluate a residue using either Python or the optional native backend."""

    backend = os.environ.get("R5G2HIGGS_RESIDUE_BACKEND", "python").strip().lower()
    if backend in ("", "python"):
        return residue_poly_batch_python(poly, deriv_orders, p)
    if backend in ("native", "auto"):
        from .residue_backend import native_available, residue_poly_batch_native

        if backend == "auto" and not native_available():
            return residue_poly_batch_python(poly, deriv_orders, p)
        return residue_poly_batch_native(poly, deriv_orders, p)
    raise ValueError(f"unknown residue backend: {backend!r}")
