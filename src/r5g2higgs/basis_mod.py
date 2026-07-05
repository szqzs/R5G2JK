#!/usr/bin/env python
"""Sp-invariant basis enumeration.

This file chooses the finite source and target bases used by the pairing
matrices.

The invariant generators are:

    a2,a3,a4,a5
    f2,f3,f4,f5
    gamma22,gamma23,...,gamma55.

The gamma variables are not treated as abstract independent variables when
we test linear independence.  Each gamma_rs is expanded into the exterior
variables b_r^j.  Products are then reduced in the raw super-commutative
monomial basis.

So the basis algorithm is:

1. enumerate all products of the invariant generators in the desired ordinary
   degree;
2. expand every gamma product into exterior b variables;
3. row-reduce those expansions over Q;
4. keep the products that survive as an independent invariant basis.

No JK residue is evaluated in this file.  It only constructs the finite bases
whose pairings are computed later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import model as fm
from . import basis_support as jk


HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "basis_probe_results.json")

RawKey = Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], int]
Vector = Dict[RawKey, Fraction]

ZERO_A = (0, 0, 0, 0)
ZERO_F = (0, 0, 0, 0)
ZERO_GAMMA = tuple(0 for _ in jk.GAMMA_LABELS)
ONE_RAW_KEY: RawKey = (ZERO_A, ZERO_F, 0)


@dataclass(frozen=True)
class Generator:
    """One formal invariant generator before products are formed."""

    name: str
    exp: fm.InvariantExp
    vector: Tuple[Tuple[RawKey, int], ...]
    ordinary_degree: int
    chern_degree: int


@dataclass(frozen=True)
class BasisItem:
    """One independent invariant product selected for a basis."""

    name: str
    exp: fm.InvariantExp
    vector: Tuple[Tuple[RawKey, Fraction], ...]
    ordinary_degree: int
    chern_degree: int


def add_tuple4(left: Sequence[int], right: Sequence[int]) -> Tuple[int, int, int, int]:
    return tuple(int(left[i]) + int(right[i]) for i in range(4))  # type: ignore[return-value]


def add_exp(left: fm.InvariantExp, right: fm.InvariantExp) -> fm.InvariantExp:
    return fm.InvariantExp(
        add_tuple4(left.a, right.a),
        add_tuple4(left.f, right.f),
        tuple(int(left.gamma[i]) + int(right.gamma[i]) for i in range(len(jk.GAMMA_LABELS))),
    )


def ordinary_degree_exp(exp: fm.InvariantExp) -> int:
    """Return ordinary/cohomological degree of an invariant monomial."""

    total = 0
    for offset, power in enumerate(exp.a):
        rank = offset + 2
        total += int(power) * 2 * rank
    for offset, power in enumerate(exp.f):
        rank = offset + 2
        total += int(power) * (2 * rank - 2)
    for power, (r, s) in zip(exp.gamma, jk.GAMMA_LABELS):
        total += int(power) * (2 * r + 2 * s - 2)
    return total


def chern_degree_exp(exp: fm.InvariantExp) -> int:
    """Return the Chern-degree grading of an invariant monomial."""

    total = 0
    for offset, power in enumerate(exp.a):
        total += int(power) * (offset + 2)
    for offset, power in enumerate(exp.f):
        total += int(power) * (offset + 2)
    for power, (r, s) in zip(exp.gamma, jk.GAMMA_LABELS):
        total += int(power) * (r + s)
    return total


def b_mask_degree(mask: int) -> int:
    total = 0
    for idx, (r, _j) in enumerate(jk.B_LABELS):
        if mask & (1 << idx):
            total += 2 * r - 1
    return total


def b_mask_chern_degree(mask: int) -> int:
    total = 0
    for idx, (r, _j) in enumerate(jk.B_LABELS):
        if mask & (1 << idx):
            total += r
    return total


def raw_key_degree(key: RawKey) -> int:
    a, f, mask = key
    return (
        sum(int(power) * 2 * (idx + 2) for idx, power in enumerate(a))
        + sum(int(power) * (2 * (idx + 2) - 2) for idx, power in enumerate(f))
        + b_mask_degree(mask)
    )


def raw_key_chern_degree(key: RawKey) -> int:
    a, f, mask = key
    return (
        sum(int(power) * (idx + 2) for idx, power in enumerate(a))
        + sum(int(power) * (idx + 2) for idx, power in enumerate(f))
        + b_mask_chern_degree(mask)
    )


def raw_key_names(key: RawKey) -> Tuple[str, ...]:
    a, f, mask = key
    names: List[str] = []
    for offset, power in enumerate(a):
        names.extend([f"a{offset + 2}"] * int(power))
    for offset, power in enumerate(f):
        names.extend([f"f{offset + 2}"] * int(power))
    for idx, (r, j) in enumerate(jk.B_LABELS):
        if mask & (1 << idx):
            names.append(f"b{r}_{j}")
    return tuple(names)


def raw_key_sort(key: RawKey) -> Tuple[Any, ...]:
    return (raw_key_chern_degree(key), key[2].bit_count(), raw_key_names(key))


def raw_key_mul(left: RawKey, right: RawKey) -> Optional[Tuple[int, RawKey]]:
    """Multiply raw monomials, including the exterior sign."""

    wedge = jk.wedge_masks(left[2], right[2])
    if wedge is None:
        return None
    sign, mask = wedge
    return sign, (add_tuple4(left[0], right[0]), add_tuple4(left[1], right[1]), mask)


def vector_mul(left: Vector, right: Vector) -> Vector:
    out: Vector = {}
    for key_left, coeff_left in left.items():
        for key_right, coeff_right in right.items():
            product = raw_key_mul(key_left, key_right)
            if product is None:
                continue
            sign, key = product
            value = out.get(key, Fraction(0)) + sign * coeff_left * coeff_right
            if value:
                out[key] = value
            else:
                out.pop(key, None)
    return out


def compact_product_name(names: Sequence[str]) -> str:
    out: List[str] = []
    idx = 0
    while idx < len(names):
        token = names[idx]
        count = 1
        idx += 1
        while idx < len(names) and names[idx] == token:
            count += 1
            idx += 1
        out.append(token if count == 1 else f"{token}^{count}")
    return " ".join(out)


def mask_from_labels(labels: Sequence[jk.BLabel]) -> int:
    mask = 0
    for label in labels:
        mask |= jk.mask_for_b_label(label)
    return mask


@lru_cache(maxsize=None)
def generator_list(rank: int = 5) -> Tuple[Generator, ...]:
    """Return the ordered invariant generators used in this computation."""

    if rank != 5:
        raise ValueError("this computation specializes to rank 5")
    generators: List[Generator] = []
    for r in range(2, 6):
        a = [0, 0, 0, 0]
        a[r - 2] = 1
        exp = fm.InvariantExp(tuple(a), ZERO_F, ZERO_GAMMA)
        raw = (tuple(a), ZERO_F, 0)
        generators.append(Generator(f"a{r}", exp, ((raw, 1),), 2 * r, r))
    for r in range(2, 6):
        f = [0, 0, 0, 0]
        f[r - 2] = 1
        exp = fm.InvariantExp(ZERO_A, tuple(f), ZERO_GAMMA)
        raw = (ZERO_A, tuple(f), 0)
        generators.append(Generator(f"f{r}", exp, ((raw, 1),), 2 * r - 2, r))
    for idx, (r, s) in enumerate(jk.GAMMA_LABELS):
        gamma = [0 for _ in jk.GAMMA_LABELS]
        gamma[idx] = 1
        terms: Dict[RawKey, int] = {}
        for coeff, labels in jk.gamma_product_to_b_terms(tuple(gamma)):
            key = (ZERO_A, ZERO_F, mask_from_labels(labels))
            terms[key] = terms.get(key, 0) + int(coeff)
        exp = fm.InvariantExp(ZERO_A, ZERO_F, tuple(gamma))
        vector = tuple(sorted(terms.items(), key=lambda item: raw_key_sort(item[0])))
        generators.append(Generator(f"gamma{r}{s}", exp, vector, 2 * r + 2 * s - 2, r + s))
    return tuple(generators)


@lru_cache(maxsize=None)
def enumerate_invariant_products(rank: int, ordinary_degree: int) -> Tuple[BasisItem, ...]:
    """Enumerate all invariant-generator products of a fixed ordinary degree."""

    gens = generator_list(rank)
    out: List[BasisItem] = []

    def rec(start: int, rem: int, names: List[str], exp: fm.InvariantExp, vector: Vector) -> None:
        if rem == 0:
            if vector:
                out.append(
                    BasisItem(
                        compact_product_name(names),
                        exp,
                        tuple(sorted(vector.items(), key=lambda item: raw_key_sort(item[0]))),
                        ordinary_degree_exp(exp),
                        chern_degree_exp(exp),
                    )
                )
            return
        for idx in range(start, len(gens)):
            gen = gens[idx]
            if gen.ordinary_degree > rem:
                continue
            next_vector = vector_mul(vector, {key: Fraction(coeff) for key, coeff in gen.vector})
            if not next_vector:
                continue
            rec(
                idx,
                rem - gen.ordinary_degree,
                names + [gen.name],
                add_exp(exp, gen.exp),
                next_vector,
            )

    rec(0, ordinary_degree, [], fm.InvariantExp(ZERO_A, ZERO_F, ZERO_GAMMA), {ONE_RAW_KEY: Fraction(1)})
    return tuple(out)


@lru_cache(maxsize=None)
def raw_monomials_of_degree(rank: int, ordinary_degree: int) -> Tuple[RawKey, ...]:
    if rank != 5:
        raise ValueError("this computation specializes to rank 5")
    odd_masks: List[int] = []

    def rec_odd(idx: int, used: int, mask: int) -> None:
        if used > ordinary_degree:
            return
        if idx == len(jk.B_LABELS):
            odd_masks.append(mask)
            return
        rec_odd(idx + 1, used, mask)
        r, _j = jk.B_LABELS[idx]
        rec_odd(idx + 1, used + 2 * r - 1, mask | (1 << idx))

    rec_odd(0, 0, 0)
    out: List[RawKey] = []

    even_weights = tuple([(2 * r, "a", r - 2) for r in range(2, 6)] + [(2 * r - 2, "f", r - 2) for r in range(2, 6)])

    def rec_even(pos: int, rem: int, a: List[int], f: List[int], mask: int) -> None:
        if rem == 0:
            out.append((tuple(a), tuple(f), mask))  # type: ignore[arg-type]
            return
        if pos == len(even_weights):
            return
        weight, kind, offset = even_weights[pos]
        max_power = rem // weight
        for power in range(max_power + 1):
            if power:
                if kind == "a":
                    a[offset] = power
                else:
                    f[offset] = power
            rec_even(pos + 1, rem - power * weight, a, f, mask)
            if power:
                if kind == "a":
                    a[offset] = 0
                else:
                    f[offset] = 0

    for mask in odd_masks:
        used = b_mask_degree(mask)
        if used <= ordinary_degree:
            rec_even(0, ordinary_degree - used, [0, 0, 0, 0], [0, 0, 0, 0], mask)

    return tuple(sorted(set(out), key=raw_key_sort))


def sparse_reduce(row: Vector, pivots: Dict[RawKey, Vector]) -> Vector:
    """Reduce one raw expansion against previously chosen pivot rows."""

    out = {key: Fraction(value) for key, value in row.items() if value}
    for pivot_key in sorted(pivots, key=raw_key_sort):
        coeff = out.get(pivot_key)
        if not coeff:
            continue
        pivot = pivots[pivot_key]
        for key, value in pivot.items():
            new_value = out.get(key, Fraction(0)) - coeff * value
            if new_value:
                out[key] = new_value
            else:
                out.pop(key, None)
    return out


def normalize_row(row: Vector) -> Tuple[RawKey, Vector]:
    pivot = min(row, key=raw_key_sort)
    inv = Fraction(1, 1) / row[pivot]
    return pivot, {key: value * inv for key, value in row.items() if value}


@lru_cache(maxsize=None)
def independent_invariant_basis(rank: int, ordinary_degree: int) -> Tuple[Tuple[BasisItem, ...], Dict[str, int]]:
    """Choose an independent invariant basis in one ordinary degree."""

    products = enumerate_invariant_products(rank, ordinary_degree)
    pivots: Dict[RawKey, Vector] = {}
    keep: List[BasisItem] = []
    for item in products:
        reduced = sparse_reduce(dict(item.vector), pivots)
        if not reduced:
            continue
        pivot, normalized = normalize_row(reduced)
        pivots[pivot] = normalized
        keep.append(item)
    meta = {
        "ordinary_degree": ordinary_degree,
        "generated_invariant_products": len(products),
        "raw_free_dimension": len(raw_monomials_of_degree(rank, ordinary_degree)),
        "invariant_rank": len(keep),
        "dependent_products": len(products) - len(keep),
    }
    return tuple(keep), meta


def independent_basis_by_chern(
    rank: int,
    ordinary_degree: int,
    chern_degrees: Optional[Sequence[int]] = None,
) -> Tuple[Dict[int, Tuple[BasisItem, ...]], Dict[str, int], Dict[str, Dict[str, int]]]:
    """Choose independent source bases, separated by Chern degree."""

    products = enumerate_invariant_products(rank, ordinary_degree)
    selected = set(chern_degrees) if chern_degrees is not None else None
    grouped: Dict[int, List[BasisItem]] = {}
    for item in products:
        if item.ordinary_degree != ordinary_degree:
            raise ValueError(f"{item.name!r} has ordinary degree {item.ordinary_degree}, expected {ordinary_degree}")
        cdegree = item.chern_degree
        if selected is not None and cdegree not in selected:
            continue
        grouped.setdefault(cdegree, []).append(item)

    basis_by_chern: Dict[int, Tuple[BasisItem, ...]] = {}
    raw_counts: Dict[str, int] = {}
    meta_by_chern: Dict[str, Dict[str, int]] = {}
    raw_free_dimension = len(raw_monomials_of_degree(rank, ordinary_degree))
    for cdegree in sorted(grouped):
        pivots: Dict[RawKey, Vector] = {}
        keep: List[BasisItem] = []
        for item in grouped[cdegree]:
            reduced = sparse_reduce(dict(item.vector), pivots)
            if not reduced:
                continue
            pivot, normalized = normalize_row(reduced)
            pivots[pivot] = normalized
            keep.append(item)
        basis_by_chern[cdegree] = tuple(keep)
        raw_counts[str(cdegree)] = len(grouped[cdegree])
        meta_by_chern[str(cdegree)] = {
            "ordinary_degree": ordinary_degree,
            "chern_degree": cdegree,
            "raw_product_count": len(grouped[cdegree]),
            "raw_free_dimension": raw_free_dimension,
            "invariant_rank": len(keep),
            "dependent_products": len(grouped[cdegree]) - len(keep),
        }
    return basis_by_chern, raw_counts, meta_by_chern


def parse_degree_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    return str(value)


def basis_probe(
    *,
    rank: int = 5,
    source_degree: int = 22,
    w_degree: int = 26,
    chern_degrees: Sequence[int] = tuple(range(11, 23)),
) -> Dict[str, Any]:
    source_by_chern, raw_counts, meta_by_chern = independent_basis_by_chern(rank, source_degree, chern_degrees)
    w_basis, w_meta = independent_invariant_basis(rank, w_degree)
    return {
        "runner": "jk_only_basis_probe",
        "status": "passed",
        "rank": rank,
        "source_ordinary_degree": source_degree,
        "w_ordinary_degree": w_degree,
        "requested_chern_degrees": list(chern_degrees),
        "source_basis_dimensions_by_chern": {str(c): len(source_by_chern.get(c, ())) for c in chern_degrees},
        "source_raw_product_counts_by_chern": raw_counts,
        "source_basis_meta_by_chern": meta_by_chern,
        "w_basis_meta": w_meta,
        "w_basis_dimension": len(w_basis),
        "source_basis_names_by_chern": {
            str(c): [item.name for item in source_by_chern.get(c, ())]
            for c in chern_degrees
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--source-degree", type=int, default=22)
    parser.add_argument("--w-degree", type=int, default=26)
    parser.add_argument("--chern-degrees", default="11-22")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.time()
    payload = basis_probe(
        rank=args.rank,
        source_degree=args.source_degree,
        w_degree=args.w_degree,
        chern_degrees=parse_degree_list(args.chern_degrees),
    )
    payload["elapsed_seconds"] = time.time() - started
    ready = json_ready(payload)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(ready, handle, indent=2, sort_keys=True)
    print(json.dumps(ready, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
