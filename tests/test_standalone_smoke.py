from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from r5g2higgs import DEFAULT_PRIME, GAMMA_LABELS, InvariantExp
from r5g2higgs import gamma_mod, pairing_mod, pairing_q, reference_sympy as ref, relation_q, residue_q, sparse_mod as spm, sparse_q, tau_mod, tau_q
from r5g2higgs.arithmetic import rational_mod
from r5g2higgs.basis_mod import basis_probe
from r5g2higgs.linear_mod import vector_dot_mod_p
from r5g2higgs.matrix_mod import entry_total_exponent, matrix_shape, pairing_entry
from r5g2higgs.residue_mod import residue_poly_batch, residue_poly_termwise


def gamma_exp(*labels: tuple[int, int]) -> tuple[int, ...]:
    values = [0] * len(GAMMA_LABELS)
    for label in labels:
        values[GAMMA_LABELS.index(label)] += 1
    return tuple(values)


def test_sparse_polynomial_core() -> None:
    p = DEFAULT_PRIME
    y1 = {(1, 0, 0, 0): 1}
    y2 = {(0, 1, 0, 0): 1}
    one = {spm.ZERO_ALPHA: 1}

    poly = spm.add(spm.add(one, y1, p), y2, p, scale=2)
    squared = spm.pow_poly(poly, 2, p)

    assert squared == {
        (0, 0, 0, 0): 1,
        (1, 0, 0, 0): 2,
        (0, 1, 0, 0): 4,
        (2, 0, 0, 0): 1,
        (1, 1, 0, 0): 4,
        (0, 2, 0, 0): 4,
    }
    assert spm.directional_derivative(squared, (1, -1, 0, 0), p) == spm.sub(
        spm.derivative(squared, 0, p),
        spm.derivative(squared, 1, p),
        p,
    )


def test_tau_polynomials_match_sympy_reference() -> None:
    p = DEFAULT_PRIME
    for r in range(2, 6):
        assert tau_mod.rank5_tau(r, p) == spm.sorted_items(spm.from_sympy_poly(ref.tau(r), ref.Y, p))

    assert len(tau_mod.rank5_tau(2, p)) == 10
    assert len(tau_mod.rank5_tau(5, p)) == 56


def test_exact_tau_layer_reduces_to_modular_tau_layer() -> None:
    p = DEFAULT_PRIME
    for r in range(2, 6):
        assert sparse_q.reduce_mod_p(dict(tau_q.rank5_tau_q(r)), p) == dict(tau_mod.rank5_tau(r, p))

        for idx in range(4):
            assert sparse_q.reduce_mod_p(dict(tau_q.tau_gradient_q(r)[idx]), p) == dict(tau_mod.tau_gradient(r, p)[idx])
            for jdx in range(4):
                assert sparse_q.reduce_mod_p(dict(tau_q.tau_hessian_q(r)[idx][jdx]), p) == dict(
                    tau_mod.tau_hessian(r, p)[idx][jdx]
                )

    for r in range(3, 6):
        assert sparse_q.reduce_mod_p(dict(tau_q.c_direction_term_q(r)), p) == dict(tau_mod.c_direction_term(r, p))
        for j in range(1, 5):
            assert sparse_q.reduce_mod_p(dict(tau_q.b_perturbation_q(r, j)), p) == dict(
                tau_mod.b_perturbation(r, j, p)
            )

    for a_exp in ((1, 0, 0, 0), (0, 1, 1, 0), (2, 0, 0, 1), (1, 1, 1, 1)):
        assert sparse_q.reduce_mod_p(dict(tau_q.tau_power_q(a_exp)), p) == dict(tau_mod.tau_power(a_exp, p))


def test_residue_batch_matches_termwise_reference() -> None:
    p = DEFAULT_PRIME
    poly = {
        (0, 0, 0, 0): 7,
        (2, 1, 0, 0): 11,
        (3, 2, 1, 0): 13,
        (0, 0, 2, 3): 17,
    }
    deriv = (1, 0, 2, 0)
    assert residue_poly_batch(poly, deriv, p) == residue_poly_termwise(poly, deriv, p)
    assert residue_poly_batch(poly, deriv, p) == 1212474583062236369


def test_exact_residue_layer_reduces_to_modular_residue_layer() -> None:
    p = DEFAULT_PRIME
    poly_q = {
        (0, 0, 0, 0): Fraction(7, 3),
        (2, 1, 0, 0): Fraction(-11, 5),
        (3, 2, 1, 0): Fraction(13, 7),
        (0, 0, 2, 3): Fraction(17, 11),
    }
    poly_mod = sparse_q.reduce_mod_p(poly_q, p)
    for deriv in ((0, 0, 0, 0), (1, 0, 2, 0), (0, 2, 1, 1), (3, 1, 0, 2)):
        exact = residue_q.residue_poly_q(poly_q, deriv)
        assert exact == residue_q.residue_poly_termwise_q(poly_q, deriv)
        assert rational_mod(exact, p) == residue_poly_batch(poly_mod, deriv, p)


def test_gamma_exterior_helpers() -> None:
    p = DEFAULT_PRIME
    left = gamma_mod.bit_for_label((2, 1))
    right = gamma_mod.bit_for_label((2, 3))

    assert gamma_mod.wedge_masks(left, left) is None
    assert gamma_mod.wedge_masks(left, right) == (1, left | right)
    assert gamma_mod.wedge_masks(right, left) == (-1, left | right)
    assert len(gamma_mod.gamma_mask_expansion(gamma_exp((2, 2)), p)) == 2


def test_basis_dimensions() -> None:
    payload = basis_probe(chern_degrees=tuple(range(11, 23)))
    assert payload["source_basis_dimensions_by_chern"] == {
        "11": 7,
        "12": 44,
        "13": 94,
        "14": 111,
        "15": 81,
        "16": 53,
        "17": 28,
        "18": 16,
        "19": 7,
        "20": 4,
        "21": 1,
        "22": 1,
    }
    assert payload["w_basis_dimension"] == 1039

    assert matrix_shape(12).source_dimension == 44
    assert matrix_shape(12).w_dimension == 1039


def test_pairing_sample_values() -> None:
    p = DEFAULT_PRIME
    direct_examples = {
        "zero": (InvariantExp(), 521034284106091052),
        "a2": (InvariantExp(a=(1, 0, 0, 0)), 1952920970927685354),
        "f2": (InvariantExp(f=(1, 0, 0, 0)), 521034284106091052),
        "f3": (InvariantExp(f=(0, 1, 0, 0)), 0),
        "a2_f3": (InvariantExp(a=(1, 0, 0, 0), f=(0, 1, 0, 0)), 714676139051499328),
        "gamma22": (InvariantExp(gamma=gamma_exp((2, 2))), 517533297074374825),
        "gamma25": (InvariantExp(gamma=gamma_exp((2, 5))), 610894949505499662),
        "gamma22_f3": (
            InvariantExp(f=(0, 1, 0, 0), gamma=gamma_exp((2, 2))),
            94396821244480507,
        ),
    }
    for _label, (exp, expected) in direct_examples.items():
        assert pairing_mod.pairing_total_mod(exp, p) == expected

    assert entry_total_exponent(14, 0, 0) == InvariantExp(a=(10, 0, 0, 0), f=(4, 0, 0, 0))
    assert pairing_entry(14, 0, 0, backend="python") == 1604819510747902861
    assert pairing_entry(14, 3, 10, backend="python") == 1972176177343518377
    assert pairing_entry(13, 0, 0, backend="python") == 0
    assert pairing_entry(15, 2, 5, backend="python") == 1669007804929659613


def test_exact_pairing_layer_reduces_to_modular_pairing_layer() -> None:
    p = DEFAULT_PRIME
    examples = (
        InvariantExp(),
        InvariantExp(a=(1, 0, 0, 0)),
        InvariantExp(f=(1, 0, 0, 0)),
        InvariantExp(f=(0, 1, 0, 0)),
        InvariantExp(a=(1, 0, 0, 0), f=(0, 1, 0, 0)),
        InvariantExp(gamma=gamma_exp((2, 2))),
        InvariantExp(gamma=gamma_exp((2, 5))),
        InvariantExp(f=(0, 1, 0, 0), gamma=gamma_exp((2, 2))),
    )
    for exp in examples:
        assert rational_mod(pairing_q.pairing_total_q(exp), p) == pairing_mod.pairing_total_mod(exp, p)


def test_exact_c12_relation_dot_reduces_to_modular_dot_for_sample_columns() -> None:
    p = DEFAULT_PRIME
    for w_index in (0, 2):
        exact = relation_q.relation_dot_q(w_index)
        modular_column = [
            pairing_entry(12, row_index, w_index, p=p, backend="python")
            for row_index in range(len(relation_q.C12_RELATION_INTEGER_VECTOR))
        ]
        assert exact == 0
        assert rational_mod(exact, p) == vector_dot_mod_p(relation_q.C12_RELATION_INTEGER_VECTOR, modular_column, p)


def test_committed_certificate_shapes() -> None:
    repo = Path(__file__).resolve().parents[1]
    full = json.loads((repo / "results/full_rank/full_rank_certificate.json").read_text())
    c12 = json.loads((repo / "results/c12_relation/c12_relation_certificate.json").read_text())

    assert all(d["rank"] == d["source_dimension"] == d["row_count"] for d in full["degrees"])
    assert c12["rank"] == 43
    assert c12["source_row_count"] == 44
    assert c12["source_nullity"] == 1
    assert len(c12["relation_vector"]) == 44
    assert c12["all_column_dots_zero"] is True


def test_manifest_hashes_match_committed_results() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / "results/MANIFEST.json").read_text())
    listed = {item["path"] for item in manifest["files"]}
    actual = {
        str(path.relative_to(repo))
        for path in (repo / "results").rglob("*.json")
        if path.name != "MANIFEST.json"
    }

    assert listed == actual
    for item in manifest["files"]:
        data = (repo / item["path"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
