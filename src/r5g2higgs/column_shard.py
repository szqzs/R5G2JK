"""Column-shard artifacts for Phase 4 matrix computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .constants import DEFAULT_PRIME, SOURCE_DEGREE, W_DEGREE
from .matrix_mod import (
    matrix_shape,
    pairing_columns,
    rank_mod_p,
    selected_row_indices,
    selected_w_indices,
)

SHARD_SCHEMA = "r5g2higgs.column_shard.v1"


def values_mod_checksum(values: Sequence[int], p: int = DEFAULT_PRIME) -> int:
    """Small modular checksum used for quick human-readable comparisons."""

    return sum(int(value) % p for value in values) % p


def values_sha256(values: Sequence[int], p: int = DEFAULT_PRIME) -> str:
    """Stable SHA-256 digest of a column's ordered values."""

    digest = hashlib.sha256()
    digest.update(f"p={int(p)}\n".encode("ascii"))
    digest.update(f"len={len(values)}\n".encode("ascii"))
    for value in values:
        digest.update(f"{int(value) % p}\n".encode("ascii"))
    return digest.hexdigest()


def column_metadata(values: Sequence[int], p: int = DEFAULT_PRIME) -> Dict[str, int | str]:
    normalized = tuple(int(value) % p for value in values)
    return {
        "entry_count": len(normalized),
        "nonzero_count": sum(1 for value in normalized if value),
        "mod_checksum": values_mod_checksum(normalized, p),
        "values_sha256": values_sha256(normalized, p),
    }


def build_column_shard(
    chern_degree: int,
    w_indices: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
    p: int = DEFAULT_PRIME,
    backend: str | None = "native",
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Dict[str, Any]:
    """Compute and package a deterministic column-shard payload."""

    shape = matrix_shape(chern_degree, source_degree=source_degree, w_degree=w_degree)
    rows = selected_row_indices(
        chern_degree,
        row_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    cols = selected_w_indices(
        chern_degree,
        w_indices,
        source_degree=source_degree,
        w_degree=w_degree,
    )
    column_values = pairing_columns(
        chern_degree,
        cols,
        row_indices=rows,
        p=p,
        backend=backend,
        source_degree=source_degree,
        w_degree=w_degree,
    )

    columns = []
    for w_index, values in zip(cols, column_values):
        normalized = [int(value) % p for value in values]
        columns.append(
            {
                "w_index": int(w_index),
                "values": normalized,
                **column_metadata(normalized, p),
            }
        )

    payload: Dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "prime": int(p),
        "source_degree": int(source_degree),
        "w_degree": int(w_degree),
        "chern_degree": int(chern_degree),
        "source_dimension": int(shape.source_dimension),
        "w_dimension": int(shape.w_dimension),
        "row_indices": list(rows),
        "columns": columns,
    }
    payload["shard_mod_checksum"] = values_mod_checksum(
        [
            int(column["mod_checksum"])
            for column in columns
        ],
        p,
    )
    return payload


def verify_column_shard(payload: Dict[str, Any]) -> None:
    """Validate shard structure and checksums.

    Raises `ValueError` when the payload is malformed or checksum verification
    fails.
    """

    if payload.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"unsupported column-shard schema: {payload.get('schema')!r}")
    p = int(payload["prime"])
    rows = tuple(int(row) for row in payload["row_indices"])
    if len(set(rows)) != len(rows):
        raise ValueError("row_indices contains duplicates")

    columns = payload.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("column shard must contain at least one column")

    seen_w = set()
    mod_checksums = []
    for column in columns:
        if not isinstance(column, dict):
            raise ValueError("column entries must be objects")
        w_index = int(column["w_index"])
        if w_index in seen_w:
            raise ValueError(f"duplicate W index in shard: {w_index}")
        seen_w.add(w_index)

        values = [int(value) for value in column["values"]]
        if len(values) != len(rows):
            raise ValueError(
                f"column {w_index} has {len(values)} values for {len(rows)} row indices"
            )
        if any(value < 0 or value >= p for value in values):
            raise ValueError(f"column {w_index} contains a noncanonical modulo-p value")

        expected = column_metadata(values, p)
        for key, expected_value in expected.items():
            if column.get(key) != expected_value:
                raise ValueError(
                    f"column {w_index} checksum mismatch for {key}: "
                    f"{column.get(key)!r} != {expected_value!r}"
                )
        mod_checksums.append(int(expected["mod_checksum"]))

    expected_shard_checksum = values_mod_checksum(mod_checksums, p)
    if payload.get("shard_mod_checksum") != expected_shard_checksum:
        raise ValueError(
            "shard_mod_checksum mismatch: "
            f"{payload.get('shard_mod_checksum')!r} != {expected_shard_checksum!r}"
        )


def write_column_shard(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Verify and write a column-shard JSON file."""

    verify_column_shard(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def read_column_shard(path: str | Path, *, verify: bool = True) -> Dict[str, Any]:
    """Read a column-shard JSON file and optionally verify checksums."""

    payload = json.loads(Path(path).read_text())
    if verify:
        verify_column_shard(payload)
    return payload


def shard_file_sha256(path: str | Path) -> str:
    """Return SHA-256 of the serialized shard file."""

    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _shared_shard_field(payloads: Sequence[Dict[str, Any]], field: str) -> Any:
    first = payloads[0][field]
    for payload in payloads[1:]:
        if payload[field] != first:
            raise ValueError(f"column shards disagree on {field}")
    return first


def block_columns_from_shards(
    payloads: Sequence[Dict[str, Any]],
    *,
    w_indices: Sequence[int] | None = None,
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    """Assemble column vectors from verified shard payloads.

    Returns `(w_indices, columns)`, where `columns` is a tuple of column vectors
    in the requested W order.
    """

    if not payloads:
        raise ValueError("at least one column shard payload is required")
    for payload in payloads:
        verify_column_shard(payload)

    p = int(_shared_shard_field(payloads, "prime"))
    _shared_shard_field(payloads, "source_degree")
    _shared_shard_field(payloads, "w_degree")
    _shared_shard_field(payloads, "chern_degree")
    _shared_shard_field(payloads, "source_dimension")
    _shared_shard_field(payloads, "w_dimension")
    _shared_shard_field(payloads, "row_indices")

    by_w: Dict[int, Tuple[int, ...]] = {}
    for payload in payloads:
        for column in payload["columns"]:
            w_index = int(column["w_index"])
            values = tuple(int(value) % p for value in column["values"])
            if w_index in by_w:
                if by_w[w_index] != values:
                    raise ValueError(f"conflicting values for W index {w_index}")
            else:
                by_w[w_index] = values

    selected = tuple(by_w.keys()) if w_indices is None else tuple(int(i) for i in w_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("requested w_indices contains duplicates")
    missing = [w_index for w_index in selected if w_index not in by_w]
    if missing:
        raise ValueError(f"requested W indices are missing from shards: {missing}")
    return selected, tuple(by_w[w_index] for w_index in selected)


def row_major_block_from_columns(columns: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Transpose column vectors into row-major matrix form."""

    if not columns:
        return ()
    height = len(columns[0])
    for idx, column in enumerate(columns):
        if len(column) != height:
            raise ValueError(f"column {idx} has height {len(column)}; expected {height}")
    return tuple(
        tuple(int(column[row_idx]) for column in columns)
        for row_idx in range(height)
    )


def row_major_block_from_shards(
    payloads: Sequence[Dict[str, Any]],
    *,
    w_indices: Sequence[int] | None = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Assemble a row-major block from one or more column shards."""

    _selected_w, columns = block_columns_from_shards(payloads, w_indices=w_indices)
    return row_major_block_from_columns(columns)


def rank_from_column_shards(
    payloads: Sequence[Dict[str, Any]],
    *,
    w_indices: Sequence[int] | None = None,
) -> int:
    """Assemble a row-major block from shards and compute its rank modulo p."""

    if not payloads:
        raise ValueError("at least one column shard payload is required")
    p = int(_shared_shard_field(payloads, "prime"))
    block = row_major_block_from_shards(payloads, w_indices=w_indices)
    return rank_mod_p(block, p)
