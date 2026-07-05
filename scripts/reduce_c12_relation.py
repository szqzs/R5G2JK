#!/usr/bin/env python
"""Extract and certify the primary c=12 relation line from column shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r5g2higgs.column_shard import row_major_block_from_shards, shard_file_sha256
from r5g2higgs.constants import DEFAULT_PRIME
from r5g2higgs.linear_mod import (
    column_dot_values,
    left_kernel_from_minor,
    matrix_rank_with_selected_columns,
    select_independent_columns,
    select_independent_rows,
    submatrix,
)
from r5g2higgs.matrix_mod import rank_mod_p
from r5g2higgs.shard_plan import indices_digest, read_shard_payloads


RELATION_SCHEMA = "r5g2higgs.c12_relation_reduce.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_root(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def unique_shard_field(payloads: Sequence[Dict[str, Any]], field: str) -> Any:
    if not payloads:
        raise ValueError("no shard payloads loaded")
    first = payloads[0][field]
    for payload in payloads[1:]:
        if payload[field] != first:
            raise ValueError(f"shards disagree on {field}")
    return first


def collect_w_indices(payloads: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
    seen = set()
    for payload in payloads:
        for column in payload["columns"]:
            w_index = int(column["w_index"])
            if w_index in seen:
                raise ValueError(f"duplicate W index across shards: {w_index}")
            seen.add(w_index)
    return tuple(sorted(seen))


def load_complete_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        return None
    return payload


def shard_inputs(paths: Sequence[Path]) -> Tuple[Dict[str, str], ...]:
    return tuple(
        {
            "path": str(path),
            "sha256": shard_file_sha256(path),
        }
        for path in paths
    )


def reduce_relation(
    *,
    shards_dir: Path,
    output_path: Path,
    summary_path: Path,
    expected_rank_arg: int | None,
    require_complete_summary: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    shard_paths = tuple(sorted(shards_dir.glob("*.json")))
    if not shard_paths:
        raise ValueError(f"no shard JSON files found in {shards_dir}")

    summary = load_complete_summary(summary_path)
    if require_complete_summary and summary is None:
        raise ValueError(f"complete degree summary not found at {summary_path}")

    payloads = read_shard_payloads(shard_paths)
    p = int(unique_shard_field(payloads, "prime"))
    chern_degree = int(unique_shard_field(payloads, "chern_degree"))
    source_degree = int(unique_shard_field(payloads, "source_degree"))
    w_degree = int(unique_shard_field(payloads, "w_degree"))
    row_indices = tuple(int(row) for row in unique_shard_field(payloads, "row_indices"))
    w_indices = collect_w_indices(payloads)
    block = row_major_block_from_shards(payloads, w_indices=w_indices)
    row_count = len(block)
    w_count = len(w_indices)
    if row_count != len(row_indices):
        raise ValueError("assembled block row count disagrees with shard row_indices")

    rank = rank_mod_p(block, p)
    expected_rank = row_count - 1 if expected_rank_arg is None else int(expected_rank_arg)
    if rank != expected_rank:
        raise ValueError(f"expected rank {expected_rank}, got rank {rank}")

    selected_column_positions = select_independent_columns(block, p, target_rank=expected_rank)
    column_restricted_block = submatrix(block, range(row_count), selected_column_positions, p)
    selected_row_positions = select_independent_rows(
        column_restricted_block,
        p,
        target_rank=expected_rank,
    )
    if matrix_rank_with_selected_columns(block, selected_column_positions, p) != expected_rank:
        raise ValueError("selected columns do not have the expected rank")

    relation = left_kernel_from_minor(
        block,
        selected_row_positions,
        selected_column_positions,
        p,
    )
    dots = column_dot_values(block, relation.vector, p)
    nonzero_dots = [
        {
            "column_position": idx,
            "w_index": w_indices[idx],
            "dot": dot,
        }
        for idx, dot in enumerate(dots)
        if dot
    ]
    if nonzero_dots:
        raise ValueError(f"relation failed on {len(nonzero_dots)} columns")

    selected_w_indices = tuple(w_indices[idx] for idx in selected_column_positions)
    payload: Dict[str, Any] = {
        "schema": RELATION_SCHEMA,
        "status": "complete",
        "time_utc": utc_now(),
        "seconds": time.perf_counter() - started,
        "prime": p,
        "chern_degree": chern_degree,
        "source_degree": source_degree,
        "w_degree": w_degree,
        "source_row_count": row_count,
        "w_count": w_count,
        "rank": rank,
        "source_nullity": row_count - rank,
        "expected_rank": expected_rank,
        "shards_dir": str(shards_dir),
        "summary_path": str(summary_path),
        "input_shard_count": len(shard_paths),
        "input_shards": list(shard_inputs(shard_paths)),
        "row_indices": list(row_indices),
        "row_indices_digest": indices_digest(row_indices),
        "w_indices": list(w_indices),
        "w_indices_digest": indices_digest(w_indices),
        "selected_row_positions": list(relation.selected_rows),
        "selected_source_row_indices": [row_indices[idx] for idx in relation.selected_rows],
        "omitted_row_position": relation.omitted_row,
        "omitted_source_row_index": row_indices[relation.omitted_row],
        "selected_column_positions": list(relation.selected_columns),
        "selected_w_indices": list(selected_w_indices),
        "minor_determinant": relation.minor_determinant,
        "relation_vector": list(relation.vector),
        "relation_coefficients": [
            {
                "row_position": idx,
                "source_row_index": row_indices[idx],
                "coefficient": coefficient,
            }
            for idx, coefficient in enumerate(relation.vector)
        ],
        "relation_vector_sha256": relation.vector_sha256,
        "all_column_dots_zero": True,
        "nonzero_dot_count": 0,
    }
    if summary is not None:
        payload["degree_summary"] = summary
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/c12_relation_primary")
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--shards-dir", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected-rank", type=int, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="do not require a complete run summary before reducing shards",
    )
    args = parser.parse_args()

    artifact_root = resolve_root(args.artifact_root)
    degree_dir = artifact_root / f"c{args.degree}"
    shards_dir = resolve_root(args.shards_dir) if args.shards_dir else degree_dir / "shards"
    summary_path = resolve_root(args.summary_path) if args.summary_path else degree_dir / "summary.json"
    output_path = resolve_root(args.output) if args.output else degree_dir / "relation_reduce.json"

    result = reduce_relation(
        shards_dir=shards_dir,
        output_path=output_path,
        summary_path=summary_path,
        expected_rank_arg=args.expected_rank,
        require_complete_summary=not args.allow_incomplete,
    )
    print(f"relation reduce complete: {output_path}")
    print(
        f"rank={result['rank']} nullity={result['source_nullity']} "
        f"vector_sha256={result['relation_vector_sha256']}"
    )


if __name__ == "__main__":
    main()
