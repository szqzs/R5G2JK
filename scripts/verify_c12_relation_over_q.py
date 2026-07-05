#!/usr/bin/env python
"""Verify the lifted c=12 relation by exact rational JK pairings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r5g2higgs.matrix_mod import source_basis, target_basis
from r5g2higgs.relation_q import C12_RELATION_INTEGER_VECTOR, relation_column_buckets_q, relation_dot_q

SCHEMA = "r5g2higgs.c12_relation_over_q_verify.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_root(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def fraction_payload(value: Fraction) -> Dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "text": str(value),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_w_indices(text: str | None) -> List[int] | None:
    if text is None:
        return None
    out = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return out


def selected_indices(all_indices: Sequence[int], start: int, limit: int | None, explicit: Sequence[int] | None) -> List[int]:
    if explicit is not None:
        return [int(idx) for idx in explicit]
    chosen = [idx for idx in all_indices if idx >= start]
    if limit is not None:
        chosen = chosen[: int(limit)]
    return chosen


def chunks(values: Sequence[int], chunk_size: int) -> List[List[int]]:
    chunk_size = max(1, int(chunk_size))
    return [list(values[start : start + chunk_size]) for start in range(0, len(values), chunk_size)]


def load_checked_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    return {int(item) for item in payload.get("checked_w_indices", [])}


def verify_one_w_index(w_index: int) -> Dict[str, Any]:
    value = relation_dot_q(int(w_index))
    bucket_count = len(relation_column_buckets_q(int(w_index)))
    return {
        "w_index": int(w_index),
        "bucket_count": bucket_count,
        "is_zero": value == 0,
        "dot": None if value == 0 else fraction_payload(value),
    }


def verify_w_index_chunk(w_indices: Sequence[int]) -> Dict[str, Any]:
    started = time.perf_counter()
    results = [verify_one_w_index(w_index) for w_index in w_indices]
    return {
        "w_indices": [int(w_index) for w_index in w_indices],
        "results": results,
        "seconds": time.perf_counter() - started,
    }


def verify_indices(
    w_indices: Sequence[int],
    output_path: Path,
    *,
    resume: bool,
    checkpoint_every: int,
    fail_fast: bool,
    jobs: int,
    chunk_size: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    source = source_basis(12)
    target = target_basis()
    if len(C12_RELATION_INTEGER_VECTOR) != len(source):
        raise ValueError("c=12 relation vector length does not match source basis")

    checked = load_checked_indices(output_path) if resume else set()
    checked_w_indices = sorted(checked)
    nonzero_dots: List[Dict[str, Any]] = []
    if resume and output_path.exists():
        existing_payload = json.loads(output_path.read_text())
        nonzero_dots = list(existing_payload.get("nonzero_dots", []))

    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "field": "Q",
        "started_at_utc": utc_now(),
        "time_utc": utc_now(),
        "seconds": 0.0,
        "chern_degree": 12,
        "source_degree": 22,
        "w_degree": 26,
        "source_dimension": len(source),
        "w_dimension": len(target),
        "relation_integer_vector": list(C12_RELATION_INTEGER_VECTOR),
        "requested_w_indices": list(w_indices),
        "checked_w_indices": checked_w_indices,
        "checked_count": len(checked_w_indices),
        "nonzero_dot_count": len(nonzero_dots),
        "nonzero_dots": nonzero_dots,
        "jobs": int(jobs),
        "chunk_size": int(chunk_size),
    }
    write_json(output_path, payload)

    checkpoint_every = max(1, int(checkpoint_every))
    pending_w_indices = [int(w_index) for w_index in w_indices if int(w_index) not in checked]
    for w_index in pending_w_indices:
        if w_index < 0 or w_index >= len(target):
            raise IndexError(f"W index {w_index} outside W dimension {len(target)}")

    def record_result(result: Dict[str, Any]) -> None:
        nonlocal checked_w_indices
        w_index = int(result["w_index"])
        checked.add(w_index)
        checked_w_indices = sorted(checked)
        bucket_count = int(result["bucket_count"])
        if not result["is_zero"]:
            nonzero_dots.append(
                {
                    "w_index": w_index,
                    "bucket_count": bucket_count,
                    "dot": result["dot"],
                }
            )
        payload.update(
            {
                "time_utc": utc_now(),
                "seconds": time.perf_counter() - started,
                "checked_w_indices": checked_w_indices,
                "checked_count": len(checked_w_indices),
                "last_checked_w_index": w_index,
                "last_bucket_count": bucket_count,
                "nonzero_dot_count": len(nonzero_dots),
                "nonzero_dots": nonzero_dots,
            }
        )

    if int(jobs) <= 1:
        since_checkpoint = 0
        for result in (verify_one_w_index(w_index) for w_index in pending_w_indices):
            record_result(result)
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every or not result["is_zero"]:
                write_json(output_path, payload)
                since_checkpoint = 0
            if (not result["is_zero"]) and fail_fast:
                payload["status"] = "failed"
                write_json(output_path, payload)
                raise ValueError(f"exact relation dot is nonzero for W index {result['w_index']}: {result['dot']['text']}")
    elif pending_w_indices:
        completed_chunks = 0
        work_chunks = chunks(pending_w_indices, chunk_size)
        payload.update({"planned_chunks": len(work_chunks), "completed_chunks": 0})
        write_json(output_path, payload)
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            futures = {executor.submit(verify_w_index_chunk, chunk): chunk for chunk in work_chunks}
            for future in as_completed(futures):
                chunk_result = future.result()
                completed_chunks += 1
                for result in chunk_result["results"]:
                    record_result(result)
                payload.update(
                    {
                        "completed_chunks": completed_chunks,
                        "last_chunk_w_indices": chunk_result["w_indices"],
                        "last_chunk_seconds": chunk_result["seconds"],
                    }
                )
                write_json(output_path, payload)
                if nonzero_dots and fail_fast:
                    payload["status"] = "failed"
                    write_json(output_path, payload)
                    for pending in futures:
                        pending.cancel()
                    first = nonzero_dots[0]
                    raise ValueError(f"exact relation dot is nonzero for W index {first['w_index']}: {first['dot']['text']}")

    payload.update(
        {
            "status": "complete" if not nonzero_dots else "failed",
            "time_utc": utc_now(),
            "seconds": time.perf_counter() - started,
            "checked_w_indices": sorted(checked),
            "checked_count": len(checked),
            "nonzero_dot_count": len(nonzero_dots),
            "all_checked_dots_zero": len(nonzero_dots) == 0,
        }
    )
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/c12_relation_over_q/verify_over_q.json")
    parser.add_argument("--w-indices", default=None, help="comma-separated W indices; defaults to all")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="record nonzero dots instead of stopping immediately")
    parser.add_argument("--jobs", type=int, default=1, help="number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=8, help="W indices per worker task")
    args = parser.parse_args()

    output_path = resolve_root(args.output)
    explicit = parse_w_indices(args.w_indices)
    all_indices = tuple(range(len(target_basis())))
    w_indices = selected_indices(all_indices, args.start, args.limit, explicit)
    result = verify_indices(
        w_indices,
        output_path,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        fail_fast=not args.keep_going,
        jobs=args.jobs,
        chunk_size=args.chunk_size,
    )
    print(f"exact c=12 relation verification: {output_path}")
    print(
        f"status={result['status']} checked={result['checked_count']} "
        f"nonzero={result['nonzero_dot_count']} seconds={result['seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
