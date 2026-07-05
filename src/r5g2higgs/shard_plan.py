"""Shard planning and resume helpers for Phase 4 matrix runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from .column_shard import (
    build_column_shard,
    read_column_shard,
    verify_column_shard,
    write_column_shard,
)
from .constants import DEFAULT_PRIME, SOURCE_DEGREE, W_DEGREE
from .matrix_mod import selected_row_indices, selected_w_indices


@dataclass(frozen=True)
class ColumnShardJob:
    """One planned column-shard computation."""

    path: Path
    chern_degree: int
    row_indices: Tuple[int, ...]
    w_indices: Tuple[int, ...]
    p: int = DEFAULT_PRIME
    source_degree: int = SOURCE_DEGREE
    w_degree: int = W_DEGREE


@dataclass(frozen=True)
class ColumnShardRunResult:
    """Result of ensuring a list of shard jobs exists."""

    paths: Tuple[Path, ...]
    computed_paths: Tuple[Path, ...]
    reused_paths: Tuple[Path, ...]


def chunked(values: Sequence[int], size: int) -> Tuple[Tuple[int, ...], ...]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    clean_values = tuple(int(value) for value in values)
    return tuple(
        clean_values[start : start + size]
        for start in range(0, len(clean_values), size)
    )


def indices_digest(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in indices:
        digest.update(f"{int(value)}\n".encode("ascii"))
    return digest.hexdigest()[:16]


def index_label(prefix: str, indices: Sequence[int]) -> str:
    values = tuple(int(value) for value in indices)
    if not values:
        return f"{prefix}empty"
    return f"{prefix}{len(values)}_{values[0]}_{values[-1]}_{indices_digest(values)}"


def shard_filename(
    chern_degree: int,
    row_indices: Sequence[int],
    w_indices: Sequence[int],
    *,
    p: int = DEFAULT_PRIME,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> str:
    row_part = index_label("r", row_indices)
    w_part = index_label("w", w_indices)
    return (
        f"c{int(chern_degree)}_sd{int(source_degree)}_wd{int(w_degree)}_"
        f"p{int(p)}_{row_part}_{w_part}.json"
    )


def plan_column_shards(
    output_dir: str | Path,
    chern_degree: int,
    w_indices: Sequence[int],
    *,
    row_indices: Sequence[int] | None = None,
    columns_per_shard: int = 4,
    p: int = DEFAULT_PRIME,
    source_degree: int = SOURCE_DEGREE,
    w_degree: int = W_DEGREE,
) -> Tuple[ColumnShardJob, ...]:
    """Plan deterministic column-shard jobs.

    The chosen `columns_per_shard` is a speed knob: larger chunks amortize basis
    setup and Python/Rust call overhead, while smaller chunks improve resume
    granularity.
    """

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
    root = Path(output_dir)
    jobs = []
    for col_chunk in chunked(cols, columns_per_shard):
        path = root / shard_filename(
            chern_degree,
            rows,
            col_chunk,
            p=p,
            source_degree=source_degree,
            w_degree=w_degree,
        )
        jobs.append(
            ColumnShardJob(
                path=path,
                chern_degree=int(chern_degree),
                row_indices=rows,
                w_indices=col_chunk,
                p=int(p),
                source_degree=int(source_degree),
                w_degree=int(w_degree),
            )
        )
    return tuple(jobs)


def payload_matches_job(payload: Dict[str, Any], job: ColumnShardJob) -> bool:
    try:
        verify_column_shard(payload)
    except ValueError:
        return False
    if int(payload.get("prime")) != job.p:
        return False
    if int(payload.get("source_degree")) != job.source_degree:
        return False
    if int(payload.get("w_degree")) != job.w_degree:
        return False
    if int(payload.get("chern_degree")) != job.chern_degree:
        return False
    if tuple(int(row) for row in payload.get("row_indices", ())) != job.row_indices:
        return False
    columns = payload.get("columns")
    if not isinstance(columns, list):
        return False
    return tuple(int(column.get("w_index")) for column in columns) == job.w_indices


def existing_shard_matches_job(job: ColumnShardJob) -> bool:
    if not job.path.exists():
        return False
    try:
        payload = read_column_shard(job.path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return payload_matches_job(payload, job)


def ensure_column_shards(
    jobs: Sequence[ColumnShardJob],
    *,
    backend: str | None = "native",
) -> ColumnShardRunResult:
    """Ensure all planned shard files exist and verify.

    Existing valid shards are reused. Missing, invalid, or stale shards are
    recomputed and overwritten.
    """

    computed = []
    reused = []
    paths = []
    for job in jobs:
        paths.append(job.path)
        if existing_shard_matches_job(job):
            reused.append(job.path)
            continue

        payload = build_column_shard(
            job.chern_degree,
            job.w_indices,
            row_indices=job.row_indices,
            p=job.p,
            backend=backend,
            source_degree=job.source_degree,
            w_degree=job.w_degree,
        )
        write_column_shard(job.path, payload)
        computed.append(job.path)

    return ColumnShardRunResult(
        paths=tuple(paths),
        computed_paths=tuple(computed),
        reused_paths=tuple(reused),
    )


def read_shard_payloads(paths: Iterable[str | Path]) -> Tuple[Dict[str, Any], ...]:
    return tuple(read_column_shard(path) for path in paths)
