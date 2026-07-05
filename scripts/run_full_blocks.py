#!/usr/bin/env python
"""Run resumable full W-column computations for selected Chern degrees."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r5g2higgs import DEFAULT_PRIME
from r5g2higgs.column_shard import read_column_shard, row_major_block_from_shards
from r5g2higgs.matrix_mod import matrix_shape, rank_mod_p, selected_row_indices, selected_w_indices
from r5g2higgs.shard_plan import (
    ensure_column_shards,
    existing_shard_matches_job,
    indices_digest,
    plan_column_shards,
    read_shard_payloads,
)

RUN_SCHEMA = "r5g2higgs.full_blocks_run.v1"
DEGREE_SCHEMA = "r5g2higgs.full_degree_summary.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_int_list(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    seen = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"descending range not allowed: {part}")
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for value in values:
            if value not in seen:
                out.append(value)
                seen.add(value)
    return tuple(out)


def block_checksum(block: Sequence[Sequence[int]], p: int) -> int:
    total = 0
    for row in block:
        for value in row:
            total = (total + int(value)) % p
    return total


def resolve_root(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clean_degree_list(degrees: Sequence[int], exclude: Sequence[int]) -> Tuple[int, ...]:
    excluded = set(int(value) for value in exclude)
    out = []
    seen = set()
    for degree in degrees:
        degree = int(degree)
        if degree in excluded or degree in seen:
            continue
        out.append(degree)
        seen.add(degree)
    return tuple(out)


def ordered_degrees(degrees: Tuple[int, ...], order: str) -> Tuple[int, ...]:
    if order == "natural":
        return degrees
    keyed = [(matrix_shape(degree).source_dimension, degree) for degree in degrees]
    if order == "small-first":
        return tuple(degree for _dim, degree in sorted(keyed))
    if order == "large-first":
        return tuple(degree for _dim, degree in sorted(keyed, reverse=True))
    raise ValueError(f"unknown degree order: {order}")


def full_range(values_count: int) -> Tuple[int, ...]:
    return tuple(range(values_count))


def existing_complete_summary(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != DEGREE_SCHEMA:
        return None
    if payload.get("status") != "complete":
        return None
    return payload


def ensure_one_shard_worker(job_index: int, total_jobs: int, job: Any, backend: str) -> Dict[str, Any]:
    """Process-pool worker for one shard.

    The job writes only its own deterministic shard path, so several workers can
    run safely in parallel. Existing valid shards return quickly as reused.
    """

    started = time.perf_counter()
    before_valid = existing_shard_matches_job(job)
    result = ensure_column_shards((job,), backend=backend)
    seconds = time.perf_counter() - started
    computed = bool(result.computed_paths)
    reused = bool(result.reused_paths)
    return {
        "job_index": job_index,
        "total_jobs": total_jobs,
        "w_indices": tuple(job.w_indices),
        "path": str(job.path),
        "valid_before": before_valid,
        "computed": computed,
        "reused": reused,
        "seconds": seconds,
    }


def shard_status_word(result: Dict[str, Any]) -> str:
    if result["computed"]:
        return "computed"
    if result["reused"]:
        return "reused"
    return "checked"


def run_degree(
    *,
    chern_degree: int,
    output_root: Path,
    row_indices_arg: Tuple[int, ...] | None,
    w_indices_arg: Tuple[int, ...] | None,
    columns_per_shard: int,
    p: int,
    backend: str,
    recompute_complete: bool,
    jobs_parallel: int,
    fresh_process_per_shard: bool = False,
    early_stop_full_rank: bool = True,
) -> Dict[str, Any]:
    shape = matrix_shape(chern_degree)
    rows = selected_row_indices(chern_degree, row_indices_arg)
    w_indices = selected_w_indices(
        chern_degree,
        full_range(shape.w_dimension) if w_indices_arg is None else w_indices_arg,
    )
    degree_dir = output_root / f"c{chern_degree}"
    shards_dir = degree_dir / "shards"
    progress_path = degree_dir / "progress.jsonl"
    summary_path = degree_dir / "summary.json"

    if not recompute_complete:
        existing = existing_complete_summary(summary_path)
        if existing is not None:
            print(
                f"c{chern_degree}: already complete, "
                f"rank={existing.get('rank')} checksum={existing.get('checksum')}",
                flush=True,
            )
            return existing

    jobs = plan_column_shards(
        shards_dir,
        chern_degree,
        w_indices,
        row_indices=rows,
        columns_per_shard=columns_per_shard,
        p=p,
    )
    valid_before = sum(1 for job in jobs if existing_shard_matches_job(job))
    degree_started = time.perf_counter()
    degree_started_at = utc_now()
    start_event = {
        "event": "degree_start",
        "time_utc": degree_started_at,
        "chern_degree": chern_degree,
        "source_dimension": shape.source_dimension,
        "w_dimension": shape.w_dimension,
        "row_count": len(rows),
        "w_count": len(w_indices),
        "columns_per_shard": columns_per_shard,
        "planned_shards": len(jobs),
        "valid_shards_before": valid_before,
        "backend": backend,
        "prime": p,
        "jobs": jobs_parallel,
        "early_stop_full_rank": early_stop_full_rank,
    }
    append_jsonl(progress_path, start_event)
    print(
        f"c{chern_degree}: start rows={len(rows)} w={len(w_indices)} "
        f"shards={len(jobs)} valid_before={valid_before} jobs={jobs_parallel}",
        flush=True,
    )

    computed_count = 0
    reused_count = 0
    completion_count = 0
    completed_payloads_by_path: Dict[str, Dict[str, Any]] = {}

    def completed_w_indices() -> Tuple[int, ...]:
        return tuple(
            sorted(
                int(column["w_index"])
                for payload in completed_payloads_by_path.values()
                for column in payload["columns"]
            )
        )

    def build_summary(completion_reason: str, early_stop: bool) -> Dict[str, Any]:
        payloads = tuple(completed_payloads_by_path.values())
        available_w_indices = completed_w_indices()
        block = row_major_block_from_shards(payloads, w_indices=available_w_indices)
        rank = rank_mod_p(block, p)
        checksum = block_checksum(block, p)
        seconds_total = time.perf_counter() - degree_started
        return {
            "schema": DEGREE_SCHEMA,
            "status": "complete",
            "completion_reason": completion_reason,
            "early_stop_full_rank": early_stop,
            "time_utc": utc_now(),
            "started_at_utc": degree_started_at,
            "chern_degree": chern_degree,
            "prime": p,
            "backend": backend,
            "source_dimension": shape.source_dimension,
            "w_dimension": shape.w_dimension,
            "row_count": len(rows),
            "w_count": len(available_w_indices),
            "requested_w_count": len(w_indices),
            "available_w_count": len(available_w_indices),
            "full_source_rows": rows == full_range(shape.source_dimension),
            "full_w_columns": available_w_indices == full_range(shape.w_dimension),
            "all_requested_w_columns": available_w_indices == tuple(sorted(w_indices)),
            "row_indices_digest": indices_digest(rows),
            "w_indices_digest": indices_digest(available_w_indices),
            "requested_w_indices_digest": indices_digest(w_indices),
            "columns_per_shard": columns_per_shard,
            "jobs": jobs_parallel,
            "planned_shards": len(jobs),
            "completed_shards": completion_count,
            "computed_shards": computed_count,
            "reused_shards": reused_count,
            "rank": rank,
            "source_nullity": len(rows) - rank,
            "checksum": checksum,
            "seconds": seconds_total,
            "shards_dir": str(shards_dir),
            "progress_log": str(progress_path),
        }

    def finish_degree(summary: Dict[str, Any], event_name: str) -> Dict[str, Any]:
        write_json(summary_path, summary)
        append_jsonl(progress_path, {"event": event_name, **summary})
        print(
            f"c{chern_degree}: complete rank={summary['rank']} "
            f"nullity={summary['source_nullity']} w={summary['w_count']}/{len(w_indices)} "
            f"reason={summary['completion_reason']} checksum={summary['checksum']} "
            f"seconds={summary['seconds']:.3f}",
            flush=True,
        )
        return summary

    def maybe_finish_early(summary: Dict[str, Any]) -> Dict[str, Any] | None:
        if not early_stop_full_rank:
            return None
        if summary["rank"] == len(rows):
            return finish_degree(summary, "degree_complete_early_full_rank")
        return None

    def record_shard_result(result: Dict[str, Any]) -> Dict[str, Any] | None:
        nonlocal computed_count, reused_count, completion_count

        completion_count += 1
        computed = bool(result["computed"])
        reused = bool(result["reused"])
        computed_count += int(computed)
        reused_count += int(reused)
        completed_payloads_by_path[str(result["path"])] = read_column_shard(result["path"])
        rank_summary = build_summary("early_stop_full_source_rank", True)
        event = {
            "event": "shard_done",
            "time_utc": utc_now(),
            "chern_degree": chern_degree,
            "job_index": result["job_index"],
            "completion_index": completion_count,
            "total_jobs": len(jobs),
            "w_indices": list(result["w_indices"]),
            "path": result["path"],
            "valid_before": result["valid_before"],
            "computed": computed,
            "reused": reused,
            "seconds": result["seconds"],
            "computed_count": computed_count,
            "reused_count": reused_count,
            "current_rank": rank_summary["rank"],
            "current_source_nullity": rank_summary["source_nullity"],
            "current_w_count": rank_summary["w_count"],
        }
        append_jsonl(progress_path, event)
        w_indices_done = result["w_indices"]
        print(
            f"c{chern_degree}: shard {completion_count}/{len(jobs)} "
            f"job={result['job_index']} w={w_indices_done[0]}..{w_indices_done[-1]} "
            f"{shard_status_word(result)} {result['seconds']:.3f}s "
            f"rank={rank_summary['rank']}/{len(rows)} "
            f"nullity={rank_summary['source_nullity']} "
            f"cols={rank_summary['w_count']}/{len(w_indices)}",
            flush=True,
        )
        return maybe_finish_early(rank_summary)

    def run_pending_jobs_parallel(
        pending_jobs: Sequence[Tuple[int, Any]],
        *,
        max_tasks_per_child: int | None = None,
    ) -> Dict[str, Any] | None:
        pending_iter = iter(pending_jobs)
        futures: Dict[Future[Dict[str, Any]], None] = {}

        def submit_next(pool: ProcessPoolExecutor) -> bool:
            try:
                job_index, job = next(pending_iter)
            except StopIteration:
                return False
            future = pool.submit(ensure_one_shard_worker, job_index, len(jobs), job, backend)
            futures[future] = None
            return True

        with ProcessPoolExecutor(
            max_workers=jobs_parallel,
            max_tasks_per_child=max_tasks_per_child,
        ) as pool:
            for _ in range(min(jobs_parallel, len(pending_jobs))):
                submit_next(pool)
            while futures:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    summary = record_shard_result(future.result())
                    if summary is not None:
                        for pending_future in futures:
                            pending_future.cancel()
                        return summary
                while len(futures) < jobs_parallel and submit_next(pool):
                    pass
        return None

    if fresh_process_per_shard:
        pending_jobs = []
        for job_index, job in enumerate(jobs, start=1):
            started = time.perf_counter()
            if existing_shard_matches_job(job):
                summary = record_shard_result(
                    {
                        "job_index": job_index,
                        "total_jobs": len(jobs),
                        "w_indices": tuple(job.w_indices),
                        "path": str(job.path),
                        "valid_before": True,
                        "computed": False,
                        "reused": True,
                        "seconds": time.perf_counter() - started,
                    }
                )
                if summary is not None:
                    return summary
            else:
                pending_jobs.append((job_index, job))

        if jobs_parallel <= 1:
            for job_index, job in pending_jobs:
                summary = record_shard_result(
                    ensure_one_shard_worker(job_index, len(jobs), job, backend)
                )
                if summary is not None:
                    return summary
        else:
            summary = run_pending_jobs_parallel(pending_jobs, max_tasks_per_child=1)
            if summary is not None:
                return summary
    elif jobs_parallel <= 1:
        for job_index, job in enumerate(jobs, start=1):
            summary = record_shard_result(
                ensure_one_shard_worker(job_index, len(jobs), job, backend)
            )
            if summary is not None:
                return summary
    else:
        summary = run_pending_jobs_parallel(tuple(enumerate(jobs, start=1)))
        if summary is not None:
            return summary

    if len(completed_payloads_by_path) != len(jobs):
        payloads = read_shard_payloads(job.path for job in jobs)
        completed_payloads_by_path.clear()
        completed_payloads_by_path.update((str(job.path), payload) for job, payload in zip(jobs, payloads))
    summary = build_summary("all_requested_columns", False)
    return finish_degree(summary, "degree_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrees", default="11-22")
    parser.add_argument("--exclude-degrees", default="12")
    parser.add_argument(
        "--degree-order",
        choices=("natural", "small-first", "large-first"),
        default="small-first",
    )
    parser.add_argument("--rows", default=None, help="optional row subset, e.g. 0-3,93")
    parser.add_argument("--w-indices", default=None, help="optional W subset, e.g. 0,10,37")
    parser.add_argument("--columns-per-shard", type=int, default=4)
    parser.add_argument("--output-root", default="artifacts/full_c_ne_12")
    parser.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    parser.add_argument("--backend", default="native")
    parser.add_argument("--jobs", type=int, default=1, help="parallel shard workers per degree")
    parser.add_argument(
        "--fresh-process-per-shard",
        action="store_true",
        help="compute each missing shard in a new worker process to cap memory growth",
    )
    parser.add_argument("--recompute-complete", action="store_true")
    parser.add_argument(
        "--no-early-stop-full-rank",
        dest="early_stop_full_rank",
        action="store_false",
        help="compute all requested W-columns even after the rows have full rank",
    )
    parser.set_defaults(early_stop_full_rank=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    degrees = clean_degree_list(
        parse_int_list(args.degrees),
        parse_int_list(args.exclude_degrees),
    )
    degrees = ordered_degrees(degrees, args.degree_order)
    output_root = resolve_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    row_indices_arg = None if args.rows is None else parse_int_list(args.rows)
    w_indices_arg = None if args.w_indices is None else parse_int_list(args.w_indices)

    run_manifest = {
        "schema": RUN_SCHEMA,
        "time_utc": utc_now(),
        "degrees": list(degrees),
        "excluded_degrees": list(parse_int_list(args.exclude_degrees)),
        "degree_order": args.degree_order,
        "row_selection": "all" if row_indices_arg is None else list(row_indices_arg),
        "w_selection": "all" if w_indices_arg is None else list(w_indices_arg),
        "columns_per_shard": args.columns_per_shard,
        "prime": args.prime,
        "backend": args.backend,
        "jobs": args.jobs,
        "fresh_process_per_shard": args.fresh_process_per_shard,
        "early_stop_full_rank": args.early_stop_full_rank,
        "output_root": str(output_root),
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, sort_keys=True), flush=True)

    if args.dry_run:
        for degree in degrees:
            shape = matrix_shape(degree)
            rows = selected_row_indices(degree, row_indices_arg)
            w_indices = selected_w_indices(
                degree,
                full_range(shape.w_dimension) if w_indices_arg is None else w_indices_arg,
            )
            jobs = plan_column_shards(
                output_root / f"c{degree}" / "shards",
                degree,
                w_indices,
                row_indices=rows,
                columns_per_shard=args.columns_per_shard,
                p=args.prime,
            )
            print(
                f"c{degree}: rows={len(rows)} w={len(w_indices)} "
                f"shards={len(jobs)} source_dim={shape.source_dimension}",
                flush=True,
            )
        return

    run_started = time.perf_counter()
    summaries = []
    for degree in degrees:
        summaries.append(
            run_degree(
                chern_degree=degree,
                output_root=output_root,
                row_indices_arg=row_indices_arg,
                w_indices_arg=w_indices_arg,
                columns_per_shard=args.columns_per_shard,
                p=args.prime,
                backend=args.backend,
                recompute_complete=args.recompute_complete,
                jobs_parallel=args.jobs,
                fresh_process_per_shard=args.fresh_process_per_shard,
                early_stop_full_rank=args.early_stop_full_rank,
            )
        )

    aggregate = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "time_utc": utc_now(),
        "seconds": time.perf_counter() - run_started,
        "manifest": run_manifest,
        "degrees": summaries,
    }
    write_json(output_root / "run_summary.json", aggregate)
    print(f"run complete: {output_root / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
