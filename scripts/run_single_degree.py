#!/usr/bin/env python
"""Run one degree without rewriting the root-level full-run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_full_blocks import parse_int_list, resolve_root, run_degree

from r5g2higgs import DEFAULT_PRIME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--w-indices", default=None)
    parser.add_argument("--columns-per-shard", type=int, default=4)
    parser.add_argument("--output-root", default="artifacts/full_c_ne_12")
    parser.add_argument("--prime", type=int, default=DEFAULT_PRIME)
    parser.add_argument("--backend", default="native")
    parser.add_argument("--jobs", type=int, default=1)
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
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")

    output_root = resolve_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    row_indices = None if args.rows is None else parse_int_list(args.rows)
    w_indices = None if args.w_indices is None else parse_int_list(args.w_indices)

    summary = run_degree(
        chern_degree=args.degree,
        output_root=output_root,
        row_indices_arg=row_indices,
        w_indices_arg=w_indices,
        columns_per_shard=args.columns_per_shard,
        p=args.prime,
        backend=args.backend,
        recompute_complete=args.recompute_complete,
        jobs_parallel=args.jobs,
        fresh_process_per_shard=args.fresh_process_per_shard,
        early_stop_full_rank=args.early_stop_full_rank,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
