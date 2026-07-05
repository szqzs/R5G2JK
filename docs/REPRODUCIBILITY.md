# Reproducibility

This file records the reproduction commands for:

1. Quick verification of committed certificates.
2. Recomputing individual degrees.
3. Recomputing the full `c != 12` rank checks.
4. Recomputing the `c = 12` relation line.
5. Verifying the lifted `c = 12` relation over `Q`.
6. Running the Macaulay2 linear-algebra certificate.

The primary reader-facing check is the table in `README.md`, which states the
finite rank assertions without asking the reader to run code.

## Setup

Use Python `3.10` or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

This installs the rank-5 package from `src/`, SymPy, and pytest.  The modular
rank recomputations are fastest with the optional Rust residue kernel.  The
exact rational check over `Q` does not use the Rust kernel.  The Macaulay2
certificate requires Macaulay2.

If you do not want to install the package, run the Python commands below with
`PYTHONPATH=src`.

## Quick Certificate Checks

For a small local smoke test:

```bash
python -m pytest -q
```

The committed result files can already be inspected directly:

```bash
python -m json.tool results/full_rank/full_rank_certificate.json
python -m json.tool results/c12_relation/c12_relation_certificate.json
python -m json.tool results/c12_relation/c12_relation_over_q_certificate.json
python -m json.tool results/MANIFEST.json
```

The smoke tests check:

1. the manifest hashes;
2. `rank = source_dimension` for every listed `c != 12`;
3. the `c = 12` rank/nullity statement;
4. the hash and zero-pairing metadata for the displayed modular relation vector;
5. the exact rational zero-pairing metadata for the lifted integer relation.

The Macaulay2 certificate is:

```bash
M2 --script macaulay2/verify_unipotent_injectivity.m2
```

## Recompute Pairing-Matrix Results

For a small pure-Python recomputation of one full-rank degree:

```bash
python scripts/run_single_degree.py \
  --degree 21 \
  --output-root artifacts/check_c21 \
  --backend python \
  --jobs 1
```

This computes columns until the one source row in Chern degree `21` has rank
`1`, writes `artifacts/check_c21/c21/summary.json`, and stops.

For a small exact rational check of the lifted relation:

```bash
python scripts/verify_c12_relation_over_q.py \
  --limit 1 \
  --output artifacts/c12_relation_over_q_one/verify_one.json
```

This checks the lifted integer relation against one target basis element over
`Q`.  The full exact check below checks all `1039` target basis elements.

Build the optional Rust residue kernel:

```bash
cargo build --release --manifest-path native/residue_kernel/Cargo.toml
```

Recompute all listed `c != 12` full-rank certificates:

```bash
R5G2HIGGS_RESIDUE_BACKEND=native \
python scripts/run_full_blocks.py \
  --degrees 11-22 \
  --exclude-degrees 12 \
  --degree-order small-first \
  --columns-per-shard 4 \
  --output-root artifacts/full_c_ne_12 \
  --backend native \
  --jobs 1 \
  --fresh-process-per-shard
```

For the `c != 12` run, early stopping is on by default.  As soon as the
current Chern degree reaches full row rank, the script records that certificate
and proceeds to the next Chern degree.  It computes all requested columns only
if `--no-early-stop-full-rank` is explicitly passed.

Without Rust, the same command can be run with `--backend python` and without
the `R5G2HIGGS_RESIDUE_BACKEND=native` environment variable.  That path is
slower but uses the readable Python residue implementation.

Recompute the `c = 12` matrix and relation line:

```bash
R5G2HIGGS_RESIDUE_BACKEND=native \
python scripts/run_single_degree.py \
  --degree 12 \
  --output-root artifacts/c12_relation_primary \
  --backend native \
  --jobs 3 \
  --columns-per-shard 4 \
  --fresh-process-per-shard \
  --no-early-stop-full-rank

python scripts/reduce_c12_relation.py \
  --artifact-root artifacts/c12_relation_primary \
  --expected-rank 43
```

Verify the lifted `c = 12` relation over `Q`:

```bash
python scripts/verify_c12_relation_over_q.py \
  --jobs 4 \
  --chunk-size 1 \
  --checkpoint-every 1 \
  --output artifacts/c12_relation_over_q/verify_over_q.json
```

This exact rational check is resumable:

```bash
python scripts/verify_c12_relation_over_q.py \
  --resume \
  --jobs 4 \
  --chunk-size 1 \
  --checkpoint-every 1 \
  --output artifacts/c12_relation_over_q/verify_over_q.json
```
