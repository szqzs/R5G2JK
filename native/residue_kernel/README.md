# Native Residue Kernel

This optional Rust library accelerates the iterated-residue step used by the
pairing-matrix computation.

The Python implementation in `src/r5g2higgs/residue_mod.py` is the readable
reference.  The Rust library exports the same residue operations through
`ctypes`, so long recomputations can use the faster backend.

Build:

```bash
cargo build --release --manifest-path native/residue_kernel/Cargo.toml
```

Use the native backend from Python:

```bash
R5G2HIGGS_RESIDUE_BACKEND=native python scripts/run_single_degree.py --degree 11
```

If the native library is not built, the Python code can still run with the
pure Python backend.
