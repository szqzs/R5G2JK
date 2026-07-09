# Rank5Genus2Relation

This repository records the computer-assisted part of a rank-five,
fixed-determinant, determinant-degree-one, genus-two calculation.  The first
relation studied here lies in cohomological degree `22`.

## Setup

The Python code requires Python `3.10` or newer.  The only runtime Python
dependency is SymPy; pytest is used for the small checks.

From a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

The optional Rust residue kernel is used only to speed up long modular
recomputations.  The exact rational `Q` check uses the Python code.  The
Macaulay2 script requires a separate Macaulay2 installation.

## Results

We have two kinds of results below.

First, for every Chern degree `c != 12` listed below, we calculate the rank of
the pairing matrix modulo `p`, where

```math
p=2305843009213693951=2^{61}-1.
```

We find that all these matrices have full row rank modulo `p`.  The matrices
come from rational/integer formulas before reduction modulo `p`, so a nonzero
minor modulo `p` is a nonzero rational minor.  Therefore the original pairing
matrices over `Q` also have full row rank.  Equivalently, there are no
homogeneous relations in these Chern degrees.

Second, for `c = 12`, we calculate the pairing matrix over $`\mathbb F_p`$
and find nullity `1`.  This gives an explicit generator for the
one-dimensional relation line over $`\mathbb F_p`$.  We then lift this vector
to an integer vector and check it over `Q`: multiplying the lifted vector by
the rational pairing matrix gives the zero row vector.  Therefore the lifted
vector is genuinely a relation over `Q`.

For the following Chern degrees, the pairing matrix has full row rank.

| Chern degree `c` | Source dimension | Rank mod `p` | Nullity over `Q` | Conclusion |
|---:|---:|---:|---:|---|
| 11 | 7 | 7 | 0 | Full row rank |
| 13 | 94 | 94 | 0 | Full row rank |
| 14 | 111 | 111 | 0 | Full row rank |
| 15 | 81 | 81 | 0 | Full row rank |
| 16 | 53 | 53 | 0 | Full row rank |
| 17 | 28 | 28 | 0 | Full row rank |
| 18 | 16 | 16 | 0 | Full row rank |
| 19 | 7 | 7 | 0 | Full row rank |
| 20 | 4 | 4 | 0 | Full row rank |
| 21 | 1 | 1 | 0 | Full row rank |
| 22 | 1 | 1 | 0 | Full row rank |

The exceptional degree is `c = 12`.

| Chern degree `c` | Source dimension | Rank mod `p` | Nullity over `Q` | Conclusion |
|---:|---:|---:|---:|---|
| 12 | 44 | 43 | 1 | One-dimensional relation line |

The modular `c = 12` certificate records the rank/nullity calculation and the
relation vector modulo `p`.  The exact rational certificate records that the
lifted integer vector pairs to zero with all `1039` target columns over `Q`.

## The Relation

The notation below is the notation used throughout the repository.  The
generators $`a_r`$ and $`f_r`$ are the even generators, and $`\gamma_{rs}`$
denotes the $`\mathrm{Sp}(4)`$-invariant combination of the odd generators used
in these formulas.

The `c = 12` relation is:

```math
\begin{aligned}
{}&
32a_2^5f_2
-48a_2^4f_4
-16a_2^4\gamma_{22}
-208a_2^3a_3f_3
-192a_2^3a_4f_2
+4a_2^3\gamma_{33}\\
&-312a_2^2a_3^2f_2
+240a_2^2a_3f_5
+92a_2^2a_3\gamma_{23}
+256a_2^2a_4f_4\\
&+96a_2^2a_4\gamma_{22}
+240a_2^2a_5f_3
+40a_2^2\gamma_{35}
+16a_2^2\gamma_{44}\\
&+288a_2a_3^2f_4
+87a_2a_3^2\gamma_{22}
+576a_2a_3a_4f_3
+480a_2a_3a_5f_2\\
&+60a_2a_3\gamma_{25}
+12a_2a_3\gamma_{34}
+256a_2a_4^2f_2
-64a_2a_4\gamma_{24}\\
&-48a_2a_4\gamma_{33}
-250a_2a_5f_5
-220a_2a_5\gamma_{23}
-35a_2\gamma_{55}\\
&+108a_3^3f_3
+288a_3^2a_4f_2
-18a_3^2\gamma_{24}
-18a_3^2\gamma_{33}\\
&-450a_3a_4f_5
-168a_3a_4\gamma_{23}
-450a_3a_5f_4
-150a_3a_5\gamma_{22}\\
&-90a_3\gamma_{45}
-255a_4^2f_4
-64a_4^2\gamma_{22}
-450a_4a_5f_3\\
&-30a_4\gamma_{35}
-125a_5^2f_2
+50a_5\gamma_{25}
+150a_5\gamma_{34}.
\end{aligned}
```

The modular `c = 12` line is extracted by
[`reduce_relation`](scripts/reduce_c12_relation.py#L92).  The lifted integer
vector is stored as
[`C12_RELATION_INTEGER_VECTOR`](src/r5g2higgs/relation_q.py#L19), and exact
zero-pairing over `Q` is checked target-by-target by
[`relation_dot_q`](src/r5g2higgs/relation_q.py#L108).

### Chern-Character Convention

The relation above is written in the Jeffrey--Kirwan generators obtained from
the Chern classes of the universal bundle.  If we instead use the
Künneth components of the normalized Chern characters, written as

```math
\mathrm{ch}_r
=
1_C\otimes\beta_r
+\sum_{j=1}^4\delta_j\otimes\psi_{r,j}
+\mathrm{pt}_C\otimes\alpha_r,
```

together with the symplectic basis whose intersection matrix is
$`\left(\begin{smallmatrix}0&I_2\\-I_2&0\end{smallmatrix}\right)`$ and set

```math
\gamma_{rs}
=
\psi_{r,1}\psi_{s,3}-\psi_{r,3}\psi_{s,1}
+\psi_{r,2}\psi_{s,4}-\psi_{r,4}\psi_{s,2},
```

then, with $`\mathrm{ch}_1=0`$, replacing the Chern-class generators by these
Chern-character generators gives

```math
\begin{aligned}
a_2&=-\beta_2,
&b_2^j&=-\psi_{2,j},
&f_2&=-\alpha_2,\\
a_3&=2\beta_3,
&b_3^j&=2\psi_{3,j},
&f_3&=2\alpha_3,\\
a_4&=\frac{1}{2}\beta_2^2-6\beta_4,
&b_4^j&=\beta_2\psi_{2,j}-6\psi_{4,j},
&f_4&=\beta_2\alpha_2-\frac{1}{2}\gamma_{22}-6\alpha_4,\\
a_5&=24\beta_5-2\beta_2\beta_3,
&b_5^j&=24\psi_{5,j}-2\beta_2\psi_{3,j}-2\beta_3\psi_{2,j},
&f_5&=24\alpha_5-2\beta_2\alpha_3-2\beta_3\alpha_2+2\gamma_{23}.
\end{aligned}
```

Here the $`\gamma_{rs}`$ occurring in the original relation are first expanded
in the $`b_r^j`$ and then transformed using the formulas above.  Equivalently,
if $`\Gamma_{rs}`$ denotes the contraction formed from the Chern-class
generators $`b_r^j`$, then

```math
\begin{aligned}
\Gamma_{22}&=\gamma_{22},
&\Gamma_{23}&=-2\gamma_{23},\\
\Gamma_{24}&=-\beta_2\gamma_{22}+6\gamma_{24},
&\Gamma_{25}&=2\beta_3\gamma_{22}+2\beta_2\gamma_{23}-24\gamma_{25},\\
\Gamma_{33}&=4\gamma_{33},
&\Gamma_{34}&=2\beta_2\gamma_{23}-12\gamma_{34},\\
\Gamma_{35}&=-4\beta_3\gamma_{23}-4\beta_2\gamma_{33}+48\gamma_{35},\\
\Gamma_{44}&=\beta_2^2\gamma_{22}-12\beta_2\gamma_{24}+36\gamma_{44},\\
\Gamma_{45}&=-2\beta_2\beta_3\gamma_{22}-2\beta_2^2\gamma_{23}
+24\beta_2\gamma_{25}+12\beta_3\gamma_{24}
+12\beta_2\gamma_{34}-144\gamma_{45},\\
\Gamma_{55}&=4\beta_3^2\gamma_{22}+8\beta_2\beta_3\gamma_{23}
-96\beta_3\gamma_{25}+4\beta_2^2\gamma_{33}
-96\beta_2\gamma_{35}+576\gamma_{55}.
\end{aligned}
```

After this change of generators, multiplying the transformed relation by `8`
gives the following primitive integral relation in the notation of the paper:

```math
\begin{aligned}
\Upsilon_{2,5}^{\mathrm{ch}}={}&
130\alpha_2\beta_2^5
-3120\alpha_2\beta_2^3\beta_4
+6720\alpha_2\beta_2^2\beta_3^2
-180480\alpha_2\beta_2\beta_3\beta_5\\
&+288\alpha_2\beta_2\beta_4^2
-31104\alpha_2\beta_3^2\beta_4
+576000\alpha_2\beta_5^2\\
&+4480\alpha_3\beta_2^3\beta_3
-90240\alpha_3\beta_2^2\beta_5
-62208\alpha_3\beta_2\beta_3\beta_4\\
&+13824\alpha_3\beta_3^3
+1036800\alpha_3\beta_4\beta_5\\
&-780\alpha_4\beta_2^4
+288\alpha_4\beta_2^2\beta_4
-31104\alpha_4\beta_2\beta_3^2\\
&+1036800\alpha_4\beta_3\beta_5
+440640\alpha_4\beta_4^2\\
&-90240\alpha_5\beta_2^2\beta_3
+1152000\alpha_5\beta_2\beta_5
+1036800\alpha_5\beta_3\beta_4\\
&-65\beta_2^4\gamma_{22}
+960\beta_2^3\gamma_{33}
-5120\beta_2^2\beta_3\gamma_{23}
+1560\beta_2^2\beta_4\gamma_{22}\\
&-17280\beta_2^2\gamma_{35}
+4608\beta_2^2\gamma_{44}
+480\beta_2\beta_3^2\gamma_{22}
-19200\beta_2\beta_3\gamma_{25}\\
&+13824\beta_2\beta_3\gamma_{34}
-18432\beta_2\beta_4\gamma_{24}
-14976\beta_2\beta_4\gamma_{33}\\
&+88320\beta_2\beta_5\gamma_{23}
+161280\beta_2\gamma_{55}
-20736\beta_3^2\gamma_{24}
-2304\beta_3^2\gamma_{33}\\
&+48384\beta_3\beta_4\gamma_{23}
+48000\beta_3\beta_5\gamma_{22}
+207360\beta_3\gamma_{45}\\
&+18288\beta_4^2\gamma_{22}
+69120\beta_4\gamma_{35}
-230400\beta_5\gamma_{25}
-345600\beta_5\gamma_{34}.
\end{aligned}
```

## Result Files

The compact committed result files are:

```text
results/full_rank/full_rank_certificate.json
results/full_rank/by_degree/c11.json
results/full_rank/by_degree/c13.json
results/full_rank/by_degree/c14.json
results/full_rank/by_degree/c15.json
results/full_rank/by_degree/c16.json
results/full_rank/by_degree/c17.json
results/full_rank/by_degree/c18.json
results/full_rank/by_degree/c19.json
results/full_rank/by_degree/c20.json
results/full_rank/by_degree/c21.json
results/full_rank/by_degree/c22.json
results/c12_relation/c12_relation_certificate.json
results/c12_relation/c12_relation_over_q_certificate.json
results/MANIFEST.json
```

`results/MANIFEST.json` gives SHA-256 hashes for the committed certificate
files.

## Code At A Glance

The repository is organized as follows.

```text
src/         implementation of the pairing formula and matrix entries
scripts/     commands for reproducing rank and relation calculations
results/     compact committed result certificates
macaulay2/   Macaulay2 code for the stabilizer-dimension check
native/      optional Rust acceleration for the residue step
docs/        optional longer explanations
tests/       small consistency checks
```

The main Python package is `src/r5g2higgs/`.

The mathematical bridge from the Jeffrey-Kirwan formula to these files is
`docs/FORMULA_TO_ALGORITHM.md`.  A file-by-file explanation is in
`docs/CODE_GUIDE.md`.

```text
src/r5g2higgs/basis_mod.py        basis enumeration
src/r5g2higgs/tau_mod.py          tau polynomials and derivatives
src/r5g2higgs/kernel_mod.py       even JK kernel
src/r5g2higgs/gamma_mod.py        odd/gamma contraction
src/r5g2higgs/residue_mod.py      readable Python residue evaluator
src/r5g2higgs/pairing_mod.py      one modular pairing value
src/r5g2higgs/matrix_mod.py       matrix rows, columns, and ranks
src/r5g2higgs/linear_mod.py       finite-field linear algebra
src/r5g2higgs/relation_q.py       exact-Q check of the lifted relation
```

The pairing-matrix code does two things:

1. construct basis elements in each Chern degree;
2. evaluate the Jeffrey-Kirwan pairing formula modulo `p` to build the
   relevant matrix entries.

The rank computation then checks:

- full row rank for `c != 12`;
- rank `43` and nullity `1` for `c = 12`;
- the displayed `c = 12` relation vector pairs to zero with all target
  columns modulo `p`;
- the lifted integer `c = 12` relation vector pairs to zero with all target
  columns over `Q`.

The optional Rust kernel is in:

```text
native/residue_kernel/
```

It accelerates the residue step used inside `pairing_mod.py`.  The Python
residue code remains in the repository as the readable reference.

## Reproducing The Results

First, run the small standalone checks:

```bash
python -m pytest -q
```

To inspect the committed certificates directly:

```bash
python -m json.tool results/full_rank/full_rank_certificate.json
python -m json.tool results/c12_relation/c12_relation_certificate.json
python -m json.tool results/c12_relation/c12_relation_over_q_certificate.json
python -m json.tool results/MANIFEST.json
```

To build the optional Rust residue kernel:

```bash
cargo build --release --manifest-path native/residue_kernel/Cargo.toml
```

To recompute the `c != 12` full-rank certificates:

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

For these `c != 12` runs, early stopping is on by default.  The script records
the current degree as soon as full row rank is reached, then moves on to the
next Chern degree.  It computes unnecessary remaining columns only if
`--no-early-stop-full-rank` is explicitly passed.

To recompute the `c = 12` matrix and relation line:

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

To verify the lifted `c = 12` relation over `Q`:

```bash
python scripts/verify_c12_relation_over_q.py \
  --jobs 4 \
  --chunk-size 1 \
  --checkpoint-every 1 \
  --output artifacts/c12_relation_over_q/verify_over_q.json
```

This is an exact rational calculation.  It may take a while; `--resume` can be
added to continue from an existing checkpoint.

The commands above assume the editable install from the setup section.  If you
prefer not to install the package, prefix the Python commands with
`PYTHONPATH=src`.

For the Macaulay2 part, the intended command is:

```bash
M2 --script macaulay2/verify_stabilizer_lie_dimension.m2
```

## Optional Longer Explanations

The main facts are already on this page.  The files under `docs/` give more
detail for readers who want it:

```text
docs/FORMULA_TO_ALGORITHM.md  formula, specialization, and code links
docs/CODE_GUIDE.md             file-by-file guide to the implementation
docs/REPRODUCIBILITY.md        commands for rerunning the checks
```
