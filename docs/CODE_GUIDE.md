# Code Guide

This guide explains what the code does, file by file.  It is written for
readers who want a direct map from the formulas to the implementation.

The short version:

```text
basis elements
  -> pairing formula
  -> matrix columns
  -> finite-field rank
  -> certificates
```

The separate Macaulay2 file checks an exact rational linear-algebra statement
about the line spanned by the `c = 12` relation.

## Main Data Model

### `src/r5g2higgs/model.py`

This file defines `InvariantExp`.

An `InvariantExp` records the exponents of a monomial in the invariant
generators:

```math
a_2,a_3,a_4,a_5,\qquad
f_2,f_3,f_4,f_5,\qquad
\gamma_{22},\gamma_{23},\ldots,\gamma_{55}.
```

For example, an expression such as

```math
a_2^3 f_4\gamma_{23}
```

is stored as three exponent tuples:

| Tuple | Stored exponents |
|---|---|
| `a` | `(3,0,0,0)` |
| `f` | `(0,0,1,0)` |
| `gamma` | one `1` in the $`\gamma_{23}`$ position, zeros elsewhere |

This object is intentionally small.  Most of the code passes around
`InvariantExp` objects rather than symbolic strings.

### `src/r5g2higgs/constants.py`

This file fixes the global conventions:

```math
n=5,\qquad g=2,\qquad
\deg_{\mathrm{source}}=22,\qquad
\deg_{\mathrm{target}}=26,\qquad
p=2305843009213693951.
```

It also records the ordered lists of `b` labels and `gamma` labels.  The order
matters because vectors and bitmasks use these lists as coordinates.

## Basis Enumeration

### `src/r5g2higgs/basis_mod.py`

This file builds the invariant bases used for the pairing matrices.

It starts with the formal generators:

```math
a_2,a_3,a_4,a_5,\qquad
f_2,f_3,f_4,f_5,\qquad
\gamma_{rs}.
```

Then it enumerates all products of the desired ordinary degree.

The subtle point is linear dependence.  The `gamma_rs` are not independent
formal variables in the raw exterior algebra.  Each `gamma_rs` expands into
the odd generators `b_r^j`.  The code expands products into the raw
super-commutative monomial basis and row-reduces them over `Q`.  The survivors
form the invariant basis.

The main public functions are:

```text
independent_invariant_basis(rank, ordinary_degree)
independent_basis_by_chern(rank, ordinary_degree, chern_degrees)
```

For this computation they produce:

| Basis piece | Dimension |
|---|---:|
| Target basis in degree $`26`$ | 1039 |
| Source basis, $`c=11`$ | 7 |
| Source basis, $`c=12`$ | 44 |
| Source basis, $`c=13`$ | 94 |
| Source basis, $`c=14`$ | 111 |
| Source basis, $`c=15`$ | 81 |
| Source basis, $`c=16`$ | 53 |
| Source basis, $`c=17`$ | 28 |
| Source basis, $`c=18`$ | 16 |
| Source basis, $`c=19`$ | 7 |
| Source basis, $`c=20`$ | 4 |
| Source basis, $`c=21`$ | 1 |
| Source basis, $`c=22`$ | 1 |

### `src/r5g2higgs/basis_support.py`

This is a small helper file for `basis_mod.py`.

It supplies:

- the ordered `b` labels;
- the ordered `gamma` labels;
- bitmasks for exterior monomials;
- wedge multiplication signs;
- expansion of `gamma` products into `b` products.

Keeping these shared helpers separate makes `basis_mod.py` easier to audit.

## Polynomial Arithmetic

### `src/r5g2higgs/sparse_mod.py`

This file implements sparse polynomials in four variables

```math
Y_1,Y_2,Y_3,Y_4.
```

A monomial is stored by its exponent tuple:

```math
Y_1^{e_1}Y_2^{e_2}Y_3^{e_3}Y_4^{e_4}
\longleftrightarrow
(e_1,e_2,e_3,e_4).
```

A polynomial is a dictionary:

```text
(a,b,c,d) -> coefficient mod p
```

The file provides addition, multiplication, powers, derivatives, and linear
forms.  It is deliberately elementary because these operations occur many
times in the pairing calculation.

### `src/r5g2higgs/arithmetic.py`

This file contains small finite-field utilities:

- reduction of rational numbers modulo `p`;
- factorial scaling used by the `f` classes;
- modular inverses.

## The Even JK Formula

### `src/r5g2higgs/tau_mod.py`

This file builds the rank-5 `tau_r` polynomials modulo `p`.

The variables are the `Y` coordinates used for the residue calculation.  The
code computes:

```math
\tau_2,\tau_3,\tau_4,\tau_5
```

and their gradients and Hessians in the `Y` variables.

It also computes the pieces that appear in the deformed expression

```math
q=\tau_2+\delta_3\tau_3+\delta_4\tau_4+\delta_5\tau_5.
```

### `src/r5g2higgs/delta_mod.py`

This file implements truncated polynomials in the formal variables

```math
\delta_3,\delta_4,\delta_5.
```

These variables are not geometric classes by themselves in the final answer.
They are bookkeeping variables used to extract coefficients corresponding to
powers of the `f3`, `f4`, and `f5` classes.

### `src/r5g2higgs/kernel_mod.py`

This file builds the even part of the Jeffrey-Kirwan integrand.

It computes:

- the Hessian of `q`;
- the inverse Hessian as a truncated delta expansion;
- the determinant ratio;
- the `B_j` denominator perturbations;
- the even kernel terms needed by the residue.

Conceptually, this file is where the even classes `a_r` and `f_r` become the
finite expression later fed to the residue routine.

## The Odd/Gamma Part

### `src/r5g2higgs/gamma_mod.py`

This file handles the odd classes through the invariant combinations
`gamma_rs`.

It does two things:

1. expand products of `gamma_rs` into exterior monomials in the `b_r^j`;
2. replace the exterior integral over the odd variables by the finite
   contraction factors determined by the Hessian of `q`.

In practice, it produces the `gamma` contribution as another truncated
delta-polynomial in the `Y` variables.

## Residues

### `src/r5g2higgs/residue_mod.py`

This is the readable Python implementation of the iterated residue.

It evaluates the coefficient prescribed by the Jeffrey-Kirwan residue in the
rank-5 coordinates.  The implementation is specialized to the fixed rank-5
problem, which keeps the code faster and simpler than a fully general symbolic
residue engine.

The two important entry points are:

```text
residue_poly_batch(poly, deriv_orders, p)
residue_poly_termwise(poly, deriv_orders, p)
```

The smoke tests compare batch and termwise evaluation on sample inputs.

### `src/r5g2higgs/residue_backend.py`

This file chooses between:

```text
pure Python residue code
optional Rust residue library
```

The environment variable

```text
R5G2HIGGS_RESIDUE_BACKEND=native
```

selects the Rust backend when the Rust library has been built.

### `native/residue_kernel/`

This is the optional Rust acceleration for the residue step.

It does not define a different mathematical algorithm.  It implements the same
residue operations more quickly, mainly by avoiding large intermediate Python
dictionaries during product-residue computations.

Build it with:

```bash
cargo build --release --manifest-path native/residue_kernel/Cargo.toml
```

## Pairing Values

### `src/r5g2higgs/pairing_mod.py`

This file computes one pairing value.

Input:

```text
InvariantExp
```

Output:

an element of $`\mathbb F_p`$.

The computation combines:

1. the `a` powers through `tau_power`;
2. the `f` powers through coefficient extraction in the delta variables;
3. the `gamma` powers through the odd contraction layer;
4. the even kernel;
5. the iterated residue;
6. the fixed scalar prefactor.

This is the main formula-evaluation file.

## Matrices And Ranks

### `src/r5g2higgs/matrix_mod.py`

This file turns basis elements into matrix entries.

For a source basis element `u` and a target basis element `v`, it forms the
total exponent

$`u\cdot v`$

and sends that exponent to `pairing_mod.py`.

It also contains the basic finite-field row-reduction function `rank_mod_p`.

### `src/r5g2higgs/column_shard.py`

Long computations write columns in small JSON shards.  A shard records:

- Chern degree;
- source rows;
- target column indices;
- prime;
- computed column values.

The shard format lets long runs resume without recomputing already completed
columns.

### `src/r5g2higgs/shard_plan.py`

This file decides which column shards should be computed, reused, or skipped.

It also checks whether an existing shard matches the requested degree, rows,
columns, and prime.

### `src/r5g2higgs/linear_mod.py`

This file contains finite-field linear algebra used after matrix columns have
been assembled.

It is especially used for `c = 12`, where the matrix has rank `43` on a
44-dimensional source.  The code selects an invertible minor and constructs a
left-kernel vector, which is the relation vector recorded in the certificate.

## Command-Line Scripts

### `scripts/run_full_blocks.py`

This script recomputes the full-rank certificates for `c != 12`.

Important behavior:

```text
early stopping is on by default
```

As soon as the currently computed columns have full row rank for one Chern
degree, the script records that degree as complete and moves to the next Chern
degree.

### `scripts/run_single_degree.py`

This script recomputes one Chern-degree matrix.

For `c = 12`, it is run with

```text
--no-early-stop-full-rank
```

because the relation certificate checks all target columns.

### `scripts/reduce_c12_relation.py`

This script reads the completed `c = 12` column shards, computes the rank, and
extracts the one-dimensional relation line.

It records:

- rank;
- source nullity;
- selected minor;
- relation vector;
- zero pairing against all target columns.

## Tests

### `tests/test_standalone_smoke.py`

These are small checks that the repository is internally coherent.

They check:

- sparse polynomial operations;
- `tau` polynomials against a SymPy reference;
- residue batch evaluation against termwise evaluation;
- exterior signs for the odd variables;
- source and target basis dimensions;
- fixed sample pairing values;
- basic shape of the committed certificates.

Run:

```bash
PYTHONPATH=src pytest -q
```

## Macaulay2

### `macaulay2/verify_stabilizer_lie_dimension.m2`

This is the only Macaulay2 script in the repository.

It checks an exact rational linear-algebra assertion about the displayed
relation.  The script builds the graded super-commutative algebra `A`, writes
the relation `R5`, lists all degree-preserving derivations of `A`, applies each
one to `R5`, and computes the rank of the map

```math
\rho_R:\mathrm{Der}^{\mathrm{gr}}(A)\longrightarrow A^{22}/\mathbb Q R_5,
\qquad
D\longmapsto [D(R_5)].
```

The quotient is built explicitly.  The script chooses a monomial appearing in
`R5`, uses the equation `R5 = 0` to eliminate that monomial, and records all
remaining degree-`22` coordinates.  It then verifies

```math
\mathrm{dim}_{\mathbb Q}\ker(\rho_R)=12.
```

Run:

```bash
M2 --script macaulay2/verify_stabilizer_lie_dimension.m2
```

Successful output ends with:

```text
Macaulay2 certificate passed: stabilizer Lie algebra dimension is 12 over QQ.
```
