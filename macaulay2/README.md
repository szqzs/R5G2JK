# Macaulay2 Certificate

This folder contains the Macaulay2 code for the stabilizer/minimal-relation
linear algebra associated with the displayed rank-5 relation.

It is separate from the Jeffrey-Kirwan pairing calculation.  The pairing code
finds the rank results and the `c = 12` relation line.  The Macaulay2 code
checks a finite algebra assertion about the resulting relation.

## Certificate

```bash
M2 --script macaulay2/verify_unipotent_injectivity.m2
```

This verifies that the rational linear map

```math
\Phi:\mathrm{Lie}(U)\oplus \mathbb Q\kappa\longrightarrow F^{22},
\qquad
(\delta,\kappa)\longmapsto \delta(R_5)-\kappa R_5
```

is injective over `Q`.

Successful output ends with:

```text
Macaulay2 certificate passed: Phi is injective over QQ.
```
