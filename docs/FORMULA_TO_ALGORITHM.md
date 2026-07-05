# From The JK Formula To The Code

This page is the mathematical bridge between the Jeffrey-Kirwan pairing
formula and the files in this repository.

The computation here is specialized to:

```math
n=5,\qquad d=1,\qquad g=2,\qquad \deg_{\mathrm{coh}}=22.
```

This repository takes the pairing formula as the input formula, specializes it
to this case, and evaluates the resulting finite residues over the finite field

```math
\mathbb F_p,\qquad p=2305843009213693951.
```

The point of this page is to make the specialization explicit enough that a
reader can compare the formula directly with the implementation.

Page references such as "JK p. 59" refer to the printed page numbers in the
arXiv v2 PDF of Jeffrey-Kirwan,
[`alg-geom/9608029`](https://arxiv.org/abs/alg-geom/9608029).

## 1. The General JK Pairing Formula

Let $`M(n,d)`$ be the fixed-determinant moduli space considered by
Jeffrey-Kirwan.  The universal Chern classes give generators
$`a_r,b_r^j,f_r`$ as follows (JK pp. 5-6, especially (2.1)):

```math
a_r,\qquad b_r^j,\qquad f_r
```

where $`2\le r\le n`$ and $`1\le j\le 2g`$.  The $`a_r`$ are point-slant
classes, the $`b_r^j`$ are one-cycle-slant classes, and the $`f_r`$ are
surface-slant classes.

The generating integral has the form appearing on the left side of Theorem
9.12(a) (JK p. 59):

```math
\int_{M(n,d)}
\exp(f_2+\delta_3 f_3+\cdots+\delta_n f_n)
\prod_{r=2}^n a_r^{m_r}
\prod_{r=2}^n\prod_{j=1}^{2g}(b_r^j)^{p_{r,j}}.
```

The formal variables $`\delta_3,\ldots,\delta_n`$ are bookkeeping variables.
They package the higher $`f`$-insertions.  On the residue side, they enter
through the polynomial $`q`$ (JK p. 57, (9.10); the same notation is recalled
in Remark 9.13 on JK p. 60):

```math
q(X)=\tau_2(X)+\delta_3\tau_3(X)+\cdots+\delta_n\tau_n(X).
```

In the formula, $`X`$ is simply a vector of $`n`$ variables

```math
X=(x_1,\ldots,x_n)\in\mathbb C^n
\qquad\text{with}\qquad
x_1+\cdots+x_n=0
```

The sum-zero condition is the fixed-determinant condition on the residue side.
The root system used by the formula is the standard $`A_{n-1}`$ root system,
which in these coordinates just means the differences (JK p. 8):

```math
\alpha_{ij}(X)=x_i-x_j
\qquad
(i\ne j).
```

We choose the positive roots $`\alpha_{ij}`$ with $`i\lt j`$, and the simple roots

```math
\alpha_j(X)=x_j-x_{j+1},
\qquad 1\le j\le n-1.
```

The residue variables are these simple-root coordinates:

```math
Y_j=\alpha_j(X)=x_j-x_{j+1}.
```

With this convention, a positive root is a consecutive sum

```math
\alpha_{ij}(X)=Y_i+Y_{i+1}+\cdots+Y_{j-1}
\qquad (i \lt j),
```

and

```math
D(X)=\prod_{1\le i\lt j\le n}(x_i-x_j)
```

is the product of all positive roots.  This is JK's Weyl odd polynomial,
defined on JK p. 7 by $`D(X)=\prod_{\gamma\gt 0}\gamma(X)`$, where $`\gamma`$
runs over the positive roots.  The same definition is repeated on JK p. 16.
Finally,
$`\tau_r(X)`$ is the $`r`$-th elementary symmetric polynomial in the torus
Chern roots $`x_1,\ldots,x_n`$ (JK p. 6, (2.1), and Remark 9.13 on JK p. 60).

The formula also uses $`n-1`$ scalar expressions

```math
B_1(X,\delta),\ldots,B_{n-1}(X,\delta)
```

defined as follows.  For $`1\le j\le n-1`$,

```math
B_j(X,\delta)
=
\frac{\partial q}{\partial x_{j+1}}(X,\delta)
-
\frac{\partial q}{\partial x_j}(X,\delta).
```

The partial derivatives are taken first in the ordinary polynomial ring with
variables $`x_1,\ldots,x_n`$, and then the result is restricted to
$`x_1+\cdots+x_n=0`$.

Equivalently, let $`\widehat e_j`$ denote the direction that moves $`x_j`$ up
and $`x_{j+1}`$ down.  JK introduce these vectors in Section 6: after defining
the simple roots $`e_j(X)=X_j-X_{j+1}`$, they identify $`t^*`$ with $`t`$ and
write

```math
\widehat e_j=(0,\ldots,0,1,-1,0,\ldots,0)
```

with the $`1`$ in the $`j`$-th position and $`-1`$ in the $`(j+1)`$-st
position (the display after (6.3), JK p. 32).  Thus:

```math
(x_1,\ldots,x_j,\ldots,x_{j+1},\ldots,x_n)
\mapsto
(x_1,\ldots,x_j+t,\ldots,x_{j+1}-t,\ldots,x_n).
```

This direction keeps $`x_1+\cdots+x_n=0`$.  Then $`B_j`$ is minus the
directional derivative of $`q`$ in this direction:

```math
B_j(X,\delta)
=
-
\left.
\frac{d}{dt}
q(x_1,\ldots,x_j+t,x_{j+1}-t,\ldots,x_n;\delta)
\right|_{t=0}.
```

This is exactly JK's notation in Theorem 9.11(a), where they write
$`B(X)_j=-(dq)_X(\widehat e_j)`$ (JK p. 59).

Because

```math
\tau_2(X)=\sum_{i\lt k}x_ix_k=-\frac{1}{2}\sum_i x_i^2
\qquad
\text{when } \sum_i x_i=0,
```

the undeformed part is

```math
B_j(X,0)=Y_j.
```

This is the coordinate form of JK's statement that the $`r=2`$ part gives the
identity map $`B^{(2)}=-d\tau_2=\mathrm{id}`$ (JK p. 62, Remark 10.1).

Thus $`\exp(-B_j(X,\delta))`$ means the ordinary exponential, applied to this
scalar expression.  In the residue calculation it is used as a formal power
series:

```math
\exp(-B_j)=
1-B_j+\frac{B_j^2}{2!}-\frac{B_j^3}{3!}+\cdots.
```

The rank-5 section below rewrites this same formula in the $`Y`$-coordinates
used by the code.

In the notation used by the implementation, the JK formula may be read as
follows.  This is Theorem 9.12(a) of JK (JK p. 59):

```math
\begin{aligned}
&\int_{M(n,d)}
\exp(f_2+\delta_3 f_3+\cdots+\delta_n f_n)
\prod_{r=2}^n a_r^{m_r}
\prod_{r=2}^n\prod_{j=1}^{2g}(b_r^j)^{p_{r,j}}
\\
&\quad =
\frac{(-1)^{n_+(g-1)}}{n!}
\sum_{w\in W_{n-1}}
\mathrm{Res}_{Y_1=0}\cdots
\mathrm{Res}_{Y_{n-1}=0}
\left(
E_{\mathrm{even}}(X,\delta,w)\,
E_{\mathrm{odd}}(X,\delta,\mathbf p)
\right).
\end{aligned}
```

Here $`n_+=n(n-1)/2`$ is the number of positive roots of $`SU(n)`$.

For readability, we split the right side of Theorem 9.12 into an "even factor"
and an "odd factor."  This split is our notation; the full expression is JK's
Theorem 9.12(a) (JK p. 59).  The even factor is

```math
E_{\mathrm{even}}(X,\delta,w)
=
\frac{
\exp\!\left((dq)_X([[\widetilde{w c}]])\right)
\prod_{r=2}^n \tau_r(X)^{m_r}
}{
D(X)^{2g-2}
\prod_{j=1}^{n-1}\left(1-\exp(-B_j(X,\delta))\right)
}.
```

The odd factor is

```math
\begin{aligned}
E_{\mathrm{odd}}(X,\delta,\mathbf p)
&=
\int_{T^{2g}}
\exp\!\left(
-\sum_{\alpha,\beta=1}^{n-1}
\sum_{\ell=1}^{g}
\zeta_\alpha^\ell\zeta_\beta^{\ell+g}
\partial^2 q_X(\widehat u_\alpha,\widehat u_\beta)
\right)
\\
&\qquad\qquad\cdot
\prod_{r=2}^n\prod_{j=1}^{2g}
\left(
\sum_{\alpha=1}^{n-1}
(d\tau_r)_X(\widehat u_\alpha)\zeta_\alpha^j
\right)^{p_{r,j}}.
\end{aligned}
```

The remaining symbols in this formula have the following roles.

- $`Y_1,\ldots,Y_{n-1}`$ are the simple-root residue coordinates just defined.
- $`D(X)`$ is the positive-root product just defined.
- $`B_j(X,\delta)`$ is the deformed simple-root expression just defined.
- $`[[\widetilde{w c}]]`$ is the fixed vector inserted into $`(dq)_X`$ inside
  the exponential factor.  It is determined by the determinant degree $`d`$
  and the permutation $`w`$.  The bracket notation $`[[\cdot]]`$ is defined
  in Definition 2.1 (JK pp. 8-9), and $`\widetilde c`$ is described in
  Remark 2.3 (JK p. 9).
- $`W_{n-1}`$ is the finite permutation group appearing in Jeffrey-Kirwan's
  formula; in the rank-5 specialization used here it permutes the first four
  coordinates.  This subgroup is specified in Proposition 2.2 (JK p. 9).
- $`\widehat u_\alpha`$ are the basis directions in the same sum-zero
  $`X`$-space; the odd part differentiates $`q`$ in these directions.  JK
  introduce these basis directions in the discussion leading to (9.9) (JK
  p. 57) and recall them in Remark 9.13 (JK p. 60).
- The variables $`\zeta_\alpha^j`$ are exterior variables.  The integral over
  $`T^{2g}`$ is an exterior-algebra coefficient extraction.  JK define the
  $`\zeta_\alpha^j`$ in Definition 10.6 (JK p. 65), and Theorem 9.12 uses
  them in the displayed odd factor (JK p. 59).

For computation, the most important point is this:

after one fixes $`n`$, $`g`$, $`d`$, the monomial, and the requested delta
coefficient, the formula is a finite algebraic recipe.

The code evaluates exactly this finite recipe modulo $`p`$.

## 2. Specialization To Rank 5 And Genus 2

From now on,

```math
n=5,\qquad g=2,\qquad d=1.
```

There are four residue variables

```math
Y_1,Y_2,Y_3,Y_4.
```

The trace-zero torus Chern roots are written as

```math
\begin{aligned}
x_1&=\frac{4Y_1+3Y_2+2Y_3+Y_4}{5},\\
x_2&=\frac{-Y_1+3Y_2+2Y_3+Y_4}{5},\\
x_3&=\frac{-Y_1-2Y_2+2Y_3+Y_4}{5},\\
x_4&=\frac{-Y_1-2Y_2-3Y_3+Y_4}{5},\\
x_5&=\frac{-Y_1-2Y_2-3Y_3-4Y_4}{5}.
\end{aligned}
```

These formulas are obtained by solving JK's simple-root equations
$`Y_j=x_j-x_{j+1}`$ together with $`x_1+\cdots+x_5=0`$ (JK p. 8).

Then

```math
\tau_r(Y)=e_r(x_1,x_2,x_3,x_4,x_5),
\qquad 2\le r\le 5.
```

This is the rank-5 specialization of JK's $`\tau_r`$ convention (JK p. 6,
(2.1), and Remark 9.13 on JK p. 60).

The deformation polynomial is

```math
q(Y,\delta)=
\tau_2(Y)+\delta_3\tau_3(Y)+\delta_4\tau_4(Y)+\delta_5\tau_5(Y).
```

This is the rank-5 specialization of (9.10) (JK p. 57).  The polynomials
$`\tau_r(Y)`$ are built in
[`rank5_x_polys`](../src/r5g2higgs/tau_mod.py#L82),
[`rank5_tau`](../src/r5g2higgs/tau_mod.py#L92), and
[`tau_power`](../src/r5g2higgs/tau_mod.py#L108).

Since $`g=2`$, the root-denominator power is

```math
D(Y)^{2g-2}=D(Y)^2.
```

Here $`D(Y)`$ is the product of the ten positive roots

```math
Y_i+Y_{i+1}+\cdots+Y_j
\qquad
1\le i\le j\le 4.
```

This is the rank-5 form of JK's positive-root convention (JK p. 8).

The determinant-degree vector is

```math
\widetilde c
=
\left(\frac{1}{5},\frac{1}{5},\frac{1}{5},\frac{1}{5},-\frac{4}{5}\right).
```

This is the $`n=5,d=1`$ case of JK's formula
$`\widetilde c=[[(d/n,\ldots,d/n,-(n-1)d/n)]]`$ in Remark 2.3 (JK p. 9).

In the $`Y`$-coordinates above, this is the $`Y_4`$-direction.  Indeed, for
$`\widetilde c=(1/5,1/5,1/5,1/5,-4/5)`$, the simple-root coordinates are

```math
Y_1=\frac{1}{5}-\frac{1}{5}=0,\qquad
Y_2=\frac{1}{5}-\frac{1}{5}=0,\qquad
Y_3=\frac{1}{5}-\frac{1}{5}=0,\qquad
Y_4=\frac{1}{5}-\left(-\frac{4}{5}\right)=1.
```

Therefore

```math
(dq)_Y(\widetilde c)
=
\partial_{Y_4}q(Y,\delta).
```

The code splits this exponential into two pieces:

```math
\exp(\partial_{Y_4}\tau_2)
\cdot
\exp\!\left(
\delta_3\partial_{Y_4}\tau_3
+\delta_4\partial_{Y_4}\tau_4
+\delta_5\partial_{Y_4}\tau_5
\right).
```

The first factor is part of the residue engine
([`residue_poly`](../src/r5g2higgs/residue_mod.py#L355)).  The second factor
is expanded in the delta variables by the even-kernel code
([`even_kernel_terms`](../src/r5g2higgs/kernel_mod.py#L308)).
The $`Y_4`$-derivative terms $`\partial_{Y_4}\tau_r`$ are produced by
[`c_direction_term`](../src/r5g2higgs/tau_mod.py#L142).

The four $`B`$-terms are obtained from the explicit derivative formula above:

```math
B_j(Y,\delta)
=
\frac{\partial q}{\partial x_{j+1}}(Y,\delta)
-
\frac{\partial q}{\partial x_j}(Y,\delta).
```

This is the rank-5 coordinate version of JK's definition
$`B(X)_j=-(dq)_X(\widehat e_j)`$ in Theorem 9.11(a) (JK p. 59).
The code implements the $`B_{j,r}`$ pieces in
[`b_perturbation`](../src/r5g2higgs/tau_mod.py#L155), and combines their
Taylor expansion in
[`denominator_taylor_terms`](../src/r5g2higgs/kernel_mod.py#L251).

Equivalently,

```math
B_j(Y,\delta)
=
Y_j
+\delta_3 B_{j,3}(Y)
+\delta_4 B_{j,4}(Y)
+\delta_5 B_{j,5}(Y),
```

where

```math
B_{j,r}(Y)
=
\frac{\partial \tau_r}{\partial x_{j+1}}(Y)
-
\frac{\partial \tau_r}{\partial x_j}(Y).
```

The code does not differentiate in the $`x`$-variables directly.  It first
rewrites everything as a polynomial in $`Y_1,\ldots,Y_4`$; this is done in
[`rank5_x_polys`](../src/r5g2higgs/tau_mod.py#L82),
[`rank5_tau`](../src/r5g2higgs/tau_mod.py#L92), and
[`b_perturbation`](../src/r5g2higgs/tau_mod.py#L155).  In these
$`Y`$-coordinates, the same derivative is computed using the four integer
directions

```math
\begin{aligned}
h_1&=(2,-1,0,0),\\
h_2&=(-1,2,-1,0),\\
h_3&=(0,-1,2,-1),\\
h_4&=(0,0,-1,2).
\end{aligned}
```

Thus, for example,

```math
d\tau_r(h_2)
=
-\frac{\partial\tau_r}{\partial Y_1}
+2\frac{\partial\tau_r}{\partial Y_2}
-\frac{\partial\tau_r}{\partial Y_3}
```

is the derivative in the second direction.  The code uses its negative for
$`B_{2,r}`$:

```math
B_{2,r}(Y)
=
-\left(
-\frac{\partial\tau_r}{\partial Y_1}
+2\frac{\partial\tau_r}{\partial Y_2}
-\frac{\partial\tau_r}{\partial Y_3}
\right).
```

The Hessian part uses

```math
H_q(Y,\delta)=
\left(
\frac{\partial^2 q}{\partial Y_i\partial Y_j}
\right)_{1\le i,j\le 4}.
```

The code uses the normalized determinant

```math
\left(
\frac{\det H_q}{\det H_{\tau_2}}
\right)^2.
```

The square is the genus-two specialization.  The denominator
$`\det H_{\tau_2}`$ removes the coordinate-volume factor coming from the
chosen $`Y`$-basis.

JK define the Hessian determinant in Remark 9.13 (JK p. 60), and discuss the
basis-dependence normalization in Remark 10.1 (JK p. 62).
The Hessians $`\mathrm{Hess}_Y(\tau_r)`$ are built by
[`tau_hessian`](../src/r5g2higgs/tau_mod.py#L127).  The normalized determinant
expansion is computed by
[`det_ratio_delta_power`](../src/r5g2higgs/kernel_mod.py#L213).

## 3. The Formula Actually Evaluated

The repository uses invariant monomials in

```math
a_2,a_3,a_4,a_5,\qquad
f_2,f_3,f_4,f_5,\qquad
\gamma_{rs}\quad(2\le r\le s\le 5).
```

The $`a_r`$ and $`f_r`$ are JK's generators from Section 2 (JK pp. 5-6).
The $`\gamma_{rs}`$ are not a separate JK notation; they are the
$`\mathrm{Sp}(4)`$-invariant combinations of JK's odd generators $`b_r^j`$.

For one invariant exponent vector, the rank-5 genus-2 pairing value is computed
from

```math
5\,
\left(\prod_{r=2}^5 \nu_r!\right)
\,[\delta_3^{\nu_3}\delta_4^{\nu_4}\delta_5^{\nu_5}]
\mathrm{JKRes}
\left(
\frac{
e^{\partial_{Y_4}q}
\prod_{r=2}^5\tau_r(Y)^{m_r}
\left(\frac{\det H_q}{\det H_{\tau_2}}\right)^2
\Gamma_{\mathbf e}(Y,\delta)
}{
D(Y)^2
\prod_{j=1}^4(1-e^{-B_j(Y,\delta)})
}
\right).
```

Here $`\mathrm{JKRes}`$ is our shorthand for the iterated residue
$`\mathrm{Res}_{Y_1=0}\cdots\mathrm{Res}_{Y_4=0}`$ in
Theorem 9.12 (JK p. 59).  It is evaluated by
[`residue_poly`](../src/r5g2higgs/residue_mod.py#L355); the optional native
acceleration is in
[`residue_products_sum_native`](../src/r5g2higgs/residue_backend.py#L224) and
[`native/residue_kernel/`](../native/residue_kernel/).

Here:

- $`m_r`$ is the exponent of $`a_r`$.
- $`\nu_r`$ is the exponent of $`f_r`$.
- $`\mathbf e`$ is the exponent vector of the $`\gamma_{rs}`$.
- $`\Gamma_{\mathbf e}`$ is the already-evaluated odd/gamma contribution.
- The scalar $`5`$ is the collapsed global prefactor for this rank,
  determinant degree, genus, and invariant normalization.  It comes from the
  prefactor and finite permutation sum in Theorem 9.12 (JK p. 59), together
  with the $`W_{n-1}`$ convention from Proposition 2.2 (JK p. 9).
- The factor $`\prod\nu_r!`$ converts the exponential generating function back
  into the ordinary monomial pairing.  The delta target is stored by
  [`InvariantExp.target_delta`](../src/r5g2higgs/model.py#L51), and the
  factorial scale is applied by
  [`f_factorial_scale_mod`](../src/r5g2higgs/arithmetic.py#L43) through
  [`pairing_total_mod_from_prepared`](../src/r5g2higgs/pairing_mod.py#L75).

The $`f_2`$-power does not introduce a new delta variable.  The $`f_2`$ class is
the base exponential direction.  The powers of $`f_3,f_4,f_5`$ are selected by
the displayed delta coefficient.  This is the coefficient-extraction use of
the formal parameters $`\delta_r`$ introduced in (9.10) (JK p. 57) and
recalled in Remark 9.13 (JK p. 60).

The gamma classes are expanded as exterior expressions.  The basic generator is

```math
\gamma_{rs}
=
b_r^1b_s^3-b_r^3b_s^1
+b_r^2b_s^4-b_r^4b_s^2.
```

This is our genus-2 invariant combination of JK's $`b_r^j`$ classes; JK define
the $`b_r^j`$ themselves in Section 2 (JK pp. 5-6).  The expansion of
$`\gamma_{rs}`$ into $`b`$-variables is implemented by
[`gamma_b_terms`](../src/r5g2higgs/reference_sympy.py#L302) and
[`gamma_product_to_b_terms`](../src/r5g2higgs/reference_sympy.py#L315), and
then used by [`gamma_hat`](../src/r5g2higgs/gamma_mod.py#L169).

After expanding a product of $`\gamma`$'s into $`b`$-variables, the exterior
integral gives pair coefficients

```math
T_{rs}(Y,\delta)
=
-\nabla_Y\tau_r(Y)^T
H_q(Y,\delta)^{-1}
\nabla_Y\tau_s(Y).
```

Thus $`\Gamma_{\mathbf e}`$ is not an extra theorem.  It is just the finite
exterior-algebra evaluation of the $`E_{\mathrm{odd}}`$ factor in the JK
formula.  The corresponding JK odd-class restrictions are Lemma 10.12 (JK
p. 69), and the resulting contraction formula is Lemma 10.13 (JK pp. 69-70).
The pair coefficient is computed by
[`hat_pair_delta`](../src/r5g2higgs/kernel_mod.py#L179), and the full
$`\Gamma_{\mathbf e}`$ contribution is assembled by
[`gamma_hat`](../src/r5g2higgs/gamma_mod.py#L169).

## 4. Which File Computes Which Part

The main exponent data type is in
[`InvariantExp`](../src/r5g2higgs/model.py#L31).

It stores one monomial by the exponent tuple

```math
(a_2,a_3,a_4,a_5),\qquad
(f_2,f_3,f_4,f_5),\qquad
(\gamma_{22},\gamma_{23},\ldots,\gamma_{55}).
```

The $`Y`$-coordinates and the $`\tau_r`$'s are in
[`rank5_tau`](../src/r5g2higgs/tau_mod.py#L92).

This file implements:

- the five $`x_i(Y)`$;
- the polynomials $`\tau_2,\tau_3,\tau_4,\tau_5`$;
- gradients and Hessians of the $`\tau_r`$'s;
- the $`\partial_{Y_4}\tau_r`$ terms;
- the $`B_{j,r}=-d\tau_r(h_j)`$ terms.

The delta arithmetic is in
[`src/r5g2higgs/delta_mod.py`](../src/r5g2higgs/delta_mod.py#L1).

It implements truncated polynomial arithmetic in
$`\delta_3,\delta_4,\delta_5`$.  The truncation is exact because the code only
needs one coefficient

```math
[\delta_3^{\nu_3}\delta_4^{\nu_4}\delta_5^{\nu_5}].
```

The even JK kernel is in
[`even_kernel_terms`](../src/r5g2higgs/kernel_mod.py#L308).

This file computes the delta expansion of:

- the higher part of $`e^{\partial_{Y_4}q}`$;
- $`(\det H_q/\det H_{\tau_2})^2`$;
- the perturbation of the factors $`1-e^{-B_j}`$;
- the pair coefficients $`T_{rs}`$ needed by the odd part.

The gamma/odd factor is in
[`gamma_hat`](../src/r5g2higgs/gamma_mod.py#L169).

It computes $`\Gamma_{\mathbf e}`$ by:

1. expanding the $`\gamma_{rs}`$'s into exterior $`b_r^j`$'s;
2. multiplying in the exterior algebra;
3. applying the pair coefficient $`T_{rs}`$;
4. returning the result as a delta/Y-polynomial.

The iterated residue is in
[`residue_poly`](../src/r5g2higgs/residue_mod.py#L355).

This file evaluates the rank-5 residue in the order

$`Y_4`$, then $`Y_3`$, then $`Y_2`$, then $`Y_1`$.

It also incorporates the base exponential
$`e^{\partial_{Y_4}\tau_2}`$ and the root denominator $`D(Y)^2`$.

The optional Rust acceleration is in
[`native/residue_kernel/`](../native/residue_kernel/).

It computes the same product-residue operation faster.  It is not a different
formula.

One complete pairing entry is assembled in
[`pairing_total_mod_from_prepared`](../src/r5g2higgs/pairing_mod.py#L75).

For one row basis element and one column basis element, the code:

1. adds their exponent vectors;
2. builds the $`a`$-part $`\prod\tau_r^{m_r}`$;
3. selects the delta coefficient from the $`f_3,f_4,f_5`$ powers;
4. multiplies the even kernel and gamma contribution;
5. takes the residue;
6. multiplies by the global scalar and the $`f`$-factorials.

One matrix entry is computed by
[`pairing_entry`](../src/r5g2higgs/matrix_mod.py#L152).

This function chooses a source basis element and a target basis element, adds
their exponent vectors, and sends the result to `pairing_mod.py`.

The finite-field rank calculation is
[`rank_mod_p`](../src/r5g2higgs/matrix_mod.py#L372).

The basis enumeration is in
[`independent_basis_by_chern`](../src/r5g2higgs/basis_mod.py#L386).

It does not evaluate residues.  It only enumerates invariant monomials and
chooses independent basis elements after expanding the $`\gamma`$'s into the
exterior algebra.

## 5. From Pairing Values To Certificates

The full-row-rank checks for $`c\ne 12`$ are run by
[`run_degree`](../scripts/run_full_blocks.py#L163) in
[`scripts/run_full_blocks.py`](../scripts/run_full_blocks.py#L163).

This script computes columns until full row rank is reached.  Once a Chern
degree has full row rank, the script stops that degree and moves on.  It does
not need the remaining columns.

The exceptional Chern degree $`c=12`$ is run by
[`main`](../scripts/run_single_degree.py#L15) in
[`scripts/run_single_degree.py`](../scripts/run_single_degree.py#L15).

For $`c=12`$, the full target side is computed because the goal is not merely
to see that a partial matrix has large rank.  The goal is to identify the
one-dimensional relation line.

After the $`c=12`$ matrix has been computed, the relation vector is extracted
and checked by
[`reduce_relation`](../scripts/reduce_c12_relation.py#L92) in
[`scripts/reduce_c12_relation.py`](../scripts/reduce_c12_relation.py#L92).

The lifted integer relation is checked over $`\mathbb Q`$ by
[`relation_dot_q`](../src/r5g2higgs/relation_q.py#L108), one target basis
element at a time.  The all-column checkpointing command is implemented by
[`verify_indices`](../scripts/verify_c12_relation_over_q.py#L105) in
[`scripts/verify_c12_relation_over_q.py`](../scripts/verify_c12_relation_over_q.py#L105).

The result files and reproduction commands are listed in the root `README.md`
and in `docs/REPRODUCIBILITY.md`.  This page only identifies the rank-5
formula and the code paths that evaluate it.
