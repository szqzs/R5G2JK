-- Direct Macaulay2 certificate for the rank-5 stabilizer Lie algebra.
--
-- This script verifies the following finite linear-algebra assertion:
--
--   rho_R : Der^gr(A) -> A^22 / Q*R5,
--           D |-> [D(R5)]
--
-- has kernel dimension 12 over Q.
--
-- The algorithm is intentionally literal:
--
--   1. write down the free graded super-commutative algebra A;
--   2. write down the relation R5;
--   3. list all degree-preserving derivations of A;
--   4. apply each derivation to R5;
--   5. reduce the result modulo the line Q*R5;
--   6. compute the rank of the resulting matrix.

A = QQ[
  f2,a2,f3,a3,f4,a4,f5,a5,
  b21,b22,b23,b24,
  b31,b32,b33,b34,
  b41,b42,b43,b44,
  b51,b52,b53,b54,
  t,
  Degrees => {
    2,4,4,6,6,8,8,10,
    3,3,3,3,
    5,5,5,5,
    7,7,7,7,
    9,9,9,9,
    1000
  },
  SkewCommutative => {
    b21,b22,b23,b24,
    b31,b32,b33,b34,
    b41,b42,b43,b44,
    b51,b52,b53,b54
  }
]

evenGenerators = {f2,a2,f3,a3,f4,a4,f5,a5}
oddGenerators = {
  b21,b22,b23,b24,
  b31,b32,b33,b34,
  b41,b42,b43,b44,
  b51,b52,b53,b54
}
genVars = evenGenerators | oddGenerators
generatorDegrees = {
  2,4,4,6,6,8,8,10,
  3,3,3,3,
  5,5,5,5,
  7,7,7,7,
  9,9,9,9
}

bvar = (i,k) -> (
  base := if i == 2 then 0 else if i == 3 then 4 else if i == 4 then 8 else 12;
  oddGenerators#(base + k - 1)
)

gamma = (i,j) -> (
    bvar(i,1)*bvar(j,3)
  - bvar(i,3)*bvar(j,1)
  + bvar(i,2)*bvar(j,4)
  - bvar(i,4)*bvar(j,2)
)

R5 = (
    32*a2^5*f2
  - 48*a2^4*f4
  - 16*a2^4*gamma(2,2)
  - 208*a2^3*a3*f3
  - 192*a2^3*a4*f2
  + 4*a2^3*gamma(3,3)
  - 312*a2^2*a3^2*f2
  + 240*a2^2*a3*f5
  + 92*a2^2*a3*gamma(2,3)
  + 256*a2^2*a4*f4
  + 96*a2^2*a4*gamma(2,2)
  + 240*a2^2*a5*f3
  + 40*a2^2*gamma(3,5)
  + 16*a2^2*gamma(4,4)
  + 288*a2*a3^2*f4
  + 87*a2*a3^2*gamma(2,2)
  + 576*a2*a3*a4*f3
  + 480*a2*a3*a5*f2
  + 60*a2*a3*gamma(2,5)
  + 12*a2*a3*gamma(3,4)
  + 256*a2*a4^2*f2
  - 64*a2*a4*gamma(2,4)
  - 48*a2*a4*gamma(3,3)
  - 250*a2*a5*f5
  - 220*a2*a5*gamma(2,3)
  - 35*a2*gamma(5,5)
  + 108*a3^3*f3
  + 288*a3^2*a4*f2
  - 18*a3^2*gamma(2,4)
  - 18*a3^2*gamma(3,3)
  - 450*a3*a4*f5
  - 168*a3*a4*gamma(2,3)
  - 450*a3*a5*f4
  - 150*a3*a5*gamma(2,2)
  - 90*a3*gamma(4,5)
  - 255*a4^2*f4
  - 64*a4^2*gamma(2,2)
  - 450*a4*a5*f3
  - 30*a4*gamma(3,5)
  - 125*a5^2*f2
  + 50*a5*gamma(2,5)
  + 150*a5*gamma(3,4)
)

verifyEquals = (label, actual, expected) -> (
  print(label | ": " | toString actual);
  if actual != expected then error(label | " expected " | toString expected | " but got " | toString actual)
)

monomialList = poly -> if poly == 0 then {} else flatten entries monomials poly
basisList = d -> flatten entries basis(d, A)

-- The derivation sending x to h is computed as the coefficient of the
-- first-order change under x -> x + t*h.  The auxiliary variable t has very
-- high degree, so it does not affect any degree used by the certificate.
D = (poly, x, h) -> sub(diff(t, sub(poly, {x => (x + t*h)})), {t => 0})

matrixFromColumns = (cols, rows) -> matrix apply(
  rows,
  r -> apply(cols, c -> coefficient(r, c))
)

verifyEquals("expanded R5 support", #monomialList(R5), 88)

degree22Basis = basisList(22)
verifyEquals("full degree-22 monomial count", #degree22Basis, 3868)

-- Choose one monomial in R5 whose coefficient is nonzero.  In the quotient
-- A^22 / Q*R5, this monomial is eliminated using the equation R5 = 0.
basisIndices = toList(0..(#degree22Basis-1))
pivotIndex = first select(basisIndices, i -> coefficient(degree22Basis#i, R5) != 0)
pivotMonomial = degree22Basis#pivotIndex
pivotCoefficient = coefficient(pivotMonomial, R5)

quotientRows = apply(select(basisIndices, i -> i != pivotIndex), i -> degree22Basis#i)

-- This is the most direct normal form modulo Q*R5: subtract the right multiple
-- of R5 so that the pivot monomial has coefficient zero, then record all other
-- coefficients.
reduceModuloRelation = poly -> poly - coefficient(pivotMonomial, poly) / pivotCoefficient * R5

derivationColumns = {}
scan(0..(#genVars-1), i -> (
  x := genVars#i;
  d := generatorDegrees#i;
  scan(basisList(d), h -> (
    derivationColumns = append(derivationColumns, D(R5, x, h))
  ))
))

quotientColumns = apply(derivationColumns, reduceModuloRelation)
quotientMatrix = matrixFromColumns(quotientColumns, quotientRows)

quotientImageRank = rank quotientMatrix
rhoKernelDimension = #derivationColumns - quotientImageRank

verifyEquals("Der^gr(A) columns", #derivationColumns, 468)
verifyEquals("quotient row count", #quotientRows, 3867)
verifyEquals("rank of rho_R", quotientImageRank, 456)
verifyEquals("kernel dimension of rho_R", rhoKernelDimension, 12)

print "Macaulay2 certificate passed: stabilizer Lie algebra dimension is 12 over QQ."
