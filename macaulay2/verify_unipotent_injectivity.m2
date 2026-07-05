-- Minimal Macaulay2 certificate used by rank5_stabilizer_proof.tex.
--
-- This script verifies only the finite linear-algebra assertion used in the
-- proof: the map
--
--   Phi : Lie(U) + Q*kappa -> F^22,
--         (delta,kappa) |-> delta(R5) - kappa R5
--
-- is injective over Q.

F = QQ[
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

sumList = L -> (
  LL := toList L;
  if #LL == 0 then 0 else sum LL
)

monomialList = poly -> if poly == 0 then {} else flatten entries monomials poly
basisList = d -> flatten entries basis(d, F)
isPrimitiveGenerator = m -> member(m, genVars)
decomposableBasis = d -> select(basisList d, m -> not isPrimitiveGenerator(m))

-- The derivation sending x to h is computed as the coefficient of the
-- first-order change under x -> x + t*h.
D = (poly, x, h) -> sub(diff(t, sub(poly, {x => (x + t*h)})), {t => 0})

matrixFromColumns = (cols, rows) -> matrix apply(
  rows,
  r -> apply(cols, c -> coefficient(r, c))
)

verifyEquals("expanded R5 support", #monomialList(R5), 88)

unipotentColumns = {}
scan(0..(#genVars-1), i -> (
  x := genVars#i;
  d := generatorDegrees#i;
  scan(decomposableBasis(d), h -> (
    unipotentColumns = append(unipotentColumns, D(R5, x, h))
  ))
))

verifyEquals("Lie(U) columns", #unipotentColumns, 390)

-- The extra column -R5 is the infinitesimal line-rescaling parameter kappa.
phiColumns = append(unipotentColumns, -R5)
rowMonomials = unique flatten apply(phiColumns, c -> monomialList(c))
phiMatrix = matrixFromColumns(phiColumns, rowMonomials)

verifyEquals("columns of Phi", #phiColumns, 391)
verifyEquals("full degree-22 monomial count", #basisList(22), 3868)
verifyEquals("nonzero rows of Phi", #rowMonomials, 1976)
verifyEquals("rank of Phi", rank phiMatrix, #phiColumns)

print "Macaulay2 certificate passed: Phi is injective over QQ."
