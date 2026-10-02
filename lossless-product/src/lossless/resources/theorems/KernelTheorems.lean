import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Matrix.Block
import Mathlib.LinearAlgebra.Finsupp.Defs

open scoped Matrix
namespace KernelTheorems
-- Algebraic identities are search premises, not automatic Float rewrites.
theorem diagonal_action {n : Type} [Fintype n] [DecidableEq n]
    (d x : n → ℚ) (i : n) : (Matrix.diagonal d *ᵥ x) i = d i * x i := by
  exact Matrix.mulVec_diagonal d x i

theorem diagonal_product {n : Type} [Fintype n] [DecidableEq n]
    (a b : n → ℚ) : Matrix.diagonal a * Matrix.diagonal b = Matrix.diagonal (a * b) := by
  exact Matrix.diagonal_mul_diagonal a b

-- Matrix reassociation is valid over rationals. Applying it to rounded GPU
-- arithmetic needs a different theorem and a checked implementation bridge.
theorem compose_action {n : Type} [Fintype n] (a b : Matrix n n ℚ) (x : n → ℚ) :
    (a * b) *ᵥ x = a *ᵥ (b *ᵥ x) := by
  exact (Matrix.mulVec_mulVec x a b).symm

#print axioms diagonal_action
#print axioms diagonal_product
#print axioms compose_action
end KernelTheorems
