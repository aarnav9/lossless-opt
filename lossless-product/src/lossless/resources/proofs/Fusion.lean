import Std
namespace Fusion

-- Abstract operations can denote fully specified rounded machine operations.
-- No associativity, distributivity or replacement of multiply/add by FMA.
def staged {I A B C : Type} (f : A → B) (g : B → C) (input : I → A) : I → C :=
  let intermediate := fun i => f (input i)
  fun i => g (intermediate i)

def fused {I A B C : Type} (f : A → B) (g : B → C) (input : I → A) : I → C :=
  fun i => g (f (input i))

theorem remove_intermediate {I A B C : Type} (f : A → B) (g : B → C) (x : I → A) :
    staged f g x = fused f g x := rfl

-- A store/load may be erased only if it preserves the represented value/bits.
theorem eliminate_exact_storage {A B C : Type} (f : A → B) (g : B → C)
    (storeLoad : B → B) (h : ∀ b, storeLoad b = b) (a : A) :
    g (storeLoad (f a)) = g (f a) := by rw [h]

-- A narrowing store can round: fusion retains that conversion explicitly.
theorem preserve_rounding_boundary {A B C D : Type} (f : A → B) (round : B → C)
    (g : C → D) (a : A) :
    (let temporary := round (f a); g temporary) = g (round (f a)) := rfl

-- Shared pure subexpressions need no real-number algebraic identities.
theorem share_pure_operation {A B C D : Type} (f : A → B) (g : B → C) (h : B → D) (a : A) :
    (g (f a), h (f a)) = (let t := f a; (g t, h t)) := rfl

-- Two independent K/V passes may form a paired pass under disjoint-output
-- semantics. Aliasing/races and the actual GPU bridge remain obligations.
theorem pair_independent_passes {I A B : Type} (k : I → A) (v : I → B) :
    ((fun i => (k i, v i).1), (fun i => (k i, v i).2)) = (k, v) := rfl

-- With rounded primitive multiply and add, preserve their expression tree.
theorem retain_mul_add {F : Type} (mul add : F → F → F) (a b c : F) :
    (let product := mul a b; add product c) = add (mul a b) c := rfl

-- An FMA replacement needs this additional, input-specific equality premise.
theorem fma_requires_equivalence {F : Type} (mul add : F → F → F) (fma : F → F → F → F)
    (a b c : F) (h : fma a b c = add (mul a b) c) :
    (let product := mul a b; add product c) = fma a b c := h.symm

#print axioms remove_intermediate
#print axioms eliminate_exact_storage
#print axioms preserve_rounding_boundary
#print axioms share_pure_operation
#print axioms pair_independent_passes
#print axioms retain_mul_add
#print axioms fma_requires_equivalence
end Fusion
