import Std

/- A serial additive model with supplied nonnegative integer costs. Neither the
costs' fidelity to elapsed time nor generated kernel equivalence is asserted.
This certificate does not trust an OR-Tools optimality status. -/
namespace PlanCost

structure Edge where
  start : Nat
  stop : Nat
  cost : Nat
deriving DecidableEq

def Valid : List Edge → Nat → Nat → Prop
  | [], src, dst => src = dst
  | e :: es, src, dst => e.start = src ∧ Valid es e.stop dst

instance validDecidable : (es : List Edge) → (src dst : Nat) → Decidable (Valid es src dst)
  | [], src, dst => inferInstanceAs (Decidable (src = dst))
  | e :: es, src, dst =>
    letI := validDecidable es e.stop dst
    inferInstanceAs (Decidable (e.start = src ∧ Valid es e.stop dst))

def cost : List Edge → Nat
  | [] => 0
  | e :: es => e.cost + cost es

theorem lower_bound (potential : Nat → Nat) (es : List Edge) (src dst : Nat)
    (hv : Valid es src dst)
    (hb : ∀ e ∈ es, potential e.stop ≤ potential e.start + e.cost) :
    potential dst ≤ potential src + cost es := by
  induction es generalizing src with
  | nil => simp [Valid] at hv; simp [cost, hv]
  | cons e es ih =>
    obtain ⟨hs, ht⟩ := hv
    have he := hb e (by simp)
    have tail := ih e.stop ht (by intro x hx; exact hb x (by simp [hx]))
    simp only [cost]
    rw [hs] at he
    omega

theorem optimal_of_attains (potential : Nat → Nat) (chosen other : List Edge)
    (src dst : Nat) (hv : Valid other src dst)
    (hb : ∀ e ∈ other, potential e.stop ≤ potential e.start + e.cost)
    (attains : potential dst = potential src + cost chosen) :
    cost chosen ≤ cost other := by
  have h := lower_bound potential other src dst hv hb
  omega

#print axioms lower_bound
#print axioms optimal_of_attains
end PlanCost
