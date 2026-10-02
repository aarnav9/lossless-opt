import Std

/- Structural observation theorems: no floating-point algebra is assumed.
   State can include exact bits. The body updates state; the pure head does not.
   Linking these functions to Python/MLX is a separate implementation obligation. -/
namespace DeadProjection

def full {S H L E : Type} (body : S → H × S) (head : H → L)
    (emitted : E) (s : S) : E × S × L :=
  let (h, next) := body s
  (emitted, next, head h)

def observe {S L E : Type} (r : E × S × L) : E × S := (r.1, r.2.1)

def pruned {S H E : Type} (body : S → H × S) (emitted : E) (s : S) : E × S :=
  (emitted, (body s).2)

theorem terminal_equivalence {S H L E : Type} (body : S → H × S)
    (head : H → L) (emitted : E) (s : S) :
    observe (full body head emitted s) = pruned body emitted s := by
  rfl

def live (counts : List Nat) (step : Nat) : Bool :=
  counts.any (fun n => decide (step < n))

theorem terminal_group_dead (counts : List Nat) (step : Nat)
    (h : ∀ n ∈ counts, n = step) : live counts step = false := by
  induction counts with
  | nil => simp [live]
  | cons n ns ih =>
    have hn := h n (by simp)
    have ht : live ns step = false := ih (by intro x hx; exact h x (by simp [hx]))
    simp [live, hn, live] at ht ⊢
    exact ht

/- One H represents an entire original projection group: no splitting or
   reshaping of a group is permitted by this theorem. -/
def visible {H L : Type} (head : H → L) : List (H × Bool) → List L
  | [] => []
  | (h, keep) :: tail => if keep then head h :: visible head tail else visible head tail

def marked {H L : Type} (head : H → L) (groups : List (H × Bool)) : List (L × Bool) :=
  groups.map (fun g => (head g.1, g.2))

theorem whole_group_equivalence {H L : Type} (head : H → L)
    (groups : List (H × Bool)) :
    visible id (marked head groups) = visible head groups := by
  induction groups with
  | nil => rfl
  | cons g gs ih =>
    rcases g with ⟨h, keep⟩
    cases keep <;> simp [marked, visible, ih, marked] at *

def calls : List Bool → Nat
  | [] => 0
  | keep :: tail => (if keep then 1 else 0) + calls tail

/- A legal implementation must retain each live indivisible head invocation.
   Pruning exactly the false flags attains the minimum in this narrow model. -/
inductive SafeMask : List Bool → List Bool → Prop
  | nil : SafeMask [] []
  | cons {l k : Bool} {ls ks : List Bool} :
      (l = true → k = true) → SafeMask ls ks → SafeMask (l :: ls) (k :: ks)

theorem minimal_kept_calls (liveMask keptMask : List Bool)
    (safe : SafeMask liveMask keptMask) :
    calls liveMask ≤ calls keptMask := by
  induction safe with
  | nil => simp [calls]
  | @cons l k ls ks h _ ih =>
    cases l <;> cases k <;> simp_all [calls] <;> omega

#print axioms terminal_equivalence
#print axioms terminal_group_dead
#print axioms whole_group_equivalence
#print axioms minimal_kept_calls
end DeadProjection
