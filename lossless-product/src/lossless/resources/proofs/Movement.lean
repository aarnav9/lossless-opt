import Std

namespace Movement
/- Fixed copy-only access model. Each source element is an arbitrary independent
bit. The fresh destination starts false. A position copies its source only if
both an input read and an output write are performed. Counts are logical word
accesses, not physical DRAM transactions, compiler events or elapsed time. -/
def copied (input : Nat → Bool) (reads writes : Nat → Nat) (i : Nat) : Bool :=
  if 0 < reads i ∧ 0 < writes i then input i else false

def Correct (n : Nat) (reads writes : Nat → Nat) : Prop :=
  ∀ input i, i < n → copied input reads writes i = input i

def accesses (count : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n+1 => accesses count n + count n

theorem required_access (n : Nat) (r w : Nat → Nat) (h : Correct n r w)
    (i : Nat) (hi : i < n) : 0 < r i ∧ 0 < w i := by
  have eq := h (fun _ => true) i hi
  by_cases hr : 0 < r i ∧ 0 < w i
  · exact hr
  · simp [copied, hr] at eq

theorem coverage_bound (c : Nat → Nat) (n : Nat)
    (h : ∀ i, i < n → 0 < c i) : n ≤ accesses c n := by
  induction n with
  | zero => simp [accesses]
  | succ n ih =>
    have hp : ∀ i, i < n → 0 < c i := by intro i hi; exact h i (by omega)
    have hn := h n (by omega)
    have hb := ih hp
    simp only [accesses]
    omega

theorem fresh_copy_lower_bound (n : Nat) (r w : Nat → Nat) (h : Correct n r w) :
    2*n ≤ accesses r n + accesses w n := by
  have hr := coverage_bound r n (fun i hi => (required_access n r w h i hi).1)
  have hw := coverage_bound w n (fun i hi => (required_access n r w h i hi).2)
  omega

theorem once_correct (n : Nat) : Correct n (fun _ => 1) (fun _ => 1) := by
  intro input i hi
  simp [copied]

theorem once_accesses (n : Nat) : accesses (fun _ => 1) n = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [accesses, ih]

theorem bound_attained (n : Nat) :
    accesses (fun _ => 1) n + accesses (fun _ => 1) n = 2*n := by
  simp [once_accesses, Nat.two_mul]

/- A resident prefix is excluded from the newly materialized set. This arithmetic
comparison does not assert MLX can update in place without hidden copies. -/
theorem resident_prefix_saving (old added : Nat) :
    2*(old+added) = 2*added + 2*old := by omega

theorem vector_partition (group width lane : Nat) (hl : lane < width) :
    (group*width+lane)/width = group ∧ (group*width+lane)%width = lane := by
  have hw : 0 < width := by omega
  simp [Nat.add_div, Nat.add_mod, Nat.div_eq_of_lt hl, Nat.mod_eq_of_lt hl,
        Nat.mul_div_left, hw, hl]

#print axioms required_access
#print axioms coverage_bound
#print axioms fresh_copy_lower_bound
#print axioms once_correct
#print axioms once_accesses
#print axioms bound_attained
#print axioms resident_prefix_saving
#print axioms vector_partition
end Movement
