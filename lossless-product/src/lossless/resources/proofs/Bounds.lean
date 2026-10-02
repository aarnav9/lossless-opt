import Std

/-!
Round-count optimality for a specified synchronous batching model.
Each request needs `count + 1` single-position evaluations: one seed and `count`
pipeline advances, retaining the final active KV state of the pinned API.
Invocations combine only requests at the same decode index and prompt length.
This is NOT a GPU latency lower bound or a floating-point implementation proof.
-/
namespace KernelBounds

def active (counts : List Nat) (step : Nat) : Nat :=
  counts.countP (fun n => decide (step ≤ n))

def waves (groups : List (List Nat)) (step : Nat) : Nat :=
  (groups.map (fun g => if active g step = 0 then 0 else 1)).sum

def ceilDiv (n width : Nat) : Nat := (n + width - 1) / width

def lowerBound (counts : List Nat) (width horizon : Nat) : Nat :=
  ((List.range (horizon + 1)).map (fun t => ceilDiv (active counts t) width)).sum

def callCost (groups : List (List Nat)) (horizon : Nat) : Nat :=
  ((List.range (horizon + 1)).map (waves groups)).sum

theorem ceilDiv_le {n width calls : Nat} (hw : 0 < width)
    (h : n ≤ width * calls) : ceilDiv n width ≤ calls := by
  unfold ceilDiv
  rw [Nat.div_le_iff_le_mul hw]
  have hc : n ≤ calls * width := by simpa [Nat.mul_comm] using h
  omega

theorem active_capacity (groups : List (List Nat)) (step width : Nat)
    (h : ∀ g ∈ groups, g.length ≤ width) :
    active groups.flatten step ≤ width * waves groups step := by
  induction groups with
  | nil => simp [active, waves]
  | cons g gs ih =>
    have hg : active g step ≤ width :=
      Nat.le_trans List.countP_le_length (h g (by simp))
    have ht := ih (by intro x hx; exact h x (by simp [hx]))
    have ha : active (g ++ gs.flatten) step = active g step + active gs.flatten step :=
      List.countP_append
    change active (g ++ gs.flatten) step ≤ _
    rw [ha]
    change active g step + active gs.flatten step ≤
      width * ((if active g step = 0 then 0 else 1) + waves gs step)
    by_cases hz : active g step = 0
    · simpa only [hz, if_pos, Nat.zero_add] using ht
    · simp only [if_neg hz, Nat.mul_add, Nat.mul_one]
      exact Nat.add_le_add hg ht

theorem sum_map_le {f g : Nat → Nat} (xs : List Nat)
    (h : ∀ x, f x ≤ g x) : (xs.map f).sum ≤ (xs.map g).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih => simpa using Nat.add_le_add (h x) ih

theorem lowerBound_le (counts : List Nat) (groups : List (List Nat))
    (width horizon : Nat) (hw : 0 < width)
    (partition : groups.flatten.Perm counts)
    (capacity : ∀ g ∈ groups, g.length ≤ width) :
    lowerBound counts width horizon ≤ callCost groups horizon := by
  apply sum_map_le
  intro t
  apply ceilDiv_le hw
  have eq : active groups.flatten t = active counts t :=
    partition.countP_eq (fun n => decide (t ≤ n))
  rw [← eq]
  exact active_capacity groups t width capacity

/- A concrete, checked witness attaining the bound is globally optimal among
   partitions in this cost model, not merely best among sampled candidates. -/
theorem optimal_of_attains (counts : List Nat) (candidate other : List (List Nat))
    (width horizon : Nat) (hw : 0 < width)
    (partition : other.flatten.Perm counts)
    (capacity : ∀ g ∈ other, g.length ≤ width)
    (attains : callCost candidate horizon = lowerBound counts width horizon) :
    callCost candidate horizon ≤ callCost other horizon := by
  rw [attains]
  exact lowerBound_le counts other width horizon hw partition capacity

/- Conditional semantic theorem: independent per-request execution preserves
   request/result associations under partitioning and permutation. In an
   application, S contains IDs and complete state. Establishing GPU row
   independence is a separate obligation; it is NOT assumed proved here. -/
theorem independent_execution {S R : Type} (execute : S → R)
    (requests : List S) (groups : List (List S))
    (partition : groups.flatten.Perm requests) :
    (groups.map (List.map execute)).flatten.Perm (requests.map execute) := by
  rw [← List.map_flatten]
  exact partition.map execute

#print axioms lowerBound_le
#print axioms optimal_of_attains
#print axioms independent_execution
end KernelBounds
