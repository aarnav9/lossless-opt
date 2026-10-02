import Address
open CacheAddress

namespace CacheLayout
/- Elements may be arbitrary bit patterns. These statements do not use
   floating-point algebra. Backend tensor primitives remain separate obligations. -/

def writeRow {α : Type} (base : Nat → Nat → α) (target : Nat) (row : Nat → α) : Nat → Nat → α :=
  fun b t => if b = target then row t else base b t

def scatterRows {α : Type} (rows base : Nat → Nat → α) : Nat → Nat → Nat → α
  | 0 => base
  | n+1 => writeRow (scatterRows rows base n) n (rows n)

/- The reference writes one cache row at a time into a zero buffer. Direct
   concatenation has element rows[b][t]. Prove equality for every valid row,
   with t representing the complete within-row tensor index. -/
theorem concat_assembly {α : Type} (rows base : Nat → Nat → α)
    (count b t : Nat) (hb : b < count) : scatterRows rows base count b t = rows b t := by
  induction count with
  | zero => omega
  | succ n ih =>
    by_cases eq : b = n
    · simp [scatterRows, writeRow, eq]
    · have h : b < n := by omega
      simpa [scatterRows, writeRow, eq] using ih h

def gather {α : Type} (rows : Nat → α) (keep : Nat → Nat) : Nat → α :=
  fun b => rows (keep b)

theorem prefix_gather {α : Type} (rows : Nat → α) (keep : Nat → Nat)
    (count : Nat) (h : ∀ b, b < count → keep b = b) :
    ∀ b, b < count → gather rows keep b = rows b := by
  intro b hb
  simp [gather, h b hb]

theorem zero_padding_preserved (keep : Nat → Nat) :
    gather (fun _ => (0 : Nat)) keep = fun _ => 0 := rfl

def extract {α : Type} (rows : Nat → Nat → α) (b padding t : Nat) : α :=
  rows b (padding + t)

theorem zero_padding_extract {α : Type} (rows : Nat → Nat → α) (b t : Nat) :
    extract rows b 0 t = rows b t := by simp [extract]

def appendAt {α : Type} (old new : Nat → α) (offset : Nat) : Nat → α :=
  fun t => if t < offset then old t else new (t - offset)

theorem append_preserves_prefix {α : Type} (old new : Nat → α) (offset t : Nat)
    (h : t < offset) : appendAt old new offset t = old t := by
  simp [appendAt, h]

theorem append_writes_new {α : Type} (old new : Nat → α) (offset j : Nat) :
    appendAt old new offset (offset + j) = new j := by
  have h : ¬ offset+j < offset := by omega
  simp [appendAt, h]

/- Any implementation with these elementwise obligations observes the same
   active cache, independently of inactive allocated capacity. -/
theorem append_implementation {α : Type} (old new actual : Nat → α)
    (offset added : Nat)
    (hp : ∀ t, t < offset → actual t = old t)
    (suffix : ∀ j, j < added → actual (offset+j) = new j) :
    ∀ t, t < offset+added → actual t = appendAt old new offset t := by
  intro t ht
  by_cases h : t < offset
  · simp [appendAt, h, hp t h]
  · have hj : t - offset < added := by omega
    have eq : offset + (t-offset) = t := by omega
    have hs := suffix (t-offset) hj
    rw [eq] at hs
    simpa [appendAt, h] using hs

theorem address_decode (r c t w d : Nat) (ht : t < c) (hd : d < w) :
    ((address r c t w d / w) / c, (address r c t w d / w) % c, address r c t w d % w) = (r,t,d) := by
  have hc : 0 < c := by omega
  have hw : 0 < w := by omega
  simp [address, Nat.add_div, Nat.add_mod, Nat.div_eq_of_lt hd, Nat.div_eq_of_lt ht,
        Nat.mod_eq_of_lt hd, Nat.mod_eq_of_lt ht, Nat.mul_div_left, hc, hw, Nat.not_le_of_gt ht, Nat.not_le_of_gt hd]
theorem block_bound (r c t n : Nat) (hr : r < n) (ht : t < c) : r*c+t < n*c := by
  calc
    r*c+t < r*c+c := Nat.add_lt_add_left ht _
    _ = (r+1)*c := by simp [Nat.add_mul]
    _ ≤ n*c := Nat.mul_le_mul_right c (by omega)
theorem address_bound (r c t w d n : Nat) (hr : r<n) (ht : t<c) (hd : d<w) :
    address r c t w d < n*c*w := by
  exact block_bound (r*c+t) w d (n*c) (block_bound r c t n hr ht) hd
#print axioms address_decode
#print axioms address_bound
theorem address_injective (r r' c t t' w d d' : Nat)
    (ht : t < c) (ht' : t' < c) (hd : d < w) (hd' : d' < w)
    (h : address r c t w d = address r' c t' w d') : (r,t,d) = (r',t',d') := by
  have eq := congrArg (fun i => ((i / w) / c, (i / w) % c, i % w)) h
  simpa only [address_decode r c t w d ht hd, address_decode r' c t' w d' ht' hd'] using eq

#print axioms address_injective

#print axioms concat_assembly
#print axioms prefix_gather
#print axioms zero_padding_preserved
#print axioms zero_padding_extract
#print axioms append_preserves_prefix
#print axioms append_writes_new
#print axioms append_implementation
end CacheLayout
