import Std
namespace ThreadMap

-- Launch geometry is a bijective enumeration, independent of element bits.
theorem flat_inverse (i width : Nat) : i % width + width * (i / width) = i := by
  exact Nat.mod_add_div i width

theorem row_div (row columns lane : Nat) (h : lane < columns) :
    (row * columns + lane) / columns = row := by
  have hc : 0 < columns := by omega
  simp [Nat.add_div, Nat.div_eq_of_lt h, Nat.mul_div_left, hc, Nat.mod_eq_of_lt h, h]

theorem row_mod (row columns lane : Nat) (h : lane < columns) :
    (row * columns + lane) % columns = lane := by
  simp [Nat.add_mod, Nat.mod_eq_of_lt h]

theorem row_unique (a b columns x y : Nat) (hx : x < columns) (hy : y < columns)
    (h : a * columns + x = b * columns + y) : a = b ∧ x = y := by
  constructor
  · have z := congrArg (fun n => n / columns) h
    simpa [row_div a columns x hx, row_div b columns y hy] using z
  · have z := congrArg (fun n => n % columns) h
    simpa [row_mod a columns x hx, row_mod b columns y hy] using z

theorem row_covered (i columns rows : Nat) (hc : 0 < columns) (hi : i < rows * columns) :
    i / columns < rows ∧ i % columns < columns ∧
    (i / columns) * columns + i % columns = i := by
  constructor
  · exact (Nat.div_lt_iff_lt_mul hc).mpr hi
  constructor
  · exact Nat.mod_lt i hc
  · simpa [Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div i columns

-- Two scalar stores per thread with a guard on each lane handles odd tails.
theorem chunk_two (i : Nat) : (i / 2) * 2 + i % 2 = i ∧ i % 2 < 2 := by omega

-- Replacing a runtime width by an equal specialization preserves the address.
theorem width_specialization (row capacity time width fixed lane : Nat) (h : fixed = width) :
    (row * capacity + time) * fixed + lane = (row * capacity + time) * width + lane := by
  simp [h]

#print axioms flat_inverse
#print axioms row_div
#print axioms row_mod
#print axioms row_unique
#print axioms row_covered
#print axioms chunk_two
#print axioms width_specialization
end ThreadMap
