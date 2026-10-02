-- Generated from indexing.ADDRESS; checked before each round.
import Std
namespace CacheAddress
def address (row capacity time width lane : Nat) : Nat := ((((row * capacity) + time) * width) + lane)
end CacheAddress
