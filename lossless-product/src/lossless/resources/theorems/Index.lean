import KernelTheorems
open Lean Elab Command
set_option maxHeartbeats 4000000

-- Export actual elaborated theorem types from this pinned imported environment.
-- Retrieval is not certification of a numerical implementation.
run_cmd liftTermElabM do
  let env ← getEnv
  for (name, info) in env.constants.toList do
    match info with
    | .thmInfo _ =>
      let n := name.toString
      if ["Matrix.", "LinearMap.", "Finsupp.", "Finset.", "Nat.", "Int.", "Rat.",
          "Function.", "Equiv.", "KernelTheorems."].any (fun p => p.isPrefixOf n) then
        let statement ← PrettyPrinter.ppExpr info.type
        let moduleName := match env.const2ModIdx[name]? with
          | some idx => env.header.moduleNames[idx.toNat]!.toString
          | none => "KernelTheorems"
        let row := Json.mkObj [("name", toJson n), ("statement", toJson statement.pretty),
                              ("module", toJson moduleName)]
        IO.println row.compress
    | _ => pure ()
