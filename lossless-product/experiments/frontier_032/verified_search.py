"""Bounded index superoptimization with solver counterexamples and real C timing."""

import argparse
from pathlib import Path
import time
import z3
from lossless._native import engine, operators
from lossless._native.common import stamp, write


class Expr:
    """Small integer index IR interpreted by Z3 and emitted as parenthesized C."""

    def __init__(self, op, *args):
        self.op, self.args = op, args

    def z3(self, env):
        if self.op == "var":
            return env[self.args[0]]
        if self.op == "const":
            return z3.IntVal(self.args[0])
        a, b = (x.z3(env) for x in self.args)
        return {
            "+": lambda: a + b,
            "-": lambda: a - b,
            "*": lambda: a * b,
            "/": lambda: a / b,
            "%": lambda: a % b,
        }[self.op]()

    def c(self):
        if self.op in {"var", "const"}:
            return str(self.args[0])
        return "(" + self.args[0].c() + self.op + self.args[1].c() + ")"


def node(op, a, b):
    return Expr(op, a, b)


def expressions(order, defect):
    q, rows, cols, rs, cs = [Expr("var", n) for n in ["q", "rows", "cols", "rs", "cs"]]
    one = Expr("const", 1)
    count = node("*", rows, cols)
    index = node("-", node("-", count, one), q) if order == "reverse" else q
    i, j = (
        (node("%", q, rows), node("/", q, rows))
        if order == "column"
        else (node("/", index, cols), node("%", index, cols))
    )
    dest = node("+", node("*", i, cols), j)
    src = node(
        "+",
        node("*", i, cs if defect == "swapped" else rs),
        node("*", j, rs if defect == "swapped" else cs),
    )
    if defect == "offbyone":
        src = node("+", src, one)
    if defect == "drop":
        dest = node("%", dest, node("-", count, one))
    return dest, src


def prove(rows, cols, layout, order, defect):
    q, r = z3.Ints("q r")
    rs, cs = (cols, 1) if layout == "c" else (1, rows)
    env = {"q": q, "rows": rows, "cols": cols, "rs": rs, "cs": cs}
    dest_ir, src_ir = expressions(order, defect)
    dest, src = dest_ir.z3(env), src_ir.z3(env)
    expected = (dest / cols) * rs + (dest % cols) * cs
    solver = z3.Solver()
    solver.set(timeout=3000)
    solver.add(q >= 0, q < rows * cols)
    solver.add(
        z3.Or(
            src != expected,
            src < 0,
            src >= rows * cols,
            dest < 0,
            dest >= rows * cols,
            z3.And(r >= 0, r < rows * cols, q != r, dest == z3.substitute(dest, (q, r))),
        )
    )
    result = solver.check()
    return {
        "result": str(result),
        "counterexample": str(solver.model()) if result == z3.sat else None,
    }


def source(order, defect):
    dest, src = expressions(order, defect)
    return (
        "#include <stddef.h>\n#include <string.h>\nvoid kernel("
        + operators.ABI
        + ") {for(size_t q=0;q<rows*cols;++q){memcpy(out+("
        + dest.c()
        + "),x+("
        + src.c()
        + "),sizeof(float));}}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=1-2min result={out} log={out.parent / (out.name + '.log')}")
    specs = [
        ("row", "none"),
        ("column", "none"),
        ("reverse", "none"),
        ("row", "swapped"),
        ("column", "drop"),
        ("row", "offbyone"),
    ]
    cases = [
        {"id": f"{s}_{l}", "split": s, "rows": r, "columns": c, "layout": l}
        for s, r, c in [("discovery", 64, 257), ("evaluation", 73, 283)]
        for l in ["c", "f"]
    ]
    # Legality is proved for the whole declared shape domain before seeing data/timings.
    t = time.perf_counter()
    proofs = []
    for order, defect in specs:
        checks = [prove(c["rows"], c["columns"], c["layout"], order, defect) for c in cases]
        proofs.append(
            {
                "order": order,
                "defect": defect,
                "admitted": all(x["result"] == "unsat" for x in checks),
                "checks": checks,
            }
        )
        stamp(f"PROOF order={order} defect={defect} admitted={proofs[-1]['admitted']}")
    proof_seconds = time.perf_counter() - t
    # Real-arithmetic identities cannot justify changing this rounded reduction tree.
    x, y, z = [z3.FPVal(v, z3.Float32()) for v in [33554432, -33554432, 1]]
    rm = z3.RNE()
    left = z3.fpAdd(rm, z3.fpAdd(rm, x, y), z)
    right = z3.fpAdd(rm, x, z3.fpAdd(rm, y, z))
    assert z3.is_true(z3.simplify(z3.Not(z3.fpEQ(left, right))))
    b = z3.BitVec("lane", 32)
    bv = z3.Solver()
    bv.add(~~b != b)
    assert bv.check() == z3.unsat
    write(
        out / "proofs.json",
        {
            "index_checks": proofs,
            "proof_seconds": proof_seconds,
            "float_reassociation_counterexample": {
                "inputs": [33554432, -33554432, 1],
                "left": str(z3.simplify(left)),
                "right": str(z3.simplify(right)),
            },
            "bitvector_double_not": "unsat",
        },
    )
    results = []
    for arm in ["unverified", "verified"]:
        spec = {
            "operator": "copy",
            "comparator": "numpy_copy",
            "cases": cases,
            "budget_seconds": 90,
            "job_timeout_seconds": 5,
            "compile_timeout_seconds": 8,
            "target_ns": 500000,
            "screening_blocks": 5,
            "confirmation_blocks": 13,
            "max_proposals": 6,
            "min_speedup": 1.02,
        }
        write(out / "spec.json", spec)
        run = engine.initialize(out / "spec.json", out / arm)
        for item in proofs:
            if arm == "verified" and not item["admitted"]:
                continue
            name = item["order"] + "_" + item["defect"]
            folder = out / name
            folder.mkdir(exist_ok=True)
            (folder / "kernel.c").write_text(source(item["order"], item["defect"]))
            write(
                folder / "proposal.json",
                {
                    "schema_version": 1,
                    "id": name,
                    "operator": "copy",
                    "source_file": "kernel.c",
                    "hypothesis": "index expression emitted from the same bounded IR checked by Z3",
                },
            )
            engine.submit(run, folder / "proposal.json")
        engine.execute(run, "discovery")
        engine.seal(run)
        engine.execute(run, "evaluation")
        summary = engine.report(run)
        results.append(
            {
                "arm": arm,
                "summary": summary,
                "total_search_seconds": summary["worker_wall_seconds"]
                + (proof_seconds if arm == "verified" else 0),
            }
        )
    write(
        out / "summary.json",
        {
            "proofs": proofs,
            "proof_seconds": proof_seconds,
            "arms": results,
            "scope": "Universal source/destination index and bijection checks for four declared finite shapes and arbitrary copied bits. No verified compiler, general-shape theorem, e-graph implementation, or GPU proof. Float tree counterexample is an independent negative control.",
        },
    )
    stamp(f"COMPLETE result={out / 'summary.json'} log={out.parent / (out.name + '.log')}")


if __name__ == "__main__":
    main()
