"""Bounded integer copy-index proofs; no compiler or floating-point proof.

Optional dependency: z3-solver. A successful finite index proof only filters
proposals; source compilation and independent runtime validation remain required.
"""

import z3
from .._native import operators


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


def prove(rows, cols, layout, order, defect="none"):
    if (
        type(rows) is not int
        or type(cols) is not int
        or rows < 1
        or cols < 1
        or rows * cols > 2**31 - 1
        or layout not in {"c", "f"}
        or order not in {"row", "column", "reverse"}
        or defect not in {"none", "swapped", "offbyone", "drop"}
        or (defect == "drop" and rows * cols == 1)
    ):
        raise ValueError("unsupported finite copy-index domain")
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


def candidate(cases, *, order="row", defect="none"):
    """Emit only if every declared C/F shape has an unsat negated obligation."""
    if not cases:
        raise ValueError("nonempty declared shape domain required")
    checks = [prove(c["rows"], c["columns"], c["layout"], order, defect) for c in cases]
    admitted = all(c["result"] == "unsat" for c in checks)
    return {
        "admitted": admitted,
        "checks": checks,
        "source": source(order, defect) if admitted else None,
    }
