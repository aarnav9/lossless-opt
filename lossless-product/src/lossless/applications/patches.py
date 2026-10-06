"""Reviewable function-body proposals. This boundary is not a code sandbox."""

import ast
import difflib
import textwrap

from ..jobs import fields, identity
from .config import relative


def function(source, symbol):
    nodes = ast.parse(source).body
    for name in symbol.split("."):
        matches = [
            node
            for node in nodes
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"ambiguous or absent source symbol: {symbol}")
        node = matches[0]
        nodes = node.body
    if not isinstance(node, ast.FunctionDef):
        raise ValueError("candidate scope must select a synchronous function")
    if (
        isinstance(nodes[0], ast.Expr)
        and isinstance(nodes[0].value, ast.Constant)
        and isinstance(nodes[0].value.value, str)
    ):
        nodes = nodes[1:]
    if not nodes or nodes[0].lineno == node.lineno:
        raise ValueError("candidate function needs a multiline executable body")
    return node, nodes[0].lineno - 1, node.end_lineno


def scope_sources(root, scope):
    if not isinstance(scope, list) or not 1 <= len(scope) <= 8:
        raise ValueError("source_scope must contain 1..8 function selections")
    result = []
    seen = set()
    for row in scope:
        fields(row, {"path", "symbol"}, "source scope", {"path", "symbol"})
        name = relative(row["path"], "source scope path")
        symbol = row["symbol"]
        if not isinstance(symbol, str) or not all(p.isidentifier() for p in symbol.split(".")):
            raise ValueError("source scope symbol must be a qualified identifier")
        path = root / name
        if path.suffix != ".py" or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("source scope must name a local Python source file")
        if path.stat().st_size > 256000:
            raise ValueError("candidate source file exceeds 256 KB")
        if (name, symbol) in seen:
            raise ValueError("duplicate source scope")
        for old_path, old_symbol in seen:
            if old_path == name and (
                old_symbol.startswith(symbol + ".") or symbol.startswith(old_symbol + ".")
            ):
                raise ValueError("overlapping source scopes")
        seen.add((name, symbol))
        source = path.read_text()
        node, first, last = function(source, symbol)
        result.append(
            {
                **row,
                "signature": ast.unparse(node.args),
                "module_imports": [
                    ast.unparse(n)
                    for n in ast.parse(source).body
                    if isinstance(n, (ast.Import, ast.ImportFrom))
                ],
                "body": textwrap.dedent("".join(source.splitlines(keepends=True)[first:last])),
            }
        )
    return result


def schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "hypothesis", "edits"],
        "properties": {
            "schema_version": {"type": "integer", "enum": [1]},
            "hypothesis": {"type": "string"},
            "edits": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "symbol", "body"],
                    "properties": {key: {"type": "string"} for key in ("path", "symbol", "body")},
                },
            },
        },
    }


def validate(proposal, scope):
    fields(
        proposal,
        {"schema_version", "hypothesis", "edits"},
        "proposal",
        {"schema_version", "hypothesis", "edits"},
    )
    if type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1:
        raise ValueError("unsupported candidate schema")
    if not isinstance(proposal["hypothesis"], str) or not 1 <= len(proposal["hypothesis"]) <= 8000:
        raise ValueError("candidate hypothesis must be bounded text")
    edits = proposal["edits"]
    if not isinstance(edits, list) or len(edits) > 8:
        raise ValueError("at most eight source edits")
    allowed = {(s["path"], s["symbol"]) for s in scope}
    seen = set()
    for edit in edits:
        fields(edit, {"path", "symbol", "body"}, "edit", {"path", "symbol", "body"})
        if not all(isinstance(edit[k], str) for k in edit):
            raise ValueError("edit fields must be strings")
        key = edit["path"], edit["symbol"]
        if key not in allowed or key in seen:
            raise ValueError("edit is outside frozen scope or duplicates a symbol")
        seen.add(key)
        if not edit["body"].strip() or len(edit["body"].encode()) > 32000:
            raise ValueError("replacement body must be nonempty and at most 32 KB")
        # Parsing under an enclosing function rejects accidental top-level escape.
        try:
            tree = ast.parse("def _candidate():\n" + textwrap.indent(edit["body"], "    "))
        except SyntaxError as error:
            raise ValueError("replacement body has invalid Python syntax") from error
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(tree)):
            raise ValueError("candidate may not rebind module or closure state")


def apply(reference, destination, proposal, scope):
    """Destination must already be a copied reference; return diff and AST identity."""
    validate(proposal, scope)
    by_path = {}
    for edit in proposal["edits"]:
        by_path.setdefault(edit["path"], []).append(edit)
    changes, canonical = [], {}
    for name, edits in by_path.items():
        original = (reference / name).read_text()
        lines = original.splitlines(keepends=True)
        replacements = []
        for edit in edits:
            _, first, last = function(original, edit["symbol"])
            indent = lines[first][: len(lines[first]) - len(lines[first].lstrip())]
            body = textwrap.indent(textwrap.dedent(edit["body"]).rstrip() + "\n", indent)
            replacements.append((first, last, body))
        for first, last, body in sorted(replacements, reverse=True):
            lines[first:last] = [body]
        changed = "".join(lines)
        compile(changed, name, "exec")
        canonical[name] = ast.dump(ast.parse(changed), include_attributes=False)
        (destination / name).write_text(changed)
        changes.extend(
            difflib.unified_diff(
                original.splitlines(True),
                changed.splitlines(True),
                fromfile="a/" + name,
                tofile="b/" + name,
            )
        )
    # Include unchanged scoped files so equivalent partial/full proposals deduplicate.
    for row in scope:
        name = row["path"]
        canonical.setdefault(
            name, ast.dump(ast.parse((destination / name).read_text()), include_attributes=False)
        )
    return "".join(changes), identity(canonical)
