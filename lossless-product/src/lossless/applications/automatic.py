"""Small, inspectable exact CPU source candidates; measurements decide admission."""

import ast
import copy
from pathlib import Path

from .retained import retained_plan


REDUCTIONS = {"sum", "prod", "mean", "min", "max", "all", "any"}


class Dispatch(ast.NodeTransformer):
    def __init__(self, aliases, methods):
        self.aliases, self.methods, self.changed = aliases, methods, 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.aliases
            and node.func.attr in self.methods
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and not node.keywords
        ):
            return node
        name, array, method = node.func.value.id, node.args[0].id, node.func.attr
        # Exact ndarray only: subclasses, lists and duck arrays keep NumPy dispatch.
        expression = (
            f"({array}.copy(order='K') if type({array}) is {name}.ndarray else ORIGINAL)"
            if method == "copy"
            else f"({array}.{method}() if type({array}) is {name}.ndarray else ORIGINAL)"
        )
        result = ast.parse(expression, mode="eval").body
        result.orelse = node
        self.changed += 1
        return ast.copy_location(result, node)


def numpy_candidates(root, source, profile):
    """Restrict edits to measured project functions, never evaluation input contents."""
    hotspots = {}
    for case in profile.get("profiles", []):
        for row in case.get("attribution", {}).get("top_cumulative", []):
            if row["origin"] == "project":
                key = row["file"], row["function"]
                hotspots[key] = hotspots.get(key, 0) + row["cumulative_seconds"]
    functions = []
    for (name, symbol), seconds in sorted(hotspots.items(), key=lambda p: -p[1]):
        if name not in source["files"] or source["files"][name]["bytes"] > 256000:
            continue
        try:
            tree = ast.parse((Path(root) / name).read_text())
        except (SyntaxError, UnicodeError):
            continue
        aliases = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "numpy"
        }
        # Avoid rewriting shadowed builtins or aliases. Only module-level functions.
        bindings = {
            n.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
        }
        bindings |= {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        bindings |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        bindings |= {
            n.asname or n.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for n in node.names
            if not (node in tree.body and isinstance(node, ast.Import) and n.name == "numpy")
        }
        if bindings & (aliases | {"type"}):
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name != symbol or node.decorator_list:
                continue
            shadowed = {
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            shadowed |= {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)}
            shadowed |= {
                n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            if not aliases or (aliases | {"type"}) & shadowed:
                continue
            functions.append((name, symbol, node, aliases))
        if len(functions) >= 8:
            break
    proposals, scope, seen = [], [], set()
    for label, methods in (
        ("reductions", REDUCTIONS),
        ("copies", {"copy"}),
        ("combined", REDUCTIONS | {"copy"}),
    ):
        edits = []
        for name, symbol, function, aliases in functions:
            transform = Dispatch(aliases, methods)
            body = copy.deepcopy(function.body)
            if (
                isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            body = [transform.visit(node) for node in body]
            if transform.changed:
                edit = dict(
                    path=name, symbol=symbol, body="\n".join(ast.unparse(n) for n in body) + "\n"
                )
                edits.append(edit)
                entry = dict(path=name, symbol=symbol)
                if entry not in scope:
                    scope.append(entry)
        key = repr(edits)
        if edits and key not in seen:
            seen.add(key)
            proposals.append(
                dict(
                    schema_version=1,
                    hypothesis=f"Guarded ndarray {label} dispatch; preserve arithmetic and subclass fallback.",
                    edits=edits,
                )
            )
    return scope, proposals


def plan_for(root, source, profile, budget):
    try:
        plan = retained_plan(root, budget=budget)
        family = "retained_pybaselines"
    except ValueError:
        scope, candidates = numpy_candidates(root, source, profile)
        if not candidates:
            return None, "No applicable local source recipe for the measured functions."
        family = "numpy_dispatch"
        plan = dict(
            schema_version=1,
            source_scope=scope,
            candidates=candidates,
            max_candidates=len(candidates),
            architecture=False,
            discovery_resources=True,
            contract="CPU Python/NumPy callable: exact observed output bits, input mutation and retained outputs on frozen cases. Array subclasses retain original dispatch. No algebra reassociation, precision change, sampling, global-state proof or numerical relaxation.",
        )
    plan.update(
        wall_time_seconds=budget,
        final_reserve_seconds=budget * 0.45,
        discovery_pairs=2,
        evaluation_pairs=6,
    )
    return plan, family
