"""Additional NumPy dispatch candidates; no change to reduction arithmetic.

Keep this separate from the original one-argument recipe: capture overhead is
measured as part of each candidate, and existing fast candidates remain available.
"""

import ast
import copy


# Stable positional signatures; later NumPy-only extensions are left untouched.
# The first entry is the ndarray method, followed by positional and keyword-only
# names. Passing values explicitly retains NumPy's handling of defaults.
SIGNATURES = {
    "sum": ("sum", "a axis dtype out keepdims initial where", ""),
    "prod": ("prod", "a axis dtype out keepdims initial where", ""),
    "mean": ("mean", "a axis dtype out keepdims", "where"),
    "min": ("min", "a axis out keepdims initial where", ""),
    "max": ("max", "a axis out keepdims initial where", ""),
    "amin": ("min", "a axis out keepdims initial where", ""),
    "amax": ("max", "a axis out keepdims initial where", ""),
    "all": ("all", "a axis out keepdims", "where"),
    "any": ("any", "a axis out keepdims", "where"),
    "argmin": ("argmin", "a axis out", "keepdims"),
    "argmax": ("argmax", "a axis out", "keepdims"),
    "cumsum": ("cumsum", "a axis dtype out", ""),
    "cumprod": ("cumprod", "a axis dtype out", ""),
    "std": ("std", "a axis dtype out ddof keepdims", "where"),
    "var": ("var", "a axis dtype out ddof keepdims", "where"),
}
LEGACY = {"sum", "prod", "mean", "min", "max", "all", "any"}


class ExtendedDispatch(ast.NodeTransformer):
    def __init__(self, aliases):
        self.aliases, self.changed = aliases, 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.aliases
            and node.func.attr in SIGNATURES
        ):
            return node
        operation = node.func.attr
        if operation in LEGACY and len(node.args) == 1 and not node.keywords:
            return node
        method, positional, keyword_only = SIGNATURES[operation]
        positional, keyword_only = positional.split(), keyword_only.split()
        keywords = [k.arg for k in node.keywords]
        if (
            any(isinstance(a, ast.Starred) for a in node.args)
            or len(node.args) > len(positional)
            or None in keywords
            or len(set(keywords)) != len(keywords)
            or set(keywords) - set(positional + keyword_only)
            or set(keywords) & set(positional[: len(node.args)])
            or (not node.args and "a" not in keywords)
        ):
            return node
        # Evaluate the original function first, then arguments in source order,
        # exactly once. The lambda's bindings cannot shadow the NumPy alias.
        prefix = "_lossless_dispatch_"
        while any(alias.startswith(prefix) for alias in self.aliases):
            prefix += "_"
        values = [node.func, *node.args, *(k.value for k in node.keywords)]
        names = [f"{prefix}{i}" for i in range(len(values))]
        refs = [ast.Name(id=n, ctx=ast.Load()) for n in names]
        args = refs[1 : 1 + len(node.args)]
        kwargs = [
            ast.keyword(arg=k.arg, value=ref)
            for k, ref in zip(node.keywords, refs[1 + len(node.args) :])
        ]
        array = args[0] if args else next(k.value for k in kwargs if k.arg == "a")
        guard = ast.parse(
            f"type({array.id}) is {node.func.value.id}.ndarray "
            f"and {array.id}.dtype.kind in 'buifc'",
            mode="eval",
        ).body
        # NumPy's __array_function__ dispatch can also inspect out/where. A
        # plain input alone is insufficient when either operand is a subclass
        # or duck array. Keep those calls on the original function.
        provided = dict(zip(positional, args))
        provided.update({k.arg: k.value for k in kwargs})
        checks = [guard]
        alias = node.func.value.id
        for key in ("out", "where"):
            if key not in provided:
                continue
            value = provided[key].id
            plain_array = f"(type({value}) is {alias}.ndarray and {value}.dtype.kind in 'buifc')"
            permitted = (
                f"{value} is None or {plain_array}"
                if key == "out"
                else f"type({value}) is type(True) or type({value}) is {alias}.bool_ or {plain_array}"
            )
            checks.append(ast.parse(permitted, mode="eval").body)
        if len(checks) > 1:
            guard = ast.BoolOp(op=ast.And(), values=checks)
        direct = ast.Call(
            func=ast.Attribute(value=array, attr=method, ctx=ast.Load()),
            args=args[1:] if args else [],
            keywords=[k for k in kwargs if k.arg != "a"],
        )
        original = ast.Call(func=refs[0], args=args, keywords=kwargs)
        result = ast.Call(
            func=ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=n) for n in names],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=ast.IfExp(test=guard, body=direct, orelse=original),
            ),
            args=copy.deepcopy(values),
            keywords=[],
        )
        self.changed += 1
        return ast.copy_location(result, node)


def source_functions(tree):
    """Qualified synchronous functions, excluding closures and decorated scopes."""

    def walk(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.decorator_list:
                symbol = prefix + node.name
                if isinstance(node, ast.FunctionDef):
                    yield symbol, node
                else:
                    yield from walk(node.body, symbol + ".")

    return list(walk(tree.body))
