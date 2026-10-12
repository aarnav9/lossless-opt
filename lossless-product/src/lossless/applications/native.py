"""Opt-in, frozen native helpers for trusted local application source searches.

Compilation and ctypes are execution boundaries, not a security sandbox or an
equivalence proof. The ordinary independent application evaluator remains the gate.
"""

import json
from pathlib import Path
import re
import subprocess
import sys

from ..jobs import fields
from .._native.common import sha, write

TYPES = {
    "pointer": "c_void_p",
    "size_t": "c_size_t",
    "int": "c_int",
    "int64": "c_int64",
    "float": "c_float",
    "double": "c_double",
}
IDENTIFIER = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}\Z")


def validate_scope(value):
    if not isinstance(value, list) or len(value) > 4:
        raise ValueError("native_helpers must list at most four declared helpers")
    seen = set()
    for helper in value:
        fields(helper, {"name", "functions"}, "native helper", {"name", "functions"})
        name = helper["name"]
        if not isinstance(name, str) or not IDENTIFIER.fullmatch(name) or name in seen:
            raise ValueError("native helper names must be distinct bounded identifiers")
        seen.add(name)
        functions = helper["functions"]
        if not isinstance(functions, list) or not 1 <= len(functions) <= 16:
            raise ValueError("native helper requires 1..16 declared function signatures")
        symbols = set()
        for function in functions:
            fields(
                function,
                {"name", "returns", "arguments"},
                "native signature",
                {"name", "returns", "arguments"},
            )
            symbol = function["name"]
            if not isinstance(symbol, str) or not IDENTIFIER.fullmatch(symbol) or symbol in symbols:
                raise ValueError("native function names must be distinct bounded identifiers")
            symbols.add(symbol)
            if not isinstance(function["returns"], str) or function["returns"] not in {
                *TYPES,
                "void",
            }:
                raise ValueError("unsupported native return type")
            if (
                not isinstance(function["arguments"], list)
                or len(function["arguments"]) > 32
                or any(not isinstance(t, str) or t not in TYPES for t in function["arguments"])
            ):
                raise ValueError("unsupported native argument types")
    return value


def names(name):
    stem = "_lossless_native_" + name
    # Python prioritizes an identically named .so over the ctypes .py wrapper.
    binary = stem + ".native" + (".dylib" if sys.platform == "darwin" else ".so")
    return [stem + ".c", stem + ".py", binary]


def schema(scope):
    return {
        "type": "array",
        "maxItems": len(scope),
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "source"],
            "properties": {
                "name": {"type": "string", "enum": [h["name"] for h in scope]},
                "source": {"type": "string"},
            },
        },
    }


def validate_sources(value, scope):
    validate_scope(scope)
    if not isinstance(value, list) or len(value) > len(scope):
        raise ValueError("native_sources exceeds the declared helper scope")
    allowed = {h["name"] for h in scope}
    seen = set()
    for item in value:
        fields(item, {"name", "source"}, "native source", {"name", "source"})
        if not isinstance(item["name"], str) or item["name"] not in allowed or item["name"] in seen:
            raise ValueError("native source is outside scope or duplicated")
        seen.add(item["name"])
        if not isinstance(item["source"], str) or not 1 <= len(item["source"].encode()) <= 65536:
            raise ValueError("native source requires 1..65536 UTF-8 bytes")


def build(project, sources, scope, *, timeout, output):
    """Compile declared helpers without loading candidate binaries in the controller."""
    import time

    validate_sources(sources, scope)
    deadline = time.monotonic() + timeout
    declarations = {h["name"]: h for h in scope}
    receipts = []
    generated = []
    for item in sources:
        files = names(item["name"])
        paths = [Path(project) / n for n in files]
        if any(p.exists() or p.is_symlink() for p in paths):
            raise ValueError("native helper would replace an existing project file")
        c, wrapper, binary = paths
        c.write_text(item["source"])
        command = [
            "clang",
            "-O3",
            "-std=c11",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
            str(c),
            "-o",
            str(binary),
            "-lm",
        ]
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("native helper compilation exhausted the search allowance")
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=min(left, 30))
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("native helper compilation timed out") from error
        record = dict(
            name=item["name"],
            command=command,
            returncode=result.returncode,
            diagnostics=(result.stdout + result.stderr)[-16000:],
            source_sha256=sha(c),
        )
        receipts.append(record)
        write(output, receipts)
        if result.returncode:
            raise ValueError("native helper did not compile; inspect the build receipt")
        declarations_json = json.dumps(declarations[item["name"]]["functions"])
        wrapper.write_text(
            '"""Generated declared ctypes ABI; validated with its complete application."""\n'
            "import ctypes\nimport json\nfrom pathlib import Path\n"
            f"_library = ctypes.CDLL(str(Path(__file__).with_name({binary.name!r})))\n"
            f"_types = {TYPES!r}\n"
            "_exports = {}\n"
            f"for _entry in json.loads({declarations_json!r}):\n"
            "    _function = getattr(_library, _entry['name'])\n"
            "    _function.argtypes = [getattr(ctypes, _types[t]) for t in _entry['arguments']]\n"
            "    _function.restype = None if _entry['returns'] == 'void' else getattr(ctypes, _types[_entry['returns']])\n"
            "    _exports[_entry['name']] = _function\n"
            "globals().update(_exports)\n"
        )
        record["hashes"] = {p.name: sha(p) for p in paths}
        generated.extend(files)
        write(output, receipts)
    return {"files": generated, "builds": receipts}
