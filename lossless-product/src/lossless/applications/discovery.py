"""Bounded project discovery without importing application code."""

import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

DEFAULT_LIMITS = {"max_files": 10000, "max_bytes": 256 * 2**20}
EXCLUDED = {
    ".git",
    ".hg",
    ".svn",
    ".lossless",
    ".aws",
    ".ssh",
    ".codex",
    ".agents",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "venv",
    "build",
    "dist",
    "lossless-runs",
    ".DS_Store",
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def excluded(name, patterns):
    path = Path(name)
    return (
        any(p in EXCLUDED or p.startswith(".venv") or p.endswith(".egg-info") for p in path.parts)
        or any(p == ".env" or p.startswith(".env.") for p in path.parts)
        or path.suffix in {".pyc", ".pyo", ".pem", ".key"}
        or any(
            fnmatch.fnmatchcase(name, p) or name.startswith(p.rstrip("/") + "/") for p in patterns
        )
    )


def git(root, *args):
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(root),
                *args,
            ],
            capture_output=True,
            timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        return result.stdout.decode("utf-8") if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def inventory(root, patterns=(), limits=None, include=()):
    root = Path(root).resolve()
    limits = limits or DEFAULT_LIMITS
    if not root.is_dir():
        raise ValueError("project.root must be an existing local directory")
    listing = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", ".")
    ignored_root = listing == "" and bool(git(root, "check-ignore", "--", "."))
    if ignored_root:
        # An explicitly chosen project may live under an enclosing repository's
        # ignored scratch directory. Its files are not part of that repository.
        listing = None
    if listing is not None:
        names = sorted(set(n for n in listing.split("\0") if n))
    else:
        names = []
        for folder, directories, files in os.walk(root, followlinks=False):
            relative = Path(folder).relative_to(root)
            directories[:] = sorted(
                d for d in directories if not excluded((relative / d).as_posix(), patterns)
            )
            for directory in directories:
                if (Path(folder) / directory).is_symlink():
                    raise ValueError(
                        f"project symlink needs an explicit exclude: {relative / directory}"
                    )
            names.extend(
                (relative / f).as_posix()
                for f in files
                if not excluded((relative / f).as_posix(), patterns)
            )
            if len(names) > limits["max_files"]:
                raise ValueError(
                    "project exceeds max_files; narrow project.root or add project.exclude"
                )
        names.sort()
    for name in include:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or excluded(name, patterns):
            raise ValueError(
                f"included fixture escapes the project or conflicts with exclusions: {name}"
            )
        if not (root / path).is_file():
            raise ValueError(f"project.include must name an existing file: {name}")
    names = sorted(set(names) | set(include))
    records, total = {}, 0
    for name in names:
        if excluded(name, patterns):
            continue
        path = root / name
        if path.is_symlink() or any(
            p.is_symlink() for p in path.parents if p != root and root in p.parents
        ):
            raise ValueError(f"project symlink needs an explicit exclude: {name}")
        if path.is_dir():
            raise ValueError(f"nested repository/submodule must be supplied explicitly: {name}")
        if not path.exists():  # Tracked files deleted in the working tree.
            continue
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"unsupported project file: {name}")
        size = path.stat().st_size
        total += size
        if len(records) >= limits["max_files"] or total > limits["max_bytes"]:
            raise ValueError(
                "project exceeds snapshot limits; narrow project.root or add project.exclude"
            )
        records[name] = {
            "sha256": digest(path),
            "bytes": size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    if not records:
        raise ValueError("no project files found after exclusions")
    return {
        "files": records,
        "bytes": total,
        "git": {
            "revision": ((git(root, "rev-parse", "HEAD") or "").strip() or None)
            if listing is not None
            else None,
            "dirty": bool((git(root, "status", "--porcelain", "--", ".") or "").strip())
            if listing is not None
            else None,
        },
        "selection": "git tracked and nonignored untracked files"
        if listing is not None
        else "directory walk (explicit project root ignored by enclosing Git repository)"
        if ignored_root
        else "directory walk",
        "explicit_includes": list(include),
    }


def discovery_order(name):
    """Prefer likely application source within the bounded scan, retaining other hints."""
    path = Path(name)
    if {"test", "tests", "testing"} & set(path.parts) or path.name.startswith("test_"):
        priority = 2
    elif {"doc", "docs", "example", "examples", "benchmark", "benchmarks", "tools"} & set(
        path.parts
    ):
        priority = 1
    else:
        priority = 0
    return priority, name


def discover(root, snapshot):
    candidates, imports, errors = [], set(), []
    python_files = [
        n for n, r in snapshot["files"].items() if n.endswith(".py") and r["bytes"] <= 256 * 1024
    ]
    python_files.sort(key=discovery_order)
    for name in python_files[:200]:
        try:
            tree = ast.parse((Path(root) / name).read_bytes(), filename=name)
        except (SyntaxError, UnicodeError, ValueError) as error:
            errors.append({"file": name, "error": type(error).__name__})
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                positional = [*node.args.posonlyargs, *node.args.args]
                candidates.append(
                    {
                        "target": f"{name}:{node.name}",
                        "required_positional": [
                            a.arg for a in positional[: len(positional) - len(node.args.defaults)]
                        ],
                        "required_keywords": [
                            a.arg
                            for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                            if d is None
                        ],
                    }
                )
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split(".")[0])
    return {
        "entry_candidates": candidates,
        "top_level_imports": sorted(imports),
        "manifests": [
            n
            for n in snapshot["files"]
            if Path(n).name
            in {
                "pyproject.toml",
                "setup.cfg",
                "requirements.txt",
                "uv.lock",
                "poetry.lock",
                "Pipfile.lock",
                "environment.yml",
            }
        ],
        "python_files_scanned": min(200, len(python_files)),
        "scan_truncated": len(python_files) > 200,
        "candidate_order": "Application source before docs/examples/tools, then tests; path heuristic only.",
        "parse_errors": errors,
        "scope": "Static hints only; no application imports, installation, model call, or execution. Invocation and representative inputs are user choices.",
    }


ENVIRONMENT_PROBE = """
import importlib.metadata as metadata, json, os, platform, shutil, sys, sysconfig
from pathlib import Path
# -S suppresses .pth/sitecustomize execution. Read venv metadata ourselves,
# including on Python versions where -S leaves sys.prefix at the base install.
venv = Path(sys.argv[1]).parent.parent
cfg = venv / 'pyvenv.cfg'
paths = []
use_system = True
if cfg.is_file():
    settings = dict(line.lower().split('=', 1) for line in cfg.read_text().splitlines() if '=' in line)
    settings = {k.strip(): v.strip() for k, v in settings.items()}
    use_system = settings.get('include-system-site-packages') == 'true'
    paths += [str(venv / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'), str(venv / 'Lib/site-packages')]
if use_system:
    base = dict(base=sys.base_prefix, platbase=sys.base_exec_prefix)
    paths += [sysconfig.get_path('purelib', vars=base), sysconfig.get_path('platlib', vars=base)]
packages = {}
for dist in metadata.distributions(path=list(dict.fromkeys(paths))):
    name = dist.metadata.get('Name')
    if name:
        packages.setdefault(name.lower().replace('_', '-'), dist.version)
try:
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
except (ValueError, OSError, AttributeError):
    memory = None
print(json.dumps(dict(python=sys.version, executable=sys.executable,
    environment_prefix=str(venv) if cfg.is_file() else sys.prefix,
    package_search_paths=paths, startup_hooks_executed=False,
    platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
    physical_memory_bytes=memory, clang=shutil.which('clang'), packages=packages,
    device_hint='Apple Silicon / Metal' if sys.platform=='darwin' and platform.machine()=='arm64' else None,
    device_execution_verified=False)))
"""


def environment(python=None):
    # Do not resolve the executable symlink: that would lose a venv's identity.
    python = os.path.abspath(python or sys.executable)
    result = subprocess.run(
        [python, "-I", "-S", "-c", ENVIRONMENT_PROBE, python],
        capture_output=True,
        text=True,
        timeout=10,
        cwd="/",
        env={
            k: v
            for k, v in os.environ.items()
            if k in {"PATH", "HOME", "TMPDIR", "LANG", "SYSTEMROOT"}
        },
    )
    if result.returncode:
        raise ValueError("selected Python could not report its environment; check --python")
    try:
        value = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError("selected Python emitted an invalid environment report") from error
    value["scope"] = (
        "Selected interpreter and installed distribution metadata; project dependencies are not installed or imported. GPU execution is not tested."
    )
    return value
