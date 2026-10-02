"""Pinned optional Lean checks and local theorem retrieval."""

import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

from ._native.common import stamp, write, sha

RESOURCES = Path(__file__).parent / "resources"
TOOLCHAIN = (RESOURCES / "proofs/lean-toolchain").read_text().strip()


def cache_root():
    return Path(
        os.environ.get("LOSSLESS_CACHE_DIR", str(Path.home() / ".cache/lossless"))
    ).expanduser()


def tool(name):
    local = Path.home() / ".elan/bin" / name
    return shutil.which(name) or (str(local) if local.is_file() else None)


def lean_binary():
    # Resolve the pinned executable without triggering an implicit download.
    direct = (
        Path(os.environ.get("ELAN_HOME", str(Path.home() / ".elan")))
        / "toolchains"
        / TOOLCHAIN.replace("/", "--").replace(":", "---")
        / "bin/lean"
    )
    if direct.is_file():
        return str(direct)
    candidate = tool("lean")
    if candidate and Path(candidate).resolve().name != "elan" and ".elan/bin" not in candidate:
        result = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and TOOLCHAIN.rsplit(":v", 1)[-1] in result.stdout:
            return candidate
    raise ValueError(
        f"pinned Lean toolchain is missing ({TOOLCHAIN}); run lossless setup --proofs lean"
    )


def search(query, limit=5):
    tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
    synonyms = {
        "copy": ["index", "fin", "equiv"],
        "layout": ["matrix", "transpose"],
        "rmsnorm_residual": ["norm", "sum"],
        "softmax": ["exp", "sum"],
    }
    for token in list(tokens):
        tokens.update(synonyms.get(token, []))
    rows = []
    with gzip.open(RESOURCES / "theorems/index.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            score = sum(
                4 * (t in row["name"].lower()) + (t in row["statement"].lower()) for t in tokens
            )
            if score:
                rows.append((score, row))
    rows.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    return {
        "query": query,
        "status": "retrieval_only; algebraic identities do not establish floating-point equivalence",
        "results": [r for _, r in rows[:limit]],
    }


def check(*, timeout=60, output=None):
    if timeout <= 0:
        raise TimeoutError("proof budget exhausted")
    lean = lean_binary()
    started = time.monotonic()
    records = []
    names = ["Bounds", "Eliminate", "Address", "Layout", "Movement", "Fusion", "ThreadMap"]
    with tempfile.TemporaryDirectory(prefix="lossless-proofs-") as directory:
        folder = Path(directory)
        for name in names:
            shutil.copy2(RESOURCES / "proofs" / (name + ".lean"), folder / (name + ".lean"))
        for name in names:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("proof budget exhausted")
            source = folder / (name + ".lean")
            if re.search(r"\b(sorry|admit|unsafe)\b", re.sub(r"--[^\n]*", "", source.read_text())):
                raise ValueError(f"unapproved proof construct in {name}")
            env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG"}}
            env["LEAN_PATH"] = str(folder)
            result = subprocess.run(
                [lean, "-o", name + ".olean", name + ".lean"],
                cwd=folder,
                env=env,
                capture_output=True,
                text=True,
                timeout=remaining,
            )
            if result.returncode or "sorryAx" in result.stdout + result.stderr:
                raise ValueError(
                    f"proof check failed for {name}: {(result.stdout + result.stderr)[-2000:]}"
                )
            records.append(
                {
                    "file": name + ".lean",
                    "sha256": sha(source),
                    "checker_output": result.stdout[-4000:],
                }
            )
    receipt = {
        "status": "checked",
        "toolchain": TOOLCHAIN,
        "checker_sha256": sha(lean),
        "files": records,
        "seconds": time.monotonic() - started,
        "scope": "Bundled scheduling, liveness, indexing and abstract transformation statements only. Does not prove generated C, floating-point GPU arithmetic, or end-to-end equivalence.",
    }
    if output:
        write(output, receipt)
    return receipt


def setup(profile="lean", *, directory=None):
    if profile not in {"lean", "mathlib"}:
        raise ValueError("proof profile must be lean or mathlib")
    folder = (
        Path(directory).resolve()
        if directory
        else cache_root() / "proofs" / TOOLCHAIN.rsplit(":", 1)[-1] / profile
    )
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "setup.log"
    stamp(
        f"START proof setup profile={profile} toolchain={TOOLCHAIN} result={folder} log={log}; initial downloads can take several minutes"
    )
    started = time.monotonic()

    def run(command, *, cwd=folder):
        import threading

        stop = threading.Event()

        def heartbeat():
            while not stop.wait(20):
                stamp(f"proof setup running; elapsed={time.monotonic() - started:.0f}s log={log}")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            with log.open("a") as stream:
                subprocess.run(
                    command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
        finally:
            stop.set()
            thread.join(timeout=1)

    elan = tool("elan")
    if not elan:
        # Explicit setup installs the official bootstrap, never during import/search.
        url = "https://raw.githubusercontent.com/leanprover/elan/v4.2.4/elan-init.sh"
        script = folder / "elan-init.sh"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError("unexpected bootstrap size")
        # Pin both the bootstrap source and the installer release it fetches.
        data = data.replace(b"/latest/download/", b"/download/v4.2.4/")
        script.write_bytes(data)
        write(folder / "bootstrap.json", {"url": url, "sha256": sha(script)})
        run(["sh", str(script), "-y", "--no-modify-path", "--default-toolchain", "none"])
        elan = tool("elan")
    if not elan:
        raise ValueError("Elan installation did not produce an executable; inspect setup.log")
    try:
        lean_binary()
    except ValueError:
        run([elan, "toolchain", "install", TOOLCHAIN])
    if profile == "mathlib":
        for name in [
            "KernelTheorems.lean",
            "Index.lean",
            "lakefile.toml",
            "lake-manifest.json",
            "lean-toolchain",
        ]:
            source = RESOURCES / "theorems" / name
            target = folder / name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(
                    f"existing proof workspace differs: {target}; select an empty --directory"
                )
            shutil.copy2(source, target)
        lake = str(Path(lean_binary()).with_name("lake"))
        run([lake, "exe", "cache", "get"])
        run([lake, "build", "KernelTheorems"])
    receipt = check(timeout=120)
    receipt.update(profile=profile, setup_seconds=time.monotonic() - started, workspace=str(folder))
    write(folder / "setup.json", receipt)
    stamp(f"COMPLETE proof setup result={folder / 'setup.json'} log={log}")
    return receipt
