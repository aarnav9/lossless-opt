"""Inspect actual built distributions, not merely ignore rules."""

import json
from pathlib import Path
import sys
import tarfile
import zipfile


def main(directory):
    records = {}
    archives = sorted(Path(directory).glob("*.whl")) + sorted(Path(directory).glob("*.tar.gz"))
    if not any(p.suffix == ".whl" for p in archives) or not any(
        p.name.endswith(".tar.gz") for p in archives
    ):
        raise ValueError("both wheel and source archive are required")
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as stream:
                names = stream.namelist()
                file_bytes = sum(info.file_size for info in stream.infolist())
        else:
            with tarfile.open(archive) as stream:
                names = stream.getnames()
                file_bytes = sum(info.size for info in stream)
        for name in names:
            parts = Path(name).parts
            if any(
                p
                in {
                    "research",
                    ".venv",
                    ".lake",
                    ".lossless",
                    "lossless-runs",
                    ".env",
                    "__pycache__",
                }
                for p in parts
            ) or name.endswith((".safetensors", ".gguf", ".so", ".dylib", ".pyc")):
                raise ValueError(f"unexpected artifact content: {name}")
        for suffix in [
            "lossless/cli.py",
            "lossless/codex_provider.py",
            "lossless/applications/worker.py",
            "lossless/applications/replay.py",
            "lossless/applications/profile_worker.py",
            "lossless/applications/assessment.py",
            "lossless/applications/search.py",
            "lossless/applications/comparison.py",
            "lossless/applications/patches.py",
            "experiments/application_intake/PYBASELINES_LICENSE.txt",
            "lossless/profiling.py",
            "lossless/constraints.py",
            "lossless/_profile_worker.py",
            "lossless/resources/proofs/Bounds.lean",
            "lossless/resources/theorems/index.jsonl.gz",
            "lossless/resources/theorems/MATHLIB_LICENSE",
            "lossless/adapters/mlx/MLX_LM_LICENSE",
            "/LICENSE",
            "NOTICE",
        ]:
            if not any(n.endswith(suffix) for n in names):
                raise ValueError(f"missing package resource: {suffix}")
        if archive.suffix != ".whl":
            for suffix in [
                "examples/softmax-fortran/recipe.py",
                "examples/softmax-fortran/kernel.c",
                "examples/application/project/app.py",
                "experiments/application_intake/frozen-candidates.json",
                "experiments/application_intake/qualify_study.py",
                "experiments/application_intake/broader_study.py",
                "experiments/application_intake/spectrum_workload.py",
                "experiments/application_intake/qualify_broader.py",
                "experiments/application_intake/continue_broader.py",
                "experiments/application_intake/broader-candidate.json",
                "experiments/application_intake/summarize_broader.py",
                "experiments/application_intake/summarize_continuation.py",
                "experiments/application_intake/qualify_continuation.py",
                "experiments/application_intake/continuation-candidate.json",
                "docs/application-broader-study.md",
                "docs/application-continuation-study.md",
                "docs/evidence/application-broader-pybaselines.json.gz",
                "docs/evidence/application-continuation-pybaselines.json.gz",
                "examples/application/cases.json",
                "experiments/kernelbench_049/upstream/LICENSE",
            ]:
                if not any(n.endswith(suffix) for n in names):
                    raise ValueError(f"missing source-distribution resource: {suffix}")
        records[archive.name] = {
            "archive_bytes": archive.stat().st_size,
            "file_bytes": file_bytes,
            "entries": len(names),
        }
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
