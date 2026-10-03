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
                p in {"research", ".venv", ".lake", "lossless-runs", ".env", "__pycache__"}
                for p in parts
            ) or name.endswith((".safetensors", ".gguf", ".so", ".dylib", ".pyc")):
                raise ValueError(f"unexpected artifact content: {name}")
        for suffix in [
            "lossless/cli.py",
            "lossless/profiling.py",
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
        records[archive.name] = {
            "archive_bytes": archive.stat().st_size,
            "file_bytes": file_bytes,
            "entries": len(names),
        }
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
