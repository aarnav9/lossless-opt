import hashlib
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(folder):
    folder = Path(folder)
    files = {
        str(p.relative_to(folder)): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
        and p.suffix in {".json", ".safetensors", ".model", ".bin", ".gguf"}
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    }
    if "config.json" not in files or not any(
        n.endswith((".safetensors", ".bin", ".gguf")) for n in files
    ):
        raise ValueError(f"Not a local model directory: {folder}")
    return files
