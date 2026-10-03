"""Trusted-local, content-addressed cache. Timing evidence is never reused."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from .common import read, sha, write


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def entry(directory, kind, identity):
    if not directory:
        yield None
        return
    folder = Path(directory) / kind / key(identity)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield folder


def load(folder):
    if folder is None:
        return None
    try:
        receipt = read(folder / "receipt.json")
        if sha(folder / "record.json") != receipt["record_sha256"]:
            return None
        if "binary_sha256" in receipt and sha(folder / "binary") != receipt["binary_sha256"]:
            return None
        return read(folder / "record.json")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(folder, record, binary=None):
    if folder is None:
        return
    write(folder / "record.json", record)
    receipt = {"record_sha256": sha(folder / "record.json")}
    if binary:
        shutil.copy2(binary, folder / "binary")
        receipt["binary_sha256"] = sha(folder / "binary")
    write(folder / "receipt.json", receipt)
