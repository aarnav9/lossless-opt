"""Base wheel smoke: run outside the checkout with only Lossless and NumPy."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import lossless
from lossless.artifacts import export
from lossless.recipes import catalog, native_candidates
from lossless.retained.bf16_codec import encode, decode
from lossless.retained.uint8 import Uint8Pipeline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-prefix", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


stamp(f"START estimate=under30sec result={out}/results.json log={out}.log")
package = Path(lossless.__file__).resolve()
assert package.is_relative_to(args.expected_prefix.resolve()), package
assert len(catalog()) == 30
for recipe in native_candidates("softmax", [dict(rows=64, columns=257, layout="f")]):
    assert "void kernel" in recipe["source"]
for name in (
    "modpoly.c",
    "modpoly.json",
    "torch_native.cpp",
    "mps_native.mm",
    "mps_kernels.metal",
    "onnx_rms.c",
    "PROVENANCE.json",
):
    assert (package.parent / "retained" / name).is_file()
for words in ([0, 0x8000, 0x7FC1, 0xFFFF], list(range(64))):
    assert decode(encode(words, 120)) == words
pipeline = Uint8Pipeline([("add", 250), ("xor", 83)], strategy="lookup")
values = np.arange(256, dtype=np.uint8).reshape(16, 16).T
assert pipeline(values).tobytes() == pipeline.reference(values).tobytes()
subprocess.run(
    [sys.executable, "-m", "lossless", "demo", "--budget", "30s", "--output", str(out / "demo")],
    check=True,
    timeout=45,
)
export(out / "demo", out / "artifact")
operation = lossless.load(out / "artifact")
candidate = operation.optimized_calls()
for seed in (211, 7921):
    bits = np.random.default_rng(seed).integers(0, 2**32, (73, 259), dtype=np.uint32)
    x = bits.view(np.float32)
    expected = operation(x)
    actual = candidate(x)
    assert x.tobytes() == expected.tobytes() == actual.tobytes()
    assert not np.shares_memory(x, actual)
report = dict(
    package=str(package),
    catalog_entries=len(catalog()),
    codec=True,
    byte_pipeline=True,
    export_load_optimized_calls=True,
)
(out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
stamp(f"COMPLETE result={out}/results.json log={out}.log")
