"""Campaign-008 named RMS/residual region proposal for ONNX Runtime."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys


def rewrite(model, *, contract, parallel=False):
    if contract != "numerical":
        raise ValueError("RMS fusion changes reductions and requires numerical mode")
    from ._onnx_rewrite import rewrite as apply

    return apply(model, parallel=parallel)


def build(directory, *, include_directory, timeout=30):
    """Build against caller-supplied ONNX Runtime C headers; never download them."""
    directory, include = Path(directory).resolve(), Path(include_directory).resolve()
    if not (include / "onnxruntime_c_api.h").is_file():
        raise ValueError("include_directory must contain onnxruntime_c_api.h")
    directory.mkdir(parents=True, exist_ok=False)
    binary = directory / ("rms.dylib" if sys.platform == "darwin" else "rms.so")
    command = [
        "clang",
        "-O3",
        "-std=c11",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
        "-I" + str(include),
        str(Path(__file__).with_name("onnx_rms.c")),
        "-o",
        str(binary),
        "-lm",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / "compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"RMS extension build failed: {directory / 'compile.log'}")
    receipt = {
        "command": command,
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("onnx_rms.c").read_bytes()
        ).hexdigest(),
        "header_sha256": hashlib.sha256((include / "onnxruntime_c_api.h").read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    (directory / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return binary
