"""Explicit current-runtime Metal model candidates, campaigns 013/014.

Old Torch-2.2 exact kernels are superseded. No import-time build or model download.
Whole logit/KV qualification must precede deployment; token agreement is weaker.
"""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

_loaded = None


def build(directory, *, timeout=90):
    global _loaded
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if (
        sys.platform != "darwin"
        or platform.machine() != "arm64"
        or torch.__version__.split("+")[0] != "2.14.0"
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("retained Metal recipes require Apple arm64, PyTorch 2.14.0 and MPS")
    if _loaded:
        return _loaded
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source, kernel = (
        Path(__file__).with_name("mps_native.mm"),
        Path(__file__).with_name("mps_kernels.metal"),
    )
    binary = directory / "native.dylib"
    command = [
        "clang++",
        "-O3",
        "-std=c++20",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
        "-x",
        "objective-c++",
        str(source),
        '-DLAB_METAL_SOURCE="' + str(kernel) + '"',
    ]
    command += ["-I" + p for p in include_paths()] + ["-L" + p for p in library_paths()]
    command += ["-Wl,-rpath," + p for p in library_paths()] + ["-ltorch", "-ltorch_cpu", "-lc10"]
    for framework in (
        "Metal",
        "Foundation",
        "MetalPerformanceShaders",
        "MetalPerformanceShadersGraph",
    ):
        command += ["-framework", framework]
    command += ["-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / "compile.log").write_text(result.stdout + result.stderr)
    receipt = dict(
        command=command,
        returncode=result.returncode,
        torch=torch.__version__,
        source_sha256=hashlib.sha256(source.read_bytes() + kernel.read_bytes()).hexdigest(),
    )
    (directory / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"Metal build failed: {directory / 'compile.log'}")
    torch.ops.load_library(str(binary))
    _loaded = receipt
    return receipt


def candidate(model, *, directory, contract):
    import importlib.metadata
    import torch
    from ._mps_models import patch

    if contract not in {"exact", "numerical"}:
        raise ValueError("contract must be exact or numerical")
    if importlib.metadata.version("transformers") != "4.43.3":
        raise ValueError("retained model wrappers require Transformers 4.43.3")
    if model.training or any(
        p.requires_grad or p.dtype != torch.float32 or p.device.type != "mps"
        for p in model.parameters()
    ):
        raise ValueError("requires eval MPS FP32 with immutable weights")
    if getattr(model.config, "model_type", None) != "llama" or model.config.hidden_size not in (
        576,
        960,
    ):
        raise ValueError("only the studied SmolLM2 model geometries are supported")
    build(directory)
    config = dict(
        metal_rope=True, metal_rms_tail=True, tail_mode=1, pack=True, qkv_pack=True, gqa=True
    )
    if contract == "numerical":
        config.update(metal_packed_silu=True, silu_mode=1, metal_down_residual=True)
    return patch(model, config)
