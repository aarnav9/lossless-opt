"""Opt-in native PyTorch region candidates from campaigns 010/011.

No global model patching, downloads or implicit compilation on import. Building
and whole-model qualification are explicit; local kernel equality alone is not
acceptance. The legacy full decoder has separate pinned-runtime admission.
"""

import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

_loaded = None


def build(directory, *, timeout=90):
    """Compile the shipped source into a new caller-owned build directory."""
    global _loaded
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if _loaded is not None:
        return dict(_loaded, already_loaded=True)
    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("native CPU builder supports macOS/Linux")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("torch_native.cpp")
    binary = directory / ("native.dylib" if sys.platform == "darwin" else "native.so")
    command = [
        "clang++",
        "-O3",
        "-std=c++17",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-shared",
        "-fPIC",
        str(source),
    ]
    command += ["-I" + p for p in include_paths()] + ["-L" + p for p in library_paths()]
    if sys.platform == "linux":
        command += [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    command += ["-Wl,-rpath," + p for p in library_paths()] + [
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / "compile.log").write_text(result.stdout + result.stderr)
    receipt = dict(
        command=command,
        returncode=result.returncode,
        torch=torch.__version__,
        platform=platform.platform(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    (directory / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"native CPU compilation failed; inspect {directory / 'compile.log'}")
    torch.ops.load_library(str(binary))
    receipt["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (directory / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    _loaded = receipt
    return receipt


def regions(model, *, directory, contract, fuse_mlp=True):
    """Copy and rewrite the narrow frozen RMS/MLP FX pattern.

    The numerical candidate uses the retained vectorized reduction. The exact
    candidate preserves ATen arithmetic and must pass bitwise whole-call tests
    for its actual graph/runtime. Training, gradients and noncontiguous inputs
    use the independent original model. Caller must keep weights immutable.
    """
    import torch
    from ._torch_rewrite import rewrite

    if contract not in {"exact", "numerical"}:
        raise ValueError("contract must be exact or numerical")
    if model.training or any(
        p.requires_grad or p.device.type != "cpu" or p.dtype != torch.float32
        for p in model.parameters()
    ):
        raise ValueError("requires an eval CPU FP32 model with frozen weights")
    build(directory)
    original = copy.deepcopy(model)
    graph = torch.fx.symbolic_trace(copy.deepcopy(model))
    changes = rewrite(graph, dict(fuse_mlp=fuse_mlp, algorithm=2 if contract == "exact" else 0))

    class Candidate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reference, self.optimized = original, graph
            self.changes, self.contract = changes, contract

        def forward(self, *args, **kwargs):
            admitted = (
                not kwargs
                and torch.is_inference_mode_enabled()
                and all(
                    isinstance(t, torch.Tensor)
                    and t.device.type == "cpu"
                    and t.dtype == torch.float32
                    and t.ndim == 2
                    and t.is_contiguous()
                    and all(t.shape)
                    for t in args
                )
            )
            return self.optimized(*args) if admitted else self.reference(*args, **kwargs)

    return Candidate().eval()


def legacy_decoder(model, *, directory, contract, packed=False, compile_outer=False):
    """Campaign-011 no-cache decoder adapter for its original Transformers API.

    Kept as an explicit numerical-contract candidate, even though final sampled
    logits happened to match bytes. Full generation was never qualified.
    """
    import importlib.metadata
    import torch
    from ._torch_decoder import patched_model

    if contract != "numerical":
        raise ValueError("campaign 011 used a numerical whole-model acceptance contract")
    version = importlib.metadata.version("transformers")
    if version != "4.43.3" or torch.__version__.split("+")[0] != "2.2.2":
        raise ValueError(
            "legacy decoder requires the recorded PyTorch 2.2.2 / Transformers 4.43.3 API"
        )
    if model.training or any(
        p.requires_grad or p.dtype != torch.float32 or p.device.type != "cpu"
        for p in model.parameters()
    ):
        raise ValueError("legacy decoder requires immutable eval CPU FP32 parameters")
    if (
        getattr(model.config, "model_type", None) != "llama"
        or model.config.hidden_size != 576
        or model.config.num_hidden_layers != 30
    ):
        raise ValueError("legacy forward scope is the studied 135M SmolLM2 geometry")
    build(directory)

    class Forward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = model

        def forward(self, ids, positions):
            if getattr(self, "direct_no_cache", False):
                h = self.lm.model.embed_tokens(ids)
                embeddings = self.lm.model.rotary_emb(h, positions)
                for layer in self.lm.model.layers:
                    h = layer(
                        h, position_ids=positions, position_embeddings=embeddings, use_cache=False
                    )[0]
                return self.lm.lm_head(self.lm.model.norm(h)).float()
            return self.lm(
                input_ids=ids, position_ids=positions, use_cache=False, return_dict=False
            )[0]

    adapted = patched_model(
        Forward().eval(),
        dict(
            algorithm=2,
            fuse_mlp=True,
            fuse_attention=True,
            merge_projections=packed,
            protect_outer=True,
            direct_no_cache=True,
        ),
    )
    if compile_outer:
        adapted = torch.compile(adapted, fullgraph=True)

    def forward(ids, positions):
        if (
            not torch.is_inference_mode_enabled()
            or ids.dtype != torch.int64
            or ids.device.type != "cpu"
            or ids.ndim != 2
            or ids.shape[1] != 1
            or positions.shape != ids.shape
            or positions.dtype != torch.int64
            or positions.device.type != "cpu"
        ):
            return model(input_ids=ids, position_ids=positions, use_cache=False, return_dict=False)[
                0
            ]
        return adapted(ids, positions)

    return forward
