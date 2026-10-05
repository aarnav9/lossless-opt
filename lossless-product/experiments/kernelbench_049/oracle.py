"""Independent CPU PyTorch oracle using the unchanged vendored upstream Model classes."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from protocol import digest, read, write
from lossless._native.common import stamp


def generate(cases, output):
    import torch

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent / "upstream"
    manifest = read(root / "manifest.json")
    sources = {r["task"]: r for r in manifest["tasks"]}
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in cases:
        record = sources[case["task"]]
        path = root / record["source"]
        assert digest(path) == record["sha256"]
        spec = importlib.util.spec_from_file_location("upstream_" + case["task"], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        torch.manual_seed(case["seed"])
        model = mod.Model(*case["init"]).float().eval()
        arrays = {"p__" + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
        rng = np.random.default_rng(case["seed"])
        for variant in range(5):
            inputs = []
            for index, shape in enumerate(case["shapes"]):
                if variant == 0:
                    v = rng.uniform(0, 1, shape)
                elif variant == 1:
                    v = rng.normal(0, 1, shape)
                elif variant == 2:
                    v = np.zeros(shape)
                elif variant == 3:
                    v = ((np.arange(np.prod(shape)).reshape(shape) % 2) * 2 - 1) * 3.0
                    v = v + rng.normal(0, 1e-3, shape)
                else:
                    v = 1 + rng.normal(0, 1e-4, shape)
                v = np.asarray(v, dtype=np.float32)
                arrays[f"x{variant}_{index}"] = v
                inputs.append(torch.from_numpy(v.copy()))
            with torch.inference_mode():
                answer = model(*inputs).detach().cpu().numpy().copy()
            assert np.isfinite(answer).all()
            arrays[f"y{variant}"] = answer
        file = output / (case["id"] + ".npz")
        np.savez(file, **arrays)
        records.append(
            dict(
                case=case,
                file=file.name,
                sha256=digest(file),
                parameters_sha256=hashlib.sha256(
                    b"".join(arrays[k].tobytes() for k in sorted(arrays) if k.startswith("p__"))
                ).hexdigest(),
                output_shapes=[list(arrays[f"y{v}"].shape) for v in range(5)],
            )
        )
        stamp(f"049 ORACLE case={case['id']} complete result={file}")
    write(
        output / "index.json", dict(torch=torch.__version__, numpy=np.__version__, records=records)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    generate(read(a.cases), a.output)
