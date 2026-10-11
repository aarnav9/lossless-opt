"""Local retained-backend conformance smoke. No downloads or authoring calls."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", type=Path)
args = parser.parse_args()
import os, json
from datetime import datetime, timezone

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from lossless.retained import torch_cpu
import torch
from torch import nn

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(s):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {s}", flush=True)


stamp(f"START estimate=under2min result={out}/results.json log={out}.log")
torch.set_num_threads(1)
torch.manual_seed(803301)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64))
        self.gate = nn.Linear(64, 96)
        self.up = nn.Linear(64, 96)
        self.down = nn.Linear(96, 64)

    def forward(self, x, r):
        z = x + r
        y = z * torch.rsqrt((z * z).mean(dim=-1, keepdim=True) + 1e-5) * self.weight
        return self.down(torch.nn.functional.silu(self.gate(y)) * self.up(y)) + z


model = MLP().eval().requires_grad_(False)
results = []
for contract in ["exact", "numerical"]:
    candidate = torch_cpu.regions(model, directory=out / "native", contract=contract)
    assert candidate.changes
    with torch.inference_mode():
        for size in [1, 3, 17, 129]:
            x, r = torch.randn(size, 64), torch.randn(size, 64)
            before = x.clone()
            expected = model(x, r)
            actual = candidate(x, r)
            exact = torch.equal(expected, actual)
            if contract == "exact":
                assert exact, (size, (actual - expected).abs().max())
            else:
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
            assert torch.equal(x, before)
            results.append(
                dict(
                    contract=contract,
                    size=size,
                    exact=exact,
                    max_abs=float((actual - expected).abs().max()),
                )
            )
    stamp(f"PASS FX regions contract={contract} cases=4")
# The full pretrained adapter has its original recorded dependencies available.
path = args.model
if path is not None:
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .requires_grad_(False)
    )
    candidate = torch_cpu.legacy_decoder(
        model, directory=out / "native", contract="numerical", packed=False
    )
    with torch.inference_mode():
        for token in [47, 1931]:
            ids = torch.tensor([[token]])
            positions = torch.zeros_like(ids)
            expected = model(
                input_ids=ids, position_ids=positions, use_cache=False, return_dict=False
            )[0]
            actual = candidate(ids, positions)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
            results.append(
                dict(
                    kind="legacy_full_forward",
                    token=token,
                    exact=torch.equal(actual, expected),
                    max_abs=float((actual - expected).abs().max()),
                )
            )
    stamp("PASS full pretrained CPU forward cases=2; no speed qualification")
(out / "results.json").write_text(
    json.dumps(dict(torch=torch.__version__, checks=results), indent=2) + "\n"
)
stamp(f"COMPLETE result={out}/results.json log={out}.log")
