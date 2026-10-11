"""Local retained-backend conformance smoke. No downloads or authoring calls."""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
args = parser.parse_args()
import os, json
from datetime import datetime, timezone

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from transformers import AutoModelForCausalLM
from lossless.retained.mps import candidate

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(s):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {s}", flush=True)


stamp(f"START estimate=under2min result={out}/results.json log={out}.log")
torch.set_num_threads(1)
model = (
    AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    .to("mps")
    .eval()
    .requires_grad_(False)
)
rows = []
for contract in ["exact", "numerical"]:
    operation = candidate(model, directory=out / "native", contract=contract)
    with torch.inference_mode():
        for ids in [
            torch.tensor([[47, 83, 29]], device="mps"),
            torch.tensor([[1931, 87, 35, 1023]], device="mps"),
        ]:
            a = model(input_ids=ids, use_cache=True)
            b = operation(input_ids=ids, use_cache=True)
            if contract == "exact":
                assert torch.equal(a.logits, b.logits)
            else:
                torch.testing.assert_close(a.logits, b.logits, atol=2e-4, rtol=2e-4)
            cache_exact = all(
                torch.equal(x, y)
                for u, v in zip(a.past_key_values, b.past_key_values)
                for x, y in zip(u, v)
            )
            if contract == "exact":
                assert cache_exact
            rows.append(
                dict(
                    contract=contract,
                    length=ids.shape[1],
                    logits_exact=torch.equal(a.logits, b.logits),
                    cache_exact=cache_exact,
                    max_abs=float((a.logits - b.logits).abs().max()),
                )
            )
            # Reuse each model's own cache for a fresh, identical continuation.
            next_ids = torch.tensor([[173]], device="mps")
            a = model(input_ids=next_ids, past_key_values=a.past_key_values, use_cache=True)
            b = operation(input_ids=next_ids, past_key_values=b.past_key_values, use_cache=True)
            exact = a.logits.cpu().numpy().tobytes() == b.logits.cpu().numpy().tobytes()
            cache_exact = all(
                x.cpu().numpy().tobytes() == y.cpu().numpy().tobytes()
                for u, v in zip(a.past_key_values, b.past_key_values)
                for x, y in zip(u, v)
            )
            if contract == "exact":
                assert exact and cache_exact
            else:
                torch.testing.assert_close(a.logits, b.logits, atol=2e-4, rtol=2e-4)
                for u, v in zip(a.past_key_values, b.past_key_values):
                    for x, y in zip(u, v):
                        torch.testing.assert_close(x, y, atol=2e-4, rtol=2e-4)
            rows.append(
                dict(
                    contract=contract,
                    kind="cached_continuation",
                    logits_exact=exact,
                    cache_exact=cache_exact,
                )
            )
    stamp(f"PASS full MPS prefill contract={contract} cases=2")
    del operation
(out / "results.json").write_text(
    json.dumps(dict(torch=torch.__version__, checks=rows), indent=2) + "\n"
)
stamp(f"COMPLETE result={out}/results.json log={out}.log")
