"""Fresh exact token/probability/KV and continuation checks; no speed promotion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


stamp(f"START estimate=under1min result={out}/results.json log={out}.log")
import mlx.core as mx
from lossless.retained.mlx_full_request import load_candidate
from lossless.adapters.mlx.batching import serial
from lossless.adapters.mlx.checks import diff, cache_diff

mx.set_default_device(mx.gpu)
operation = load_candidate(args.model, contract="exact")
model = operation._runner.model
rows = []
for prompts, counts in [([[17, 45, 67]], [3]), ([[67, 129, 347], [123, 27, 539]], [2, 4])]:
    a = serial(model, prompts, counts, True)
    b = operation.run(prompts, counts, True)
    assert a[0]["output_ids"] == b[0]["output_ids"]
    for i in range(len(prompts)):
        probability = diff(a[2][i], b[2][i])
        cache = cache_diff(a[1][i], b[1][i])
        assert probability["bitwise"] and cache["bitwise"], (probability, cache)
        tokens = mx.array([[173 + 31 * i]])
        av = model(tokens, cache=a[1][i])
        bv = model(tokens, cache=b[1][i])
        mx.eval(av, bv)
        continuation = diff(av, bv)
        continued_cache = cache_diff(a[1][i], b[1][i])
        assert continuation["bitwise"] and continued_cache["bitwise"]
        rows.append(
            dict(
                requests=len(prompts),
                row=i,
                probability=probability,
                cache=cache,
                continuation=continuation,
                continued_cache=continued_cache,
            )
        )
    stamp(f"PASS requests={len(prompts)} exact tokens/probabilities/KV/continuation")
(out / "results.json").write_text(json.dumps(dict(checks=rows, promotion=False), indent=2) + "\n")
stamp(f"COMPLETE result={out}/results.json log={out}.log")
