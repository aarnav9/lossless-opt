"""CPU speculative-decoding pilot; uses stock Transformers generation throughout."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time


def cases(split):
    if split == "discovery":
        return {
            "explain": "Explain why leaves change color in autumn in simple terms.",
            "code": "Write a Python function that returns the first n Fibonacci numbers.",
            "copy": "Repeat the following sentence ten times: The red fox runs past the tree.",
        }
    if split != "evaluation":
        raise ValueError("unknown split")
    return {
        "explain": "Explain how a refrigerator keeps food cold in simple terms.",
        "code": "Write a Python function that checks whether a string is a palindrome.",
        "copy": "Repeat the following sentence ten times: A blue bird sits beside the lake.",
        "summarize": (
            "Summarize this paragraph in three sentences: A community garden opened in "
            "a vacant lot last spring. Volunteers planted tomatoes, beans, and flowers. "
            "Local students installed rain barrels to conserve water. Neighbors share "
            "the harvest and meet on Saturdays to maintain the paths. Next year they "
            "plan to add fruit trees and a small greenhouse."
        ),
    }


def run(model, inputs, count, method, assistant=None):
    import torch

    options = {}
    if method.startswith("draft_"):
        width = int(method.split("_")[1])
        assistant.generation_config.num_assistant_tokens = width
        assistant.generation_config.num_assistant_tokens_schedule = "constant"
        assistant.generation_config.assistant_confidence_threshold = 0.0
        options["assistant_model"] = assistant
    elif method.startswith("lookup_"):
        options["prompt_lookup_num_tokens"] = int(method.split("_")[1])
    elif method != "serial":
        raise ValueError("unknown decoding method")
    with torch.inference_mode():
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=count,
            do_sample=False,
            use_cache=True,
            # Empty stop set keeps fixed work and avoids lookup's None-EOS bug.
            eos_token_id=[],
            forced_eos_token_id=None,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_scores=True,
            **options,
        )
        expected_length = inputs["input_ids"].shape[1] + count
        surplus = output.sequences.shape[1] - expected_length
        if method.startswith("lookup_") and surplus > 0:
            # Transformers 4.57.6 lookup can overshoot; charge work and crop all outputs.
            output.sequences = output.sequences[:, :expected_length]
            output.scores = output.scores[:count]
            output.past_key_values.crop(expected_length - 1)
        logprobs = torch.log_softmax(torch.stack(output.scores), dim=-1)
        elapsed = time.perf_counter() - start
    assert output.sequences.shape[1] == inputs["input_ids"].shape[1] + count, (
        method,
        count,
        output.sequences.shape[1] - inputs["input_ids"].shape[1],
    )
    return {"output": output, "logprobs": logprobs, "seconds": elapsed, "surplus_tokens": surplus}


def tensor_diff(a, b):
    import torch

    shape = a.shape == b.shape and a.dtype == b.dtype
    return {
        "shape_dtype_equal": shape,
        "bitwise": shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
        "max_abs": float((a - b).abs().max()) if shape else None,
    }


def compare(reference, candidate, model):
    import torch

    a, b = reference["output"], candidate["output"]
    ac, bc = a.past_key_values.to_legacy_cache(), b.past_key_values.to_legacy_cache()
    kv = [tensor_diff(x, y) for al, bl in zip(ac, bc) for x, y in zip(al, bl)]
    cache_shape = len(ac) == len(bc) and all(len(x) == len(y) for x, y in zip(ac, bc))
    continuation = []
    with torch.inference_mode():
        caches = [copy.deepcopy(a.past_key_values), copy.deepcopy(b.past_key_values)]
        token = a.sequences[:, -1:]
        for _ in range(3):
            logits = [model(token, past_key_values=c, use_cache=True).logits[:, -1] for c in caches]
            continuation.append(tensor_diff(*logits))
            token = logits[0].argmax(-1, keepdim=True)
    result = {
        "tokens_equal": torch.equal(a.sequences, b.sequences),
        "logprobs": tensor_diff(reference["logprobs"], candidate["logprobs"]),
        "kv_bitwise": cache_shape and bool(kv) and all(x["bitwise"] for x in kv),
        "kv_max_abs": max(x["max_abs"] for x in kv) if cache_shape else None,
        "kv_length": [a.past_key_values.get_seq_length(), b.past_key_values.get_seq_length()],
        "continuation_bitwise": all(x["bitwise"] for x in continuation),
        "continuation_max_abs": max(x["max_abs"] for x in continuation),
    }
    result["exact"] = all(
        [
            result["tokens_equal"],
            result["logprobs"]["bitwise"],
            result["kv_bitwise"],
            result["continuation_bitwise"],
        ]
    )
    return result


def speed_summary(samples):
    ratios = [a / b for a, b in samples]
    rng = random.Random(50)
    boots = sorted(
        math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(2000)
    )
    return {
        "paired_geomean": math.exp(statistics.mean(map(math.log, ratios))),
        "descriptive_bootstrap_95": [boots[49], boots[1949]],
        "serial_median_seconds": statistics.median(a for a, _ in samples),
        "candidate_median_seconds": statistics.median(b for _, b in samples),
    }


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.threads, args.tokens, args.repeats) < 1:
        parser.error("threads, tokens and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(50)
    models = json.loads(args.models.read_text())
    started = time.perf_counter()
    target, assistant = [
        AutoModelForCausalLM.from_pretrained(
            models[size]["path"],
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .cpu()
        .eval()
        for size in ["360M", "135M"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(models["360M"]["path"], local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(models["135M"]["path"], local_files_only=True)
    assert tokenizer.get_vocab() == draft_tokenizer.get_vocab(), "draft tokenizer differs"
    plan = {
        "models": models,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "numpy"]},
        "threads": args.threads,
        "dtype": "float32",
        "device": "cpu",
        "attention": "sdpa",
        "tokens": args.tokens,
        "repeats": args.repeats,
        "cases": {s: cases(s) for s in ["discovery", "evaluation"]},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "setup_seconds": time.perf_counter() - started,
        "contract": "Exact tokens, full logprob bits, active KV bits and forced continuation logits",
        "timing_scope": "Fresh-cache generation plus full log_softmax; excludes loading, tokenization and validation",
    }
    write(args.output / "plan.json", plan)
    inputs = {
        split: {
            key: tokenizer(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                return_tensors="pt",
            )
            for key, text in cases(split).items()
        }
        for split in ["discovery", "evaluation"]
    }
    methods = ["draft_2", "draft_4", "draft_8", "lookup_4", "lookup_8"]
    # Warm each execution path; every call still begins with a fresh KV cache.
    for method in ["serial", *methods]:
        run(target, inputs["discovery"]["explain"], 8, method, assistant)
    discovery = {method: [] for method in methods}
    rng = random.Random(50)
    for key, encoded in inputs["discovery"].items():
        order = methods.copy()
        rng.shuffle(order)
        for method in order:
            pair = ["serial", method]
            rng.shuffle(pair)
            values = {m: run(target, encoded, 32, m, assistant) for m in pair}
            row = {
                "key": key,
                "speedup": values["serial"]["seconds"] / values[method]["seconds"],
                "checks": compare(values["serial"], values[method], target),
            }
            discovery[method].append(row)
            print(
                "discovery",
                method,
                key,
                round(row["speedup"], 3),
                row["checks"]["exact"],
                flush=True,
            )
    selected = [
        max(
            (m for m in methods if m.startswith(prefix)),
            key=lambda m: statistics.mean(math.log(r["speedup"]) for r in discovery[m]),
        )
        for prefix in ["draft_", "lookup_"]
    ]
    # Freeze both choices before timing any evaluation input.
    write(
        args.output / "selection.json",
        {
            "methods": selected,
            "discovery": discovery,
            "selection_rule": "Fastest per drafting family, irrespective of exactness, for diagnosis only",
        },
    )
    records = []
    for key, encoded in inputs["evaluation"].items():
        for method in selected:
            lengths = []
            hook = target.register_forward_pre_hook(
                lambda module, pos, kw: lengths.append(int(kw["input_ids"].shape[1])),
                with_kwargs=True,
            )
            try:
                candidate = run(target, encoded, args.tokens, method, assistant)
            finally:
                hook.remove()
            reference = run(target, encoded, args.tokens, "serial", assistant)
            row = {
                "key": key,
                "method": method,
                "checks": compare(reference, candidate, target),
                "target_forward_lengths": lengths,
                "tokens_per_target_forward": args.tokens / len(lengths),
                "surplus_tokens": candidate["surplus_tokens"],
                "samples": [],
                "orders": [],
                "timed_tokens_equal": True,
                "output_text": tokenizer.decode(
                    reference["output"].sequences[0, encoded["input_ids"].shape[1] :]
                ),
            }
            expected = reference["output"].sequences.clone()
            del reference, candidate
            for _ in range(args.repeats):
                order = ["serial", method]
                rng.shuffle(order)
                values = {m: run(target, encoded, args.tokens, m, assistant) for m in order}
                row["samples"].append([values["serial"]["seconds"], values[method]["seconds"]])
                row["orders"].append(order)
                row["timed_tokens_equal"] &= all(
                    torch.equal(v["output"].sequences, expected) for v in values.values()
                )
                del values
            row["speed"] = speed_summary(row["samples"])
            records.append(row)
            write(args.output / "records.json", records)
            print(
                "evaluation",
                method,
                key,
                round(row["speed"]["paired_geomean"], 3),
                row["checks"],
                flush=True,
            )
    summary = {}
    for method in selected:
        rows = [r for r in records if r["method"] == method]
        exact = all(r["checks"]["exact"] and r["timed_tokens_equal"] for r in rows)
        summary[method] = {
            "geomean_speedup": math.exp(
                statistics.mean(math.log(r["speed"]["paired_geomean"]) for r in rows)
            ),
            "tokens_equal": all(
                r["checks"]["tokens_equal"] and r["timed_tokens_equal"] for r in rows
            ),
            "exact": exact,
            "passes_exact_speed_gate": exact
            and all(r["speed"]["descriptive_bootstrap_95"][0] > 1 for r in rows),
        }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
