"""Bounded CPU pilots for bicubic resizing and the SigLIP pairwise training loss.

Independent implementations of the mathematical loss, not a model-quality study.
No model weights, network access, product integration, or candidate selection.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time


ATOL, RTOL = 1e-6, 1e-5


def stamp(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def cases(split):
    if split not in {"discovery", "evaluation"}:
        raise ValueError("unknown split")
    sizes = (
        [(1, 3, 256, 320, 224, 224), (2, 16, 64, 80, 96, 96)]
        if split == "discovery"
        else [
            (1, 3, 287, 359, 224, 224),
            (2, 3, 96, 127, 192, 255),
            (1, 32, 63, 79, 101, 113),
            (1, 3, 1, 17, 7, 9),
        ]
    )
    return [
        dict(family="bicubic", shape=list(s[:4]), size=list(s[4:]), antialias=aa, layout=layout)
        for s in sizes
        for aa in [False, True]
        for layout in ["contiguous", "strided"]
    ] + [
        dict(family="siglip_loss", batch=n, dtype=dtype, backward=backward)
        for n in ([32, 256] if split == "discovery" else [17, 129, 513, 1024])
        for dtype in ["float32", "float64"]
        for backward in [False, True]
    ]


def input_tensor(case, seed, distribution):
    import torch

    shape = case.get("shape", [case.get("batch", 1)] * 2)
    dtype = getattr(torch, case.get("dtype", "float32"))
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=gen, dtype=dtype)
    if distribution == "extreme":
        x *= 80 if case["family"] == "siglip_loss" else 1000
    elif distribution == "structured":
        x = torch.arange(x.numel(), dtype=dtype).reshape(shape).remainder(3) - 1
        x.flatten()[::7] = -0.0
    elif distribution != "normal":
        raise ValueError("unknown distribution")
    if case.get("layout") == "strided":
        padded = torch.empty((*x.shape[:-1], x.shape[-1] * 2), dtype=dtype)
        padded[..., ::2] = x
        x = padded[..., ::2]
    return x


def bicubic(x, case, method):
    import torch
    from torch.nn.functional import interpolate

    options = dict(mode="bicubic", align_corners=False, antialias=case["antialias"])
    size = case["size"]
    if method == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    if method == "separable":
        x = interpolate(x, size=(x.shape[-2], size[1]), **options)
    elif method == "bilinear_control":
        options["mode"] = "bilinear"
    elif method not in {"stock", "channels_last"}:
        raise ValueError(method)
    # Public outputs are independent, contiguous NCHW tensors for every method.
    return interpolate(x, size=size, **options).contiguous()


def loss(x, method, sign=None, labels=None):
    import torch
    from torch.nn import functional as f

    if method == "stock":
        sign = torch.eye(x.shape[0], dtype=x.dtype) * 2 - 1
    if method in {"stock", "cached_sign"}:
        return -f.logsigmoid(sign * x).sum(dim=-1).mean()
    if method == "diagonal":
        values = f.logsigmoid(-x)
        values.diagonal().copy_(f.logsigmoid(x.diagonal()))
        return -values.sum(dim=-1).mean()
    if method == "bce_library":
        return f.binary_cross_entropy_with_logits(x, labels, reduction="sum") / x.shape[0]
    if method == "softmax_control":
        return f.cross_entropy(x, torch.arange(x.shape[0]))
    raise ValueError(method)


def operation(case, method):
    import torch

    started = time.perf_counter()
    if case["family"] == "bicubic":
        return lambda x: (bicubic(x, case, method),), 0.0, 0
    dtype = getattr(torch, case["dtype"])
    sign = torch.eye(case["batch"], dtype=dtype) * 2 - 1 if method == "cached_sign" else None
    labels = torch.eye(case["batch"], dtype=dtype) if method == "bce_library" else None
    setup = time.perf_counter() - started
    persistent = sum(t.numel() * t.element_size() for t in [sign, labels] if t is not None)

    def call(x):
        if case["backward"]:
            value = x.detach().requires_grad_(True)
            result = loss(value, method, sign, labels)
            return result.detach(), torch.autograd.grad(result, value)[0]
        with torch.no_grad():
            return (loss(x, method, sign, labels),)

    return call, setup, persistent


def difference(a, b):
    import torch

    metadata = a.shape == b.shape and a.dtype == b.dtype
    finite = metadata and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(exact=False, numerical=False, max_abs=None)
    av = a.detach().contiguous().reshape(-1).view(torch.uint8)
    bv = b.detach().contiguous().reshape(-1).view(torch.uint8)
    delta = (a - b).abs()
    return dict(
        exact=torch.equal(av, bv),
        numerical=bool((delta <= ATOL + RTOL * a.abs()).all()),
        max_abs=float(delta.max()),
    )


def summary(pairs):
    logs = [math.log(a / b) for a, b in pairs]
    rng = random.Random(190)
    boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(1000))
    return dict(
        speedup=math.exp(statistics.mean(logs)),
        descriptive_95=[boot[24], boot[974]],
    )


def measure(case, seed, repeats, round_seconds):
    names = (
        ["stock", "channels_last", "separable"]
        if case["family"] == "bicubic"
        else ["stock", "cached_sign", "bce_library", "diagonal"]
    )
    control = "bilinear_control" if case["family"] == "bicubic" else "softmax_control"
    ops = {name: operation(case, name) for name in [*names, control]}
    checks = {name: [] for name in ops}
    for i, distribution in enumerate(["normal", "extreme", "structured"]):
        x = input_tensor(case, seed + i, distribution)
        before = x.clone()
        reference = ops["stock"][0](x)
        for name, (call, _, _) in ops.items():
            outputs = call(x)
            snapshot = [y.clone() for y in outputs]
            # Exercise new data after retaining outputs; no result caching is legal.
            call(input_tensor(case, seed + 30 + i, distribution))
            values = [difference(a, b) for a, b in zip(reference, outputs)]
            checks[name].append(
                dict(
                    distribution=distribution,
                    outputs=values,
                    input_unchanged=difference(before, x)["exact"],
                    retained_outputs=all(
                        difference(a, b)["exact"] for a, b in zip(snapshot, outputs)
                    ),
                )
            )
    x = input_tensor(case, seed + 100, "normal")
    loops, samples = {}, {name: [] for name in names}
    first_call_seconds = {}
    for name in names:
        # These are post-validation calibration calls, not cold startup measurements.
        call = ops[name][0]
        start = time.perf_counter()
        call(x)
        used = time.perf_counter() - start
        first_call_seconds[name] = used
        loops[name] = max(1, min(100, math.ceil(round_seconds / max(used, 1e-9))))
    rng, orders = random.Random(seed), []
    for _ in range(repeats):
        order = names.copy()
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            call = ops[name][0]
            start = time.perf_counter()
            for _ in range(loops[name]):
                call(x)
            samples[name].append((time.perf_counter() - start) / loops[name])
    speed = {
        name: summary(list(zip(samples["stock"], samples[name])))
        for name in names
        if name != "stock"
    }
    return dict(
        case=case,
        checks=checks,
        samples_seconds=samples,
        orders=orders,
        loops=loops,
        calibration_seconds=first_call_seconds,
        setup_seconds={name: ops[name][1] for name in names},
        persistent_bytes={name: ops[name][2] for name in names},
        vs_stock=speed,
    )


def eligibility(row, name, relation):
    return all(
        c["input_unchanged"] and c["retained_outputs"] and all(v[relation] for v in c["outputs"])
        for c in row["checks"][name]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["discovery", "evaluation"], default="evaluation")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--round-seconds", type=float, default=0.005)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 3 or args.round_seconds <= 0:
        parser.error("positive threads/round duration and at least three repeats required")
    args.output.mkdir(parents=True, exist_ok=False)
    stamp(f"Results: {args.output.resolve()} | Log: {args.log.resolve()}")
    started = time.perf_counter()
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    suite = cases(args.split)
    plan = dict(
        study="VISION-MECHANISMS",
        split=args.split,
        seed=args.seed,
        repeats=args.repeats,
        round_seconds=args.round_seconds,
        threads=args.threads,
        runtime=dict(
            python=platform.python_version(), torch=torch.__version__, platform=platform.platform()
        ),
        torch_build=torch.__config__.show(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=suite,
        atol=ATOL,
        rtol=RTOL,
        selection="None. All methods frozen before discovery; new evaluation shapes/seeds, no tuning or selection on evaluation.",
        observables="Finite FP32/FP64 data, shape/dtype, unchanged inputs and retained results; loss plus first logit derivative for backward cases. Bicubic outputs contiguous NCHW; same antialias and align_corners=False.",
        timing="CPU synchronous whole operator call, allocations and layout copies included. Loss forward+backward includes autograd.grad. Input production and correctness checks excluded. No model, encoder, optimizer step, quality data, GPU or deployment claim.",
        baseline="Bicubic stock and channels-last including copies; SigLIP dense sign construction, cached signs, and native BCE with cached labels. Report all, no automatic promotion.",
        memory="Persistent label/sign tensor bytes only; no peak-memory measurement.",
    )
    write(args.output / "plan.json", plan)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for index, case in enumerate(suite):
        row = measure(case, args.seed + index * 1000, args.repeats, args.round_seconds)
        rows.append(row)
        write(args.output / "results.json", rows)
        stamp(f"{index + 1}/{len(suite)} {case} speed={row['vs_stock']}")
    write(
        args.output / "completion.json",
        dict(seconds=time.perf_counter() - started, cases=len(rows)),
    )
    stamp(f"Complete. Results: {args.output.resolve()} | Log: {args.log.resolve()}")


if __name__ == "__main__":
    main()
