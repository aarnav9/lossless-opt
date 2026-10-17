"""Local, sequential compiler-layer pilot on the retained ModPoly application.

This is research code. It never patches the installed product or upstream files.
All variants retain the loaded NumPy BLAS functions and floating-point flags.
"""

import argparse
import ast
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
import types

ROOT = Path(__file__).resolve().parents[3]
RESOURCE = ROOT / "lossless-product/src/lossless/retained"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(ROOT / "lossless-product/src"))

import numpy as np


def stamp(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


SIGNATURE = """void *gemv_address,void *dot_address,const double *source,
double *b0,double *b1,const double *weight,double *clipped,double *scratch,
const double *pinv,const double *vander,double *coef,double *history,
long long n,long long order,long long pinv_ld,long long vander_ld,
int pinv_layout,int vander_layout,int unit,int maximum,double tol,double minimum_norm"""
ARGUMENTS = "gemv_address,dot_address,source,b0,b1,weight,clipped,scratch,pinv,vander,coef,history,n,order,pinv_ld,vander_ld,pinv_layout,vander_layout,unit,maximum,tol,minimum_norm"


def pointwise(simd=False):
    # In-place clipping and separate-source clipping get different alias contracts.
    # These contracts follow from the retained Python body's owned allocations;
    # source==clipped is the sole intentional alias. No caller array is restricted
    # against another caller array. This is not a general public C ABI.
    output = []
    for inplace in (False, True):
        args = "const double *restrict s," if not inplace else ""
        args += "const double *restrict old,const double *restrict w,double *restrict y,double *restrict z,long long n,int unit,int store"
        source = "y" if inplace else "s"
        loop = ""
        if simd:
            loop += f"""
            for(;j+1<n;j+=2) {{
                float64x2_t a=vld1q_f64({source}+j), b=vld1q_f64(old+j);
                float64x2_t v=vbslq_f64(vcltq_f64(a,b),a,b);
                if(store)vst1q_f64(y+j,v);
                if(!unit)vst1q_f64(z+j,vmulq_f64(vld1q_f64(w+j),v));
            }}
            """
        loop += f"""
        for(;j<n;j++){{double v={source}[j]<old[j]?{source}[j]:old[j];
            if(store)y[j]=v; if(!unit)z[j]=w[j]*v;}}
        """
        output.append(
            f"static inline __attribute__((always_inline)) void clip_{int(inplace)}({args}){{long long j=0;{loop}}}"
        )
    return "\n".join(output)


def candidate_source(kind):
    original = (RESOURCE / "modpoly.c").read_text()
    if kind == "baseline":
        return original
    header = original[: original.index("int iterate(")]
    if kind == "simd":
        header += "#include <arm_neon.h>\n"
    header += pointwise(kind == "simd") + "\n"
    # Written native buffers are independently allocated by the retained caller.
    signature = SIGNATURE
    for name in ("b0", "b1", "scratch", "coef", "history"):
        signature = signature.replace(f"double *{name}", f"double *restrict {name}")
    body = original[original.index("    gemv_fn") :]
    begin = body.index("        if(unit)")
    end = body.index("        gemv(pinv_layout")
    body = body[:begin] + "@CLIP@\n" + body[end:]
    if kind == "alias":
        clip = """        if(source==clipped)clip_1(old,weight,clipped,scratch,n,unit,1);
        else clip_0(source,old,weight,clipped,scratch,n,unit,1);"""
        return header + f"int iterate({signature}) {{\n" + body.replace("@CLIP@", clip)
    # Whole-loop specialization exposes constants across the iteration boundary.
    functions = []
    for unit in (0, 1):
        for inplace in (0, 1):
            store = int(unit or inplace or kind == "specialize")
            args = "" if inplace else "source,"
            clip = f"        clip_{inplace}({args}old,weight,clipped,scratch,n,{unit},{store});"
            specialized = body.replace("@CLIP@", clip).replace(
                "unit?clipped:scratch", "clipped" if unit else "scratch"
            )
            functions.append(f"static int loop_{unit}_{inplace}({signature}) {{\n" + specialized)
    dispatch = []
    for unit in (0, 1):
        for inplace in (0, 1):
            dispatch.append(
                f"if (!!unit=={unit} && (source==clipped)=={inplace}) return loop_{unit}_{inplace}({ARGUMENTS});"
            )
    return (
        header
        + "\n".join(functions)
        + f"\nint iterate({SIGNATURE}){{"
        + "\n".join(dispatch)
        + "\nreturn 0; }\n"
    )


def build(out, name, kind, flags=()):
    folder = out / name
    folder.mkdir()
    source = folder / "helper.c"
    source.write_text(candidate_source(kind))
    binary = folder / ("helper.dylib" if sys.platform == "darwin" else "helper.so")
    base = ["clang", "-O3", "-std=c11", "-fno-fast-math", "-ffp-contract=off", "-fPIC", *flags]
    commands = [
        base
        + [
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            str(source),
            "-o",
            str(binary),
            "-lm",
        ],
        base + ["-S", "-emit-llvm", str(source), "-o", str(folder / "helper.ll")],
        base + ["-S", str(source), "-o", str(folder / "helper.s")],
    ]
    start = time.perf_counter()
    logs = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        logs.append(
            dict(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            write(folder / "build.json", logs)
            raise RuntimeError(f"build failed: {folder}/build.json")
    write(folder / "build.json", logs)
    lib = ctypes.CDLL(str(binary))
    fn = lib.iterate
    fn.argtypes = (
        [ctypes.c_void_p] * 12
        + [ctypes.c_longlong] * 4
        + [ctypes.c_int] * 4
        + [ctypes.c_double] * 2
    )
    fn.restype = ctypes.c_int
    return dict(
        name=name,
        kind=kind,
        flags=list(flags),
        function=fn,
        library=lib,
        build_seconds=time.perf_counter() - start,
        binary_bytes=binary.stat().st_size,
        source_sha256=sha(source),
        binary_sha256=sha(binary),
    )


def method(body, module, symbol):
    from lossless.applications.patches import function

    node = copy.deepcopy(function(Path(module.__file__).read_text(), symbol)[0])
    node.body = ast.parse(body).body
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(module))
    exec(compile(tree, f"<pilot:{symbol}>", "exec"), namespace)
    return namespace[node.name]


def setup_application(project):
    sys.path.insert(0, str(project))
    import pybaselines
    import pybaselines.polynomial as polynomial
    import pybaselines._algorithm_setup as setup
    from lossless.retained import modpoly_native
    from lossless.applications import retained_plan

    retained_plan(project, workers=())  # pinned source hash admission
    if modpoly_native._gemv is None:
        raise RuntimeError("This pilot requires the loaded NumPy OpenBLAS ILP64 symbols")
    recipe = json.loads((RESOURCE / "modpoly.json").read_text())
    recalc = recipe["source_edits"][0]
    setup._PolyHelper.recalc_vandermonde = method(
        recalc["body"], setup, "_PolyHelper.recalc_vandermonde"
    )
    native_method = method(recipe["compiled_edit"]["body"], polynomial, "_Polynomial.modpoly")
    python_method = method(recipe["source_edits"][1]["body"], polynomial, "_Polynomial.modpoly")
    factories = {}
    for name, implementation in (("native", native_method), ("python", python_method)):
        subclass = type(f"Pilot_{name}", (pybaselines.Baseline,), {"modpoly": implementation})
        namespace = {}
        exec(
            compile((RESOURCE / "spectrum_workload.py").read_text(), "<spectrum_workload>", "exec"),
            namespace,
        )
        namespace["Baseline"] = subclass
        factories[name] = namespace["make_processor"]
    binding = types.ModuleType("_lossless_native_modpoly")
    sys.modules[binding.__name__] = binding
    return factories, binding


def digest(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, dict):
        return {k: digest(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest(v) for v in value]
    return value


def compare(a, b):
    if isinstance(a, np.ndarray):
        structure = isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype
        return structure and a.tobytes() == b.tobytes(), structure and bool(
            np.allclose(a, b, atol=1e-10, rtol=1e-8, equal_nan=True)
        )
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False, False
        checks = [compare(a[k], b[k]) for k in a]
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return False, False
        checks = [compare(x, y) for x, y in zip(a, b)]
    else:
        return a == b, a == b
    return all(c[0] for c in checks), all(c[1] for c in checks)


def specifications(final=False):
    delta = 38 if final else 0
    result = [
        dict(id="small", size=383 + delta, batch=2),
        dict(id="weighted", size=8191 + delta, batch=2, weighted=True),
        dict(id="long", size=32749 + delta, batch=2),
        dict(id="original_weighted", size=6143 + delta, batch=2, weighted=True, original=True),
        dict(
            id="branches",
            size=1051 + delta,
            batch=3,
            weighted=True,
            original=True,
            mask=True,
            shuffled=True,
        ),
        dict(id="batch", size=1531 + delta, batch=12),
        dict(id="flat", size=577 + delta, batch=2, flat=True),
    ]
    if final:
        result += [
            dict(id="strided", size=887, batch=2, original=True, strided=True),
            dict(id="float32", size=1237, batch=2, float32=True),
        ]
    return result


def records(spec, seed):
    rng = np.random.default_rng(seed)
    result = []
    for index in range(spec["batch"]):
        x = np.linspace(-2, 3, spec["size"])
        t = (x + 2) / 5
        y = 0.7 + 1.8 * np.exp(-2.4 * t) + 0.2 * t * t
        y += rng.uniform(0.5, 3) * np.exp(-(((t - rng.uniform(0.2, 0.8)) / 0.07) ** 2))
        y += rng.normal(0, 0.015, len(x))
        if spec.get("flat"):
            y[:] = 1.0
        if spec.get("zeros"):
            y[:] = 0.0
            y[::2] = -0.0
        y *= spec.get("scale", 1.0)
        options = dict(
            poly_order=spec.get("order", 3),
            max_iter=spec.get("maximum", 120),
            tol=spec.get("tol", 1e-3),
            return_coef=True,
            use_original=spec.get("original", False),
            mask_initial_peaks=spec.get("mask", False),
        )
        if spec.get("weighted"):
            options["weights"] = rng.uniform(0.15, 1, len(x))
            if spec.get("zero_weights"):
                options["weights"][::5] = 0
        if spec.get("shuffled"):
            indices = rng.permutation(len(x))
            x, y = x[indices], y[indices]
            if "weights" in options:
                options["weights"] = options["weights"][indices]
        if spec.get("float32"):
            y = y.astype(np.float32)
        if spec.get("strided"):
            backing = np.empty(len(y) * 2, dtype=y.dtype)
            backing[::2] = y
            y = backing[::2]
        if spec.get("readonly"):
            x.flags.writeable = y.flags.writeable = False
        result.append(dict(x=x, y=y, options=options))
    return result


def activate(variant, binding):
    if variant.get("function") is not None:
        binding.iterate = variant["function"]


def run_calls(variant, calls, factories, binding, *, correctness=False):
    activate(variant, binding)
    native_calls = []
    if correctness and variant.get("function") is not None:

        def observed(*args):
            steps = variant["function"](*args)
            native_calls.append(
                dict(
                    size=args[12],
                    order=args[13],
                    unit=args[18],
                    steps=steps,
                    in_place=args[2] == args[6],
                )
            )
            return steps

        binding.iterate = observed
    start = time.perf_counter_ns()
    processor = factories[variant.get("mode", "native")]()
    factory_ns = time.perf_counter_ns() - start
    # First request costs include lazy fitter/matrix preparation. Subsequent
    # requests retain normal metadata caches but receive changing signal values.
    before = digest(calls) if correctness else None
    start = time.perf_counter_ns()
    first = processor(calls[0])
    first_ns = time.perf_counter_ns() - start
    saved = digest(first) if correctness else None
    start = time.perf_counter_ns()
    outputs = [processor(call) for call in calls[1:]]
    warm_ns = time.perf_counter_ns() - start
    if correctness:
        if digest(calls) != before or digest(first) != saved:
            raise AssertionError("input mutation or retained output changed")
        arrays = [r["baseline"] for group in [first, *outputs] for r in group]
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :]):
            raise AssertionError("returned baselines share storage")
    return dict(
        factory_ns=factory_ns,
        first_ns=first_ns,
        warm_ns=warm_ns,
        complete_ns=factory_ns + first_ns + warm_ns,
        native_calls=native_calls,
    ), [first, *outputs]


def qualify(reference, candidate, specs, factories, binding, seed):
    rows = []
    for i, spec in enumerate(specs):
        calls = [records(spec, seed + i * 100 + j) for j in range(4)]
        a_receipt, a = run_calls(reference, calls, factories, binding, correctness=True)
        _, repeated = run_calls(reference, calls, factories, binding, correctness=True)
        if not compare(a, repeated)[0]:
            raise AssertionError(f"reference is not bitwise repeatable: {spec['id']}")
        b_receipt, b = run_calls(candidate, calls, factories, binding, correctness=True)
        exact, numerical = compare(a, b)
        rows.append(
            dict(
                id=spec["id"],
                exact=exact,
                numerical=numerical,
                reference_native_calls=a_receipt["native_calls"],
                candidate_native_calls=b_receipt["native_calls"],
                reference_digest=digest(a),
                candidate_digest=digest(b),
            )
        )
    return dict(
        exact=all(r["exact"] for r in rows),
        numerical=all(r["numerical"] for r in rows),
        inputs_and_outputs_owned=True,
        reference_repeatable=True,
        cases=rows,
    )


def benchmark(reference, candidate, specs, factories, binding, seed, pairs):
    result = []
    for i, spec in enumerate(specs):
        rows = []
        # Warm interpreter and both implementations outside measured requests.
        warm = [records(spec, seed + i * 1000 + j) for j in range(3)]
        for variant in (reference, candidate):
            run_calls(variant, warm, factories, binding)
        for pair in range(pairs):
            calls = [records(spec, seed + 100000 + i * 10000 + pair * 10 + j) for j in range(5)]
            order = ["reference", "candidate"]
            if pair % 2:
                order.reverse()
            if i % 2:
                order.reverse()
            values = {}
            for name in order:
                values[name], _ = run_calls(
                    reference if name == "reference" else candidate, calls, factories, binding
                )
            rows.append(dict(order=order, **values))
        result.append(dict(id=spec["id"], pairs=rows))
        stamp(f"TIMING {candidate['name']} case={spec['id']} pairs={pairs}")
    metrics = {}
    for metric in ("warm_ns", "first_ns", "complete_ns"):
        case_ratios = {
            r["id"]: statistics.geometric_mean(
                p["reference"][metric] / p["candidate"][metric] for p in r["pairs"]
            )
            for r in result
        }
        round_ratios = [
            statistics.geometric_mean(
                r["pairs"][p]["reference"][metric] / r["pairs"][p]["candidate"][metric]
                for r in result
            )
            for p in range(pairs)
        ]
        wins = sum(v >= 1.02 for v in round_ratios)
        metrics[metric] = dict(
            speedup=statistics.geometric_mean(case_ratios.values()),
            cases=case_ratios,
            rounds=round_ratios,
            sign_p=sum(math.comb(pairs, k) for k in range(wins, pairs + 1)) / 2**pairs,
            no_case_regression=min(case_ratios.values()) >= 1 / 1.03,
        )
    return dict(metrics=metrics, raw=result)


def public(variant):
    return {k: v for k, v in variant.items() if k not in ("function", "library")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=ROOT / ".lossless/external/pybaselines-v1.2.1"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stamp(f"START estimated=3-5min results={out}/report.json log={out}.log")
    started = time.perf_counter()
    discovery_seed = 170711241 + (1000000 if args.smoke else 0)
    evaluation_seed = 170711993 + (1000000 if args.smoke else 0)
    protocol = dict(
        schema_version=1,
        discovery_seed=discovery_seed,
        evaluation_seed=evaluation_seed,
        discovery_specs=specifications(),
        evaluation_specs=specifications(True),
        discovery_pairs=2 if args.smoke else 8,
        evaluation_pairs=4 if args.smoke else 20,
        selection="Choose one exact discovery finalist; freeze before evaluation. Never replace after evaluation.",
        gate="At least 1.02x geometric mean warm and complete calls, no case slower than 1/1.03, final sign p<=0.01, exact output tree and ownership.",
        timing="Sequential same-process counterbalanced pairs, fresh factory and 5 changing-input requests; first and warm calls separate. Includes spectrum application validation/caches/results. Excludes imports, fixture generation, worker IPC and compilation; compilation separately recorded.",
        scope="Local arm64 Clang/NumPy OpenBLAS ModPoly pilot, finite application inputs, no universal equivalence proof or production promotion.",
        compiler=subprocess.check_output(["clang", "--version"], text=True),
        platform=platform.platform(),
        machine=platform.machine(),
        python=sys.version,
        numpy=np.__version__,
        runner_sha256=sha(Path(__file__)),
        resource_hashes={
            name: sha(RESOURCE / name)
            for name in ("modpoly.c", "modpoly.json", "modpoly_native.py", "spectrum_workload.py")
        },
    )
    write(out / "protocol.json", protocol)
    factories, binding = setup_application(args.project.resolve())
    from threadpoolctl import threadpool_info

    protocol["threadpools"] = threadpool_info()
    write(out / "protocol.json", protocol)
    baseline = build(out, "baseline", "baseline")
    identical = build(out, "identical_control", "baseline")
    source = dict(name="retained_python", mode="python")
    specs = specifications()[:2] if args.smoke else specifications()
    pairs = protocol["discovery_pairs"]
    source_check = qualify(baseline, source, specs, factories, binding, discovery_seed)
    if not source_check["exact"]:
        raise AssertionError("retained native and retained Python application outputs differ")
    write(out / "baseline_correctness.json", source_check)
    control = benchmark(baseline, identical, specs, factories, binding, discovery_seed, pairs)
    write(out / "control.json", control)
    ledger, variants = [], []
    experiments = [
        ("01_cpu_target", "baseline", ["-mcpu=native"]),
        ("02_alias_contract", "alias", []),
        ("03_loop_specialization", "specialize", []),
        ("04_dead_store", "elide", []),
        ("05_explicit_simd", "simd", []),
    ]
    for number in range(6):
        if number < 5:
            name, kind, flags = experiments[number]
            parent = None
        else:
            exact = [i for i, row in enumerate(ledger) if i > 0 and row["correctness"]["exact"]]
            best = max(
                exact, key=lambda i: ledger[i]["timing"]["metrics"]["complete_ns"]["speedup"]
            )
            kind = variants[best]["kind"]
            # The final combination is selected only using earlier discovery data.
            # If CPU targeting regressed, don't force it into the combined build.
            flags = (
                ["-mcpu=native"]
                if ledger[0]["timing"]["metrics"]["complete_ns"]["speedup"] > 1
                else []
            )
            name, parent = "06_measured_combination", variants[best]["name"]
        stamp(f"EXPERIMENT {number + 1}/6 {name} parent={parent}")
        variant = build(out, name, kind, flags)
        variant["parent"] = parent
        check = qualify(baseline, variant, specs, factories, binding, discovery_seed)
        timing = benchmark(baseline, variant, specs, factories, binding, discovery_seed, pairs)
        row = dict(variant=public(variant), correctness=check, timing=timing)
        ledger.append(row)
        variants.append(variant)
        write(out / name / "result.json", row)
        write(out / "discovery.json", ledger)
        stamp(
            f"RESULT {name} exact={check['exact']} warm={timing['metrics']['warm_ns']['speedup']:.4f}x complete={timing['metrics']['complete_ns']['speedup']:.4f}x"
        )
    eligible = [
        i
        for i, row in enumerate(ledger)
        if row["correctness"]["exact"]
        and all(
            row["timing"]["metrics"][m]["speedup"] >= 1.02
            and row["timing"]["metrics"][m]["no_case_regression"]
            for m in ("warm_ns", "complete_ns")
        )
    ]
    exact = [i for i, row in enumerate(ledger) if row["correctness"]["exact"]]
    chosen = max(
        eligible or exact, key=lambda i: ledger[i]["timing"]["metrics"]["complete_ns"]["speedup"]
    )
    winner = variants[chosen]
    write(
        out / "selection.json",
        dict(
            variant=public(winner),
            discovery_eligible=chosen in eligible,
            frozen_before_evaluation=True,
        ),
    )
    final_specs = specifications(True)[:2] if args.smoke else specifications(True)
    edges = [
        dict(id=name, size=263 + i * 2, batch=2, **options)
        for i, (name, options) in enumerate(
            [
                ("signed_zero", dict(zeros=True)),
                ("tiny", dict(scale=1e-150)),
                ("large", dict(scale=1e140)),
                ("order_zero", dict(order=0)),
                ("one_iteration", dict(maximum=1, tol=1e-15)),
                ("tight_stop", dict(maximum=300, tol=1e-12)),
                ("sparse_weights", dict(weighted=True, zero_weights=True)),
                ("readonly", dict(readonly=True)),
                ("aliased_clipping", dict(weighted=True, original=False)),
                ("strided_original", dict(strided=True, original=True)),
            ]
        )
    ]
    check = qualify(baseline, winner, final_specs + edges, factories, binding, evaluation_seed)
    write(out / "evaluation_correctness.json", check)
    source_final_check = qualify(
        source, winner, final_specs + edges, factories, binding, evaluation_seed
    )
    write(out / "evaluation_vs_python_correctness.json", source_final_check)
    final = benchmark(
        baseline,
        winner,
        final_specs,
        factories,
        binding,
        evaluation_seed,
        protocol["evaluation_pairs"],
    )
    # Independently compare the frozen choice to the retained source-only recipe.
    versus_python = benchmark(
        source,
        winner,
        final_specs,
        factories,
        binding,
        evaluation_seed,
        protocol["evaluation_pairs"],
    )
    memory = {}
    for variant in (baseline, winner):
        calls = [records(final_specs[-1], 771930 + j) for j in range(5)]
        tracemalloc.start()
        run_calls(variant, calls, factories, binding)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory[variant["name"]] = dict(traced_current_bytes=current, traced_peak_bytes=peak)
    control_ok = all(
        abs(control["metrics"][m]["speedup"] - 1) <= 0.02 for m in ("warm_ns", "complete_ns")
    )
    passed = (
        check["exact"]
        and source_final_check["exact"]
        and control_ok
        and all(
            final["metrics"][m]["speedup"] >= 1.02
            and final["metrics"][m]["no_case_regression"]
            and final["metrics"][m]["sign_p"] <= 0.01
            for m in ("warm_ns", "complete_ns")
        )
    )
    report = dict(
        status="completed",
        selected=public(winner),
        exact=check["exact"],
        numerical=check["numerical"],
        performance_gate_passed=passed,
        production_promoted=False,
        baseline=public(baseline),
        evaluation=final,
        versus_retained_python=versus_python,
        memory=memory,
        identical_control_passed=control_ok,
        exact_vs_retained_python=source_final_check["exact"],
        elapsed_seconds=time.perf_counter() - started,
        smoke=args.smoke,
        limitation="Finite exactness tests, one machine/environment, in-process application calls; no worker IPC or deployment requalification. Traced memory is not a native RSS bound.",
    )
    write(out / "report.json", report)
    stamp(
        f"COMPLETE exact={check['exact']} accepted_pilot={passed} warm={final['metrics']['warm_ns']['speedup']:.4f}x complete={final['metrics']['complete_ns']['speedup']:.4f}x results={out}/report.json log={out}.log"
    )


if __name__ == "__main__":
    main()
