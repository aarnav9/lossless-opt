"""Five direct LLVM IR experiments, stacked on the frozen C-level finalist.

Research only: opt verifies IR well-formedness; application replay checks sampled
bitwise behavior. Neither is a formal equivalence proof. No product files change.
"""

import argparse
import ctypes
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time

import pilot


OPT = Path("/opt/homebrew/opt/llvm@21/bin/opt")
WINNER_SHA = "b306765fee1e47d1170609b3bd535b40a727e385cd42d3f43e8316a686177bc8"
VECTORIZATION = "function(loop-vectorize,slp-vectorizer,instcombine,simplifycfg)"
SSA = r"%[-a-zA-Z$._0-9]+"


def invoke(command, logs):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=40)
    logs.append(
        dict(
            command=list(map(str, command)),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    )
    if result.returncode:
        raise RuntimeError(f"command failed: {command}\n{result.stderr}")


def named_ir(source, destination, logs):
    invoke([OPT, "-S", "-verify-each", "-passes=instnamer", source, "-o", destination], logs)


def blocks(ir):
    """Locate simple blocks in the single named entry function; preserve all text."""
    end = ir.index("\n}\n")
    matches = list(re.finditer(r"(?m)^([-a-zA-Z$._0-9]+):[^\n]*\n", ir[:end]))
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else end
        yield match.group(1), match.start(), stop, ir[match.start() : stop]


def pointwise(block, label, vector_only=False):
    return (
        f"label %{label}" in block
        and "!llvm.loop" in block
        and " phi " in block
        and "load " in block
        and ("fcmp olt" in block or "fsub " in block)
        and not re.search(r"call .*%(?:arg|i)[^(]*\(", block)
        and (not vector_only or re.search(r"load <\d+ x double>", block))
    )


def loop_metadata(ir, *, width=None, interleave=None, unroll=None):
    """Rewrite actual LLVM loop metadata, never the C source or FP instructions."""
    definitions = dict(re.findall(r"(?m)^!(\d+) = (.*)$", ir))
    next_id = max(map(int, definitions)) + 1
    additions, edits, changed = [], [], []
    for label, start, stop, block in blocks(ir):
        if not pointwise(block, label, vector_only=unroll is not None):
            continue
        old = re.search(r"!llvm.loop !(\d+)", block).group(1)
        references = re.findall(r"!(\d+)", definitions[old])[1:]
        remove = (
            ("llvm.loop.unroll.",)
            if unroll is not None
            else ("llvm.loop.vectorize.", "llvm.loop.interleave.")
        )
        kept = [n for n in references if not any(key in definitions[n] for key in remove)]
        attributes = (
            [("llvm.loop.unroll.count", "i32", unroll)]
            if unroll is not None
            else [
                ("llvm.loop.vectorize.enable", "i1", "true"),
                ("llvm.loop.vectorize.width", "i32", width),
                ("llvm.loop.interleave.count", "i32", interleave),
            ]
        )
        node = next_id
        next_id += 1
        for key, typ, value in attributes:
            additions.append(f'!{next_id} = !{{!"{key}", {typ} {value}}}')
            kept.append(str(next_id))
            next_id += 1
        additions.append(
            f"!{node} = distinct !{{!{node}, " + ", ".join(f"!{n}" for n in kept) + "}"
        )
        edits.append((start, stop, re.sub(r"!llvm.loop !\d+", f"!llvm.loop !{node}", block)))
        changed.append(dict(block=label, previous_loop_id=old, new_loop_id=node))
    if not changed:
        raise ValueError("no intended pointwise loops found")
    for start, stop, replacement in reversed(edits):
        ir = ir[:start] + replacement + ir[stop:]
    return ir + "\n" + "\n".join(additions) + "\n", dict(
        loops=changed, width=width, interleave=interleave, unroll=unroll
    )


def schedule_sqrt(ir):
    # llvm.sqrt is pure/speculatable in the seed's default FP environment. Keep
    # both BLAS calls in their original order and move only the first sqrt.
    pattern = re.compile(
        rf"(?m)^(  {SSA} = (?:tail )?call double @llvm\.sqrt\.f64\(double {SSA}\)[^\n]*\n)"
        rf"(  {SSA} = (?:tail )?call double {SSA}\([^\n]+\n)"
    )
    result, count = pattern.subn(lambda m: m[2] + m[1], ir)
    if count != 4:
        raise ValueError(f"expected four independently scheduled sqrt sites, found {count}")
    return result, dict(moved_sqrt_instructions=count, blas_call_order_unchanged=True)


def prefetch(ir, distance=256):
    # Non-inbounds GEP: the prefetch hint may refer beyond the last element and
    # does not promise dereferenceability. LLVM prefetch is explicitly nontrapping.
    gep = dict(re.findall(rf"(?m)^  ({SSA}) = getelementptr[^\n]*?, ptr ({SSA}),", ir))

    def root(pointer):
        seen = set()
        while pointer in gep and pointer not in seen:
            seen.add(pointer)
            pointer = gep[pointer]
        return pointer

    edits, sites, counter = [], [], 0
    for label, start, stop, block in blocks(ir):
        if not pointwise(block, label, vector_only=True):
            continue
        seen, lines = set(), []
        for line in block.splitlines(keepends=True):
            load = re.search(rf"load <\d+ x double>, ptr ({SSA}),", line)
            if load and root(load[1]) not in seen:
                seen.add(root(load[1]))
                name = f"%pilot.prefetch.{counter}"
                lines.append(f"  {name} = getelementptr i8, ptr {load[1]}, i64 {distance}\n")
                lines.append(f"  call void @llvm.prefetch.p0(ptr {name}, i32 0, i32 3, i32 1)\n")
                sites.append(dict(block=label, pointer=load[1], distance_bytes=distance))
                counter += 1
            lines.append(line)
        edits.append((start, stop, "".join(lines)))
    if not sites:
        raise ValueError("no vector-loop prefetch sites found")
    for start, stop, replacement in reversed(edits):
        ir = ir[:start] + replacement + ir[stop:]
    ir = ir.replace(
        "\nattributes #0",
        "\ndeclare void @llvm.prefetch.p0(ptr, i32 immarg, i32 immarg, i32 immarg)\n\nattributes #0",
        1,
    )
    return ir, dict(prefetch_sites=sites)


def load(folder, metadata):
    binary = folder / "helper.dylib"
    if "binary_sha256" in metadata and pilot.sha(binary) != metadata["binary_sha256"]:
        raise ValueError("frozen binary changed")
    if "ir_sha256" in metadata and pilot.sha(folder / "optimized.ll") != metadata["ir_sha256"]:
        raise ValueError("frozen IR changed")
    if "source_sha256" in metadata and pilot.sha(folder / "helper.c") != metadata["source_sha256"]:
        raise ValueError("frozen C source changed")
    library = ctypes.CDLL(str(binary))
    function = library.iterate
    function.argtypes = (
        [ctypes.c_void_p] * 12
        + [ctypes.c_longlong] * 4
        + [ctypes.c_int] * 4
        + [ctypes.c_double] * 2
    )
    function.restype = ctypes.c_int
    return {**metadata, "function": function, "library": library}


def assembly_inventory(path):
    source = path.read_text()
    instructions = [
        re.sub(r"\s+", " ", line.split(";")[0].strip())
        for line in source.splitlines()
        if re.match(r"^\s+[a-z][a-z0-9.]*\s", line)
    ]
    return dict(
        instruction_lines=len(instructions),
        instruction_text_sha256=hashlib.sha256("\n".join(instructions).encode()).hexdigest(),
        prefetch_instructions=sum(line.startswith("prfm ") for line in instructions),
        scalar_sqrt_instructions=sum(line.startswith("fsqrt ") for line in instructions),
    )


def build_ir(out, name, ir, pipeline="verify", details=None, parent="incumbent"):
    folder = out / name
    folder.mkdir()
    (folder / "input.ll").write_text(ir)
    logs = []
    begin = time.perf_counter()
    try:
        invoke(
            [
                OPT,
                "-S",
                "-verify-each",
                f"-passes={pipeline},function(instnamer)",
                "-pass-remarks=loop-vectorize|loop-unroll",
                folder / "input.ll",
                "-o",
                folder / "optimized.ll",
            ],
            logs,
        )
        base = ["clang", "-O3", "-fPIC", "-Xclang", "-disable-llvm-passes"]
        invoke([*base, "-dynamiclib", folder / "optimized.ll", "-o", folder / "helper.dylib"], logs)
        invoke([*base, "-S", folder / "optimized.ll", "-o", folder / "helper.s"], logs)
    finally:
        pilot.write(folder / "build.json", logs)
    metadata = dict(
        name=name,
        parent=parent,
        pipeline=pipeline,
        transform=details or {},
        build_seconds=time.perf_counter() - begin,
        ir_sha256=pilot.sha(folder / "optimized.ll"),
        input_ir_sha256=pilot.sha(folder / "input.ll"),
        binary_sha256=pilot.sha(folder / "helper.dylib"),
        binary_bytes=(folder / "helper.dylib").stat().st_size,
        assembly=assembly_inventory(folder / "helper.s"),
    )
    pilot.write(folder / "metadata.json", metadata)
    return load(folder, metadata)


def score(timing):
    return timing["metrics"]["complete_ns"]["speedup"]


def eligible(timing, final=False):
    return all(
        timing["metrics"][metric]["speedup"] >= 1.02
        and timing["metrics"][metric]["no_case_regression"]
        and (not final or timing["metrics"][metric]["sign_p"] <= 0.01)
        for metric in ("warm_ns", "complete_ns")
    )


def specs(final=False, confirmation=False):
    result = pilot.specifications(final)
    for row in result:
        row["size"] += 182 if confirmation else 128 if final else 92
    return result


def edges():
    result = [
        dict(id=name, size=347 + i * 2, batch=2, **options)
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
                ("aliased_clipping", dict(weighted=True)),
                ("strided_original", dict(strided=True, original=True)),
            ]
        )
    ]
    result += [
        dict(id=f"tail_{n}", size=n, batch=2, order=1, maximum=9, weighted=n % 2 == 1)
        for n in (2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33)
    ]
    return result


def evaluate(out, references, winner, factories, binding, *, confirmation=False, smoke=False):
    seed = (310718777 if confirmation else 310718993) + (1000000 if smoke else 0)
    cases = specs(True, confirmation)
    if smoke:
        cases = cases[:2]
    check = pilot.qualify(
        references["incumbent"], winner, cases + edges(), factories, binding, seed
    )
    pilot.write(out / "correctness.json", check)
    python = dict(name="retained_python", mode="python")
    python_check = pilot.qualify(python, winner, cases, factories, binding, seed)
    pilot.write(out / "python_correctness.json", python_check)
    pairs = 4 if smoke else 24 if confirmation else 20
    comparisons = {}
    for label, reference in references.items():
        comparisons[label] = pilot.benchmark(
            reference, winner, cases, factories, binding, seed, pairs
        )
        pilot.write(out / f"vs_{label}.json", comparisons[label])
    comparisons["python"] = pilot.benchmark(python, winner, cases, factories, binding, seed, pairs)
    pilot.write(out / "vs_python.json", comparisons["python"])
    exact = check["exact"] and python_check["exact"]
    return dict(
        exact=exact,
        numerical=check["numerical"] and python_check["numerical"],
        comparisons=comparisons,
        native_gate=exact
        and all(eligible(comparisons[k], True) for k in ("incumbent", "ir_control", "original")),
        product_gate=exact and all(eligible(value, True) for value in comparisons.values()),
    )


def main():
    global OPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=pilot.ROOT / ".lossless/external/pybaselines-v1.2.1"
    )
    parser.add_argument("--opt", type=Path, default=OPT)
    parser.add_argument("--confirm", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this frozen pilot targets macOS arm64")
    OPT = args.opt.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    pilot.stamp(f"START LLVM IR pilot estimated=under5min results={out}/report.json log={out}.log")
    protocol = dict(
        runner_sha256=pilot.sha(Path(__file__)),
        evaluator_sha256=pilot.sha(Path(pilot.__file__)),
        compiler=subprocess.check_output(["clang", "--version"], text=True),
        opt=subprocess.check_output([str(OPT), "--version"], text=True),
        discovery_seed=310718241,
        evaluation_seed=310718993,
        confirmation_seed=310718777,
        discovery_specs=specs(),
        evaluation_specs=specs(True),
        confirmation_specs=specs(True, True),
        discovery_pairs=8,
        evaluation_pairs=20,
        confirmation_pairs=24,
        smoke=args.smoke,
        floating_point="No fast-math flags, contraction or reduction changes. Original BLAS calls remain.",
        gate="Exact, >=1.02x warm and factory+5-call aggregates, <=3% case regressions, final sign p<=.01. Compare against previous C winner, unchanged IR roundtrip, original helper and Python recipe.",
        verification="LLVM -verify-each is well-formedness verification, NOT equivalence. Finite output/state/lifetime checks and reference repeatability are separate.",
        timing="Existing evaluator: in-process spectrum application; excludes interpreter/import startup, fixture generation, worker IPC and compilation.",
    )
    pilot.write(out / "protocol.json", protocol)
    factories, binding = pilot.setup_application(args.project.resolve())
    if args.confirm:
        run = args.confirm.resolve()
        prior = json.loads((run / "report.json").read_text())
        original_protocol = json.loads((run / "protocol.json").read_text())
        for key in ("runner_sha256", "evaluator_sha256"):
            if protocol[key] != original_protocol[key]:
                raise ValueError("frozen evaluator changed")
        references = {
            name: load(run / meta["name"], meta) for name, meta in prior["references"].items()
        }
        winner = load(run / prior["selected"]["name"], prior["selected"])
        final = evaluate(
            out, references, winner, factories, binding, confirmation=True, smoke=args.smoke
        )
        report = dict(
            status="completed",
            selected=pilot.public(winner),
            confirmation=True,
            frozen_from=str(run),
            production_promoted=False,
            **final,
        )
    else:
        incumbent = pilot.build(out, "incumbent", "elide")
        if incumbent["source_sha256"] != WINNER_SHA:
            raise ValueError("C-level finalist source changed")
        original = pilot.build(out, "original", "baseline")
        seeds = out / "seeds"
        seeds.mkdir()
        logs = []
        named_ir(out / "incumbent/helper.ll", seeds / "optimized.ll", logs)
        invoke(
            [
                "clang",
                "-O3",
                "-std=c11",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-fPIC",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-fno-unroll-loops",
                "-S",
                "-emit-llvm",
                out / "incumbent/helper.c",
                "-o",
                seeds / "scalar-unnamed.ll",
            ],
            logs,
        )
        named_ir(seeds / "scalar-unnamed.ll", seeds / "scalar.ll", logs)
        pilot.write(seeds / "build.json", logs)
        optimized, scalar = (seeds / "optimized.ll").read_text(), (seeds / "scalar.ll").read_text()
        ir_control = build_ir(out, "ir_control", optimized)
        vector_control = build_ir(out, "vector_control", scalar, VECTORIZATION)
        cases = specs()[:2] if args.smoke else specs()
        pairs, seed = (2 if args.smoke else 8), 310718241 + (1000000 if args.smoke else 0)
        control_checks = {
            v["name"]: pilot.qualify(incumbent, v, cases, factories, binding, seed)
            for v in (original, ir_control, vector_control)
        }
        pilot.write(out / "control_correctness.json", control_checks)
        if not all(c["exact"] for c in control_checks.values()):
            raise ValueError("a baseline/control is not bitwise equivalent")
        controls = {
            v["name"]: pilot.benchmark(incumbent, v, cases, factories, binding, seed, pairs)
            for v in (ir_control, vector_control)
        }
        pilot.write(out / "controls.json", controls)
        ledger, variants = [], []
        for number in range(1, 6):
            # Only earlier exact, regression-safe discovery winners can become
            # a parent. Otherwise start from the unchanged optimized IR.
            parents = [
                i
                for i, row in enumerate(ledger)
                if row["correctness"]["exact"] and eligible(row["timing"])
            ]
            best = max(parents, key=lambda i: score(ledger[i]["timing"])) if parents else None
            parent = variants[best]["name"] if best is not None else "ir_control"
            base = (out / parent / "optimized.ll").read_text()
            if number in (1, 2):
                width, interleave = (2, 1) if number == 1 else (4, 2)
                changed, details = loop_metadata(scalar, width=width, interleave=interleave)
                pipeline = VECTORIZATION
                parent = "vector_control"
                name = f"0{number}_vector_w{width}_i{interleave}"
                comparison_base = scalar
            elif number == 3:
                changed, details = schedule_sqrt(base)
                name, pipeline, comparison_base = "03_sqrt_schedule", "verify", base
            elif number == 4:
                changed, details = prefetch(base)
                name, pipeline, comparison_base = "04_prefetch", "verify", base
            else:
                changed, details = loop_metadata(base, unroll=2)
                name = "05_vector_loop_unroll"
                pipeline, comparison_base = (
                    "function(loop-unroll<O3>,instcombine,simplifycfg)",
                    base,
                )
            pilot.stamp(f"EXPERIMENT {number}/5 {name} parent={parent}")
            variant = build_ir(out, name, changed, pipeline, details, parent)
            (out / name / "edit.diff").write_text(
                "".join(
                    difflib.unified_diff(
                        comparison_base.splitlines(True),
                        changed.splitlines(True),
                        fromfile="parent.ll",
                        tofile="edited.ll",
                    )
                )
            )
            check = pilot.qualify(incumbent, variant, cases, factories, binding, seed)
            timing = pilot.benchmark(incumbent, variant, cases, factories, binding, seed, pairs)
            row = dict(variant=pilot.public(variant), correctness=check, timing=timing)
            ledger.append(row)
            variants.append(variant)
            pilot.write(out / name / "result.json", row)
            pilot.write(out / "discovery.json", ledger)
            pilot.stamp(
                f"RESULT {name} exact={check['exact']} warm={timing['metrics']['warm_ns']['speedup']:.4f}x complete={score(timing):.4f}x"
            )
        exact = [i for i, row in enumerate(ledger) if row["correctness"]["exact"]]
        passing = [i for i in exact if eligible(ledger[i]["timing"])]
        selected = max(passing or exact, key=lambda i: score(ledger[i]["timing"]))
        winner = variants[selected]
        pilot.write(
            out / "selection.json",
            dict(
                selected=pilot.public(winner),
                discovery_eligible=selected in passing,
                frozen_before_evaluation=True,
                diagnostic_finalist_if_none_eligible=not passing,
            ),
        )
        references = dict(incumbent=incumbent, ir_control=ir_control, original=original)
        final = evaluate(out, references, winner, factories, binding, smoke=args.smoke)
        report = dict(
            status="completed",
            selected=pilot.public(winner),
            references={k: pilot.public(v) for k, v in references.items()},
            production_promoted=False,
            discovery_eligible=selected in passing,
            confirmation=False,
            **final,
        )
    report["elapsed_seconds"] = time.perf_counter() - begin
    report["smoke"] = args.smoke
    pilot.write(out / "report.json", report)
    pilot.stamp(
        f"COMPLETE exact={report['exact']} native_gate={report['native_gate']} product_gate={report['product_gate']} versus_previous_winner={score(report['comparisons']['incumbent']):.4f}x results={out}/report.json log={out}.log"
    )


if __name__ == "__main__":
    main()
