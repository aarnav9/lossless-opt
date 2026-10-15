"""Local CPU callable intake, measured candidate selection and runnable delivery."""

import copy
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile

from .._native.common import stamp, write
from ..jobs import identity, positive, read_json
from .automatic import plan_for
from .bundle import admitted_signatures
from .config import initialize, inspect, resolve
from .discovery import inventory
from .profiling import profile
from .replay import ReplayResult, copy_project
from .runtime import clock, execute
from .search import search, snapshot


def prepare_cases(path):
    rows = copy.deepcopy(read_json(path))
    if (
        not isinstance(rows, list)
        or len(rows) < 2
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(
            "provide at least two representative cases, including independent evaluation inputs"
        )
    specified = ["split" in row for row in rows]
    if any(specified) and not all(specified):
        raise ValueError("specify a split for every case or omit all splits")
    boundary = max(1, len(rows) * 2 // 3)
    for i, row in enumerate(rows):
        row.setdefault("id", f"case_{i:03d}")
        row.setdefault("split", "discovery" if i < boundary else "evaluation")
    if {r["split"] for r in rows} != {"discovery", "evaluation"}:
        raise ValueError("both discovery and evaluation cases are required")
    return rows


def bundle_environment(python, inherit_env):
    code = "import json;from lossless.applications.bundle import environment;print(json.dumps(environment()))"
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "SYSTEMROOT", *inherit_env}}
    with tempfile.TemporaryDirectory(prefix="lossless-environment-") as temporary:
        env.update(
            HOME=temporary,
            TMPDIR=temporary + "/",
            PYTHONHASHSEED="0",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        )
        result = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=20, env=env
        )
    if result.returncode:
        raise ValueError(
            "selected Python needs the current Lossless package and NumPy; install lossless-product[applications] in that environment"
        )
    return json.loads(result.stdout)


def make_bundle(
    output, reference, reference_source, selected, selected_source, value, cases, env, result
):
    destination = output / "application"
    destination.mkdir()
    copy_project(reference, destination / "reference", reference_source)
    if selected is not None:
        copy_project(selected, destination / "optimized", selected_source)
    write(destination / "cases.json", cases)
    manifest = dict(
        schema_version=1,
        kind="cpu_application",
        selected="optimized" if selected else "reference",
        entry=value["entry"],
        environment=env,
        seed=value["replay"]["seed"],
        reference_files=reference_source["files"],
        optimized_files=selected_source["files"] if selected else {},
        input_signatures=admitted_signatures(cases, reference),
        inherit_env=value["environment"]["inherit_env"],
        environment_values_hash=identity(
            {name: os.environ.get(name) for name in value["environment"]["inherit_env"]}
        ),
        max_output_bytes=value["replay"]["max_output_bytes"],
        qualification=result,
        scope="Finite exact CPU callable evidence. New finite arrays require measured shape/dtype/layout and fixed scalar controls. Original copy remains available. No automatic retry after execution begins.",
    )
    write(destination / "manifest.json", manifest)
    (destination / "run.py").write_text(
        '"""Run in a fresh process with the selected Python environment."""\n'
        "from pathlib import Path\nfrom lossless.applications.bundle import main\n"
        'if __name__ == "__main__":\n    main(Path(__file__).resolve().parent)\n'
    )
    (destination / "callable.py").write_text(
        '"""Use open_callable() as a context manager; results are copied values."""\n'
        "from pathlib import Path\nfrom lossless.applications import CallableApplication\n"
        "def open_callable(**options):\n"
        "    return CallableApplication(Path(__file__).resolve().parent, **options)\n"
    )
    (destination / "USAGE.txt").write_text(
        "Call with ordinary Python inputs and receive the return value:\n"
        "from lossless.applications import CallableApplication\n"
        "with CallableApplication('/path/to/application') as run:\n"
        "    result = run(new_array)  # use your callable's args and kwargs\n"
        "    print(run.info)  # actual variant, mode, fallback reason\n"
        "Qualify this interface with run-project --deployment auto and --expected-requests N.\n"
        "Without a current callable.json certificate, calls use a fresh reference worker.\n"
        "Inputs/results are copied values; input mutations and aliases do not cross the boundary.\n"
        "Use variant='reference' for unsupported metadata. No retries after execution begins.\n\n"
        "Run with the same Python environment used for qualification:\n"
        "python /path/to/application/run.py --case CASE_ID --output /path/to/new-result\n"
        "For new inputs: --request request.json instead of --case. JSON accepts args/kwargs or calls; $npy paths are relative to request.json.\n"
        "Add --reference to disable optimization. Original source was never changed.\n"
        "Read results: from lossless.applications import read_outputs; values = read_outputs('/path/to/new-result')\n"
        "Source-search timings exclude launcher/interpreter, admission hashing, fixture I/O and result serialization. Deployment qualification measures the complete callable adapter separately. File delivery checks record descriptive complete-launch costs.\n"
        "Each launch starts a fresh callable/factory; a calls sequence reuses it. Outputs save values, not Python object aliases.\n"
        "For the legacy saved-fixture interface, qualify with run-project --prepared, then use:\n"
        "from lossless.applications import PreparedApplication\n"
        "with PreparedApplication('/path/to/application') as app:\n"
        "    values = app.run_case('CASE_ID')['outputs']\n"
        "New requests: app.run({'args': [{'$npy': 'new.npy'}]}, input_root='/path/to/fixtures').\n"
        "Prepared workers reuse imports but reset the seeded factory per request. A separate prepared.json certificate enables optimized requests; unsupported metadata requires an explicit reference session.\n"
    )
    return destination


def check_delivery(bundle, value, cases, folder, deadline, progress):
    checks = []
    for case in cases:
        if case["split"] != "evaluation":
            continue
        row = dict(id=case["id"])
        for variant in ("reference", "auto"):
            run = folder / case["id"] / variant
            run.mkdir(parents=True)
            args = [
                value["environment"]["python"],
                str(bundle / "run.py"),
                "--case",
                case["id"],
                "--output",
                str(run / "outputs"),
            ]
            if variant == "reference":
                args.append("--reference")
            left = deadline - clock()
            if left <= 0:
                raise TimeoutError("workflow deadline exhausted before delivery validation")
            progress(f"DELIVERY case={case['id']} variant={variant}")
            receipt = execute(
                args,
                cwd=bundle,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                stdin="",
                folder=run,
                timeout=min(left, value["replay"]["timeout_seconds"]),
                max_output_bytes=value["replay"]["max_output_bytes"],
                progress=progress,
            )
            if receipt["status"] != "completed":
                raise ValueError(
                    f"deliverable failed for {case['id']}; inspect {run / 'stderr.log'}"
                )
            row[variant] = read_json(run / "outputs/result.json")
            row[variant]["launch_seconds"] = receipt["elapsed_seconds"]
        row["exact"] = row["reference"]["fingerprint"] == row["auto"]["fingerprint"]
        row["launch_speed_ratio"] = (
            row["reference"]["launch_seconds"] / row["auto"]["launch_seconds"]
        )
        checks.append(row)
        write(folder / "checks.json", checks)
        if not row["exact"]:
            raise ValueError("runnable deliverable failed exact comparison")
    return checks


def optimize_project(
    project,
    *,
    output,
    entry=None,
    cases=None,
    input_loader=None,
    python=None,
    factory=False,
    observe=None,
    cleanup=None,
    include=(),
    exclude=(),
    inherit_env=(),
    budget=180,
    plan=None,
    prepared=False,
    prepared_inputs="fixtures",
    deployment=None,
    expected_requests=100,
    recipe_coverage="stable",
):
    """One local workflow; no implicit provider calls or numerical acceptance."""
    started = clock()
    positive(budget, "workflow budget")
    if recipe_coverage not in {"stable", "expanded"}:
        raise ValueError("recipe_coverage must be stable or expanded")
    if deployment not in {None, "auto", "oneshot", "prepared"}:
        raise ValueError("deployment must be auto, oneshot or prepared")
    if type(expected_requests) is not int or not 1 <= expected_requests <= 1000000:
        raise ValueError("expected_requests must be an integer from 1 to 1000000")
    if deployment and prepared:
        raise ValueError("choose --deployment or the legacy --prepared qualification")
    if cases is not None and input_loader is not None:
        raise ValueError("choose --cases or --input-loader, not both")
    if input_loader is not None and not entry:
        raise ValueError("--input-loader requires an explicit --entry")
    if prepared_inputs not in {"fixtures", "values"} or (
        prepared_inputs == "values" and not prepared
    ):
        raise ValueError(
            "prepared_inputs requires fixtures or values; values also requires prepared=True"
        )
    root, out = Path(project).resolve(), Path(output).resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("workflow output must be separate from the source project")
    out.mkdir(parents=True, exist_ok=False)
    deadline = started + budget
    report = dict(
        kind="cpu_project_workflow",
        status="incomplete",
        selected="reference",
        llm_calls=0,
        contract="exact",
        recipe_coverage=recipe_coverage,
        budget_seconds=budget,
        source_project=str(root),
    )

    def progress(message):
        stamp(message)
        with (out / "run.log").open("a") as log:
            from .._native.common import now

            log.write(f"[{now()}] {message}\n")

    progress(
        f"START CPU project workflow budget={budget}s result={out / 'report.json'} log={out / 'run.log'} model_calls=0"
    )
    original_input = None
    work_root = root
    try:
        if input_loader is not None:
            from .input_loader import collect

            work_root, cases, original_input, receipt = collect(
                root,
                out,
                input_loader,
                python=python,
                include=include,
                exclude=exclude,
                inherit_env=inherit_env,
                timeout=min(60, max(0.01, deadline - clock())),
                progress=progress,
            )
            report["input_loader"] = dict(
                target=input_loader, receipt=receipt, frozen_cases=str(cases)
            )
        if cases is not None:
            write(out / "inputs.json", prepare_cases(Path(cases).resolve()))
        initialize(
            work_root,
            out / "job",
            entry=entry,
            factory=factory,
            cases=out / "inputs.json" if cases is not None else None,
            python=python,
            include=include,
            exclude=exclude,
        )
        config = out / "job/lossless.json"
        value = read_json(config)
        value["analysis"] = None
        value["environment"]["inherit_env"] = list(inherit_env)
        value["profile"]["architecture"] = False
        if value["entry"]:
            value["entry"].update(observe=observe, cleanup=cleanup)
        write(config, value)
        info = inspect(config)
        if info["status"] != "ready_to_replay":
            report.update(
                status="needs_input",
                missing=info["missing"],
                entry_suggestions=info["discovery"]["entry_candidates"][:10],
                config=str(config),
            )
            return ReplayResult(out, report)
        value, resolved_cases = resolve(config)
        if factory and not observe:
            raise ValueError("factory workflows need --observe to declare relevant object state")
        env = bundle_environment(value["environment"]["python"], inherit_env)
        original = info["source"]
        frozen = out / "reference"
        copy_project(work_root, frozen, original)
        value["project"]["root"] = str(frozen)
        write(config, value)
        available = deadline - clock()
        if available <= 0:
            raise TimeoutError("workflow budget exhausted during setup")
        profiled = profile(
            config,
            output=out / "profile",
            repeats=3,
            budget=available * 0.25,
            analysis=False,
            architecture=False,
        )
        if profiled.summary["status"] != "profiled":
            from .diagnostics import diagnose_reference

            remaining = deadline - clock()
            if remaining > 2:
                report["diagnostic"] = diagnose_reference(
                    config, output=out / "diagnostic", repeats=3, budget=min(30, remaining)
                ).summary
            raise ValueError(
                "reference profiling/repeatability failed; inspect profile/report.json"
            )
        available = deadline - clock()
        if available <= 0:
            raise TimeoutError("workflow budget exhausted after profiling")
        search_budget = available * (0.45 if deployment else 0.65 if prepared else 0.8)
        if plan is None:
            selected_plan, reason = plan_for(
                frozen, original, profiled.summary, search_budget, coverage=recipe_coverage
            )
        else:
            selected_plan = copy.deepcopy(
                read_json(plan) if isinstance(plan, (str, Path)) else plan
            )
            if selected_plan.get("provider") is not None or "candidates" not in selected_plan:
                raise ValueError(
                    "run-project accepts local saved plans only; provider calls require search-application"
                )
            selected_plan.update(
                wall_time_seconds=search_budget, final_reserve_seconds=search_budget * 0.45
            )
            reason = "explicit_local_plan"
        report["candidate_family"] = reason
        selected, selected_source = None, None
        if selected_plan is not None:
            write(out / "plan.json", selected_plan)
            outcome = search(config, plan=selected_plan, output=out / "search")
            report["search"] = outcome.summary
            if outcome.summary["status"] not in {"accepted_experiment", "reference_retained"}:
                raise ValueError("search incomplete or failed; inspect search/report.json")
            if outcome.summary["status"] == "accepted_experiment":
                selected = Path(outcome.summary["candidate_project"])
                selected_source = snapshot(selected, value)
        if deployment:
            from .deployment import select_deployment

            report["callable_search_selected"] = "optimized" if selected else "reference"
            selected, selected_source, report["deployment"] = select_deployment(
                output=out / "deployment",
                reference=frozen,
                reference_source=original,
                value=value,
                cases=resolved_cases,
                env=env,
                search_folder=out / "search" if selected_plan is not None else None,
                mode=deployment,
                expected_requests=expected_requests,
                deadline=deadline - min(15, max(0, (deadline - clock()) * 0.1)),
                progress=progress,
            )
        report.update(
            status="accepted" if selected else "reference_retained",
            selected="optimized" if selected else "reference",
        )
        bundle = make_bundle(
            out,
            frozen,
            original,
            selected,
            selected_source,
            value,
            resolved_cases,
            env,
            report["status"],
        )
        report["application"] = str(bundle)
        if deployment:
            from .deployment import certify

            certify(bundle, report["deployment"])
        report["delivery"] = check_delivery(
            bundle, value, resolved_cases, out / "delivery", deadline, progress
        )
        if selected and any(c["auto"]["variant"] != "optimized" for c in report["delivery"]):
            raise ValueError(
                "optimized deliverable failed admission in its qualification environment: "
                + "; ".join(
                    str(c["auto"]["fallback_reason"])
                    for c in report["delivery"]
                    if c["auto"]["variant"] != "optimized"
                )
            )
        if prepared:
            from .prepared_check import qualify_prepared
            from .bundle import read_outputs
            from .worker import Observer

            remaining = deadline - clock()
            if remaining <= 0:
                report["prepared"] = dict(
                    status="incomplete",
                    reason="workflow budget exhausted before prepared qualification",
                )
            else:
                expected_outputs = {}
                for row in report["delivery"]:
                    values = read_outputs(out / "delivery" / row["id"] / "reference/outputs")
                    expected_outputs[row["id"]] = identity(
                        Observer(value["replay"]["max_output_bytes"] * len(values)).encode(values)
                    )
                report["prepared"] = qualify_prepared(
                    bundle,
                    output=out / "prepared",
                    expected={
                        row["id"]: row["reference"]["fingerprint"] for row in report["delivery"]
                    },
                    expected_outputs=expected_outputs,
                    python=value["environment"]["python"],
                    budget=remaining,
                    input_mode=prepared_inputs,
                )
        report["source_unchanged"] = (
            inventory(root, exclude, include=include)["files"] == original_input["files"]
            if original_input is not None
            else inventory(
                root,
                value["project"]["exclude"],
                value["project"]["limits"],
                value["project"]["include"],
            )["files"]
            == original["files"]
        )
        if deployment and report["deployment"]["status"] == "failed":
            report.update(status="failed", reason=report["deployment"].get("reason"))
        if not report["source_unchanged"]:
            raise ValueError("input project changed during workflow; rerun from a stable source")
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        report.update(status="failed", selected="reference", reason=str(error))
        manifest_path = out / "application/manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest.update(selected="reference", qualification="delivery_failed")
            write(manifest_path, manifest)
    finally:
        report["elapsed_seconds"] = clock() - started
        report["scope"] = (
            "Exact finite callable checks. Fresh-launch delivery timings are single descriptive pairs including interpreter, admission, input/output I/O and serialization; they are not qualified launch speedups. No automatic in-place installation or universal equivalence proof."
        )
        if deployment:
            report["scope"] += (
                " Deployment comparisons separately qualify complete copied-value adapter calls with declared startup amortization."
            )
        write(out / "report.json", report)
        (out / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>CPU application</title><h1>CPU application workflow</h1><pre style='white-space:pre-wrap'>"
            + html.escape(json.dumps(report, indent=2))
            + "</pre>"
        )
        progress(
            f"DONE status={report['status']} result={out / 'report.json'} log={out / 'run.log'}"
        )
    return ReplayResult(out, report)
