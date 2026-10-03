"""Retain complete search evidence, sources and failures without CLI auth/event metadata."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(path):
        return {"original_sha256": digest(path), "record": json.loads(path.read_text())}

    def source(path):
        return {"sha256": digest(path), "source": path.read_text()}

    def diagnostic(path):
        data = record(path)
        original = data["record"]
        events = [json.loads(line) for line in original["stdout"].splitlines()]
        data["record"] = {
            "returncode": original["returncode"],
            "seconds": original["seconds"],
            "usage": [e["usage"] for e in events if e["type"] == "turn.completed"],
            "response": [
                e["item"]["text"]
                for e in events
                if e.get("item", {}).get("type") == "agent_message"
            ],
        }
        return data

    def authors(folder):
        result = []
        for index in range(3):
            author = folder / f"author_{index}"
            receipt = record(author / "receipt.json")
            assert receipt["record"]["blinding_passed"]
            response = author / "response.json"
            if response.exists():
                assert digest(response) == receipt["record"]["response_sha256"]
            invocation = record(author / "invocation.json")
            argv = invocation["record"]["argv"]
            argv[argv.index("--cd") + 1] = "<temporary-empty-directory>"
            result.append(
                dict(
                    receipt=receipt,
                    response=record(response) if response.exists() else None,
                    invocation=invocation,
                )
            )
        return result

    plan = record(args.study / "plan.json")
    harness = source(Path(__file__).with_name("study.py"))
    assert harness["sha256"] == plan["record"]["script_sha256"]
    frozen = {}
    for name, expected in plan["record"]["hashes"].items():
        path = args.study / name
        assert digest(path) == expected
        frozen[name] = record(path)
    runs = {}
    for index, arm in plan["record"]["order"]:
        name = f"{arm}_{index}"
        folder = args.study / name
        state = record(folder / "state.json")
        files = {}
        for filename in [
            "seal.json",
            "failed_search.json",
            "summary.json",
            "submission_failures.json",
            "discovery_wall.json",
            "evaluation_wall.json",
            "manifest.json",
            "hardware_profile.json",
            "contract.json",
        ]:
            path = folder / filename
            if path.exists():
                files[filename] = record(path)
        jobs = {}
        for job_id, job in state["record"]["jobs"].items():
            if "result" in job:
                path = folder / job["result"]
                assert digest(path) == job["result_sha256"]
                jobs[job_id] = record(path)
        runs[name] = dict(state=state, files=files, job_results=jobs)
    pilot = args.pilot
    pilot_plan = record(pilot / "plan.json")
    assert not any(pilot.glob("*/evaluation_wall.json")), "pilot holdout was not to be opened"
    pilot_runs = {}
    for index, arm in pilot_plan["record"]["order"]:
        folder = pilot / f"{arm}_{index}"
        pilot_runs[folder.name] = {
            name: record(folder / name)
            for name in [
                "seal.json",
                "failed_search.json",
                "discovery_wall.json",
                "submission_failures.json",
            ]
            if (folder / name).exists()
        }
    value = {
        "scope": "Three independent one-shot sessions, numerical softmax, one M2. All author responses, controls, seals and benchmark job records retained. Original file digests precede path redaction. Temporary CLI paths are replaced; authentication and unrelated session metadata are excluded. Billing cost is unavailable. Five-minute pilot is separate, with holdout unopened.",
        "plan": plan,
        "frozen_inputs": frozen,
        "harness": harness,
        "comparison_harness": source(Path(__file__).with_name("compare.py")),
        "comparison_bookkeeping_amendment": record(
            args.study / "comparison_bookkeeping_amendment.json"
        )
        if (args.study / "comparison_bookkeeping_amendment.json").exists()
        else None,
        "comparison_before_bookkeeping_fix": source(args.study / "compare_before_payback_fix.py")
        if (args.study / "compare_before_payback_fix.py").exists()
        else None,
        "authors": authors(args.study),
        "runs": runs,
        "summary": record(args.study / "summary.json"),
        "pilot": {
            "plan": pilot_plan,
            "original_plan": record(pilot / "plan_before_failure_handling.json"),
            "authoring_harness": source(pilot / "authoring_harness.py"),
            "authors": authors(pilot),
            "discovery_records": pilot_runs,
            "evaluation_opened": False,
            "diagnostics": {
                name: diagnostic(pilot.parent / name / "result.json")
                for name in ["search_043_diagnostic", "search_043_schema_diagnostic"]
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, allow_nan=False).replace(str(root), "<repository>") + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
