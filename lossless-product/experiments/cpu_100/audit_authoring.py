"""Check the saved matched-author experiment boundary without new model calls."""

import argparse
from pathlib import Path

from common import DEFAULT_RUN, read, sha, stamp, tree, write


def audit(run):
    rows = [r for r in read(run / "ledger.json") if r["family"] == 5]
    if {r["number"] for r in rows} != set(range(1, 21)) or len(rows) != 20:
        raise ValueError("all twenty completed author pairs are required")
    observations = []
    for row in rows:
        folder = run / f"f5_{row['number']:02d}"
        if read(folder / "report.json") != row:
            raise ValueError(f"ledger/report mismatch in pair {row['number']}")
        plan = read(folder / "plan.json")
        control = read(folder / "control-request.json")
        diagnostics = read(folder / "diagnostics-request.json")
        extras = {
            key: diagnostics.pop(key)
            for key in ("measured_host_hotspots", "compiler_vectorization_remarks")
        }
        if control != diagnostics or not all(
            key in extras for key in ("measured_host_hotspots", "compiler_vectorization_remarks")
        ):
            raise ValueError(f"unequal common context in pair {row['number']}")
        receipts = []
        compile_settings = []
        for arm, result in row["arms"].items():
            author = folder / (arm + "-author")
            invocation = read(author / "invocation.json")
            receipt = read(author / "receipt.json")
            if (
                invocation["timeout_seconds"] != plan["timeout_per_arm"]
                or invocation["model"] != "gpt-6-astra"
                or invocation["effort"] != "medium"
            ):
                raise ValueError(f"unequal author settings in pair {row['number']}")
            if receipt["tool_items"]:
                raise ValueError(f"forbidden author tool use in pair {row['number']}")
            if sha(author / "prompt.json") != invocation["prompt_sha256"]:
                raise ValueError("author prompt identity changed")
            prompt = read(author / "prompt.json")
            if prompt.pop("transport_timeout_upper_bound_seconds") != plan["timeout_per_arm"]:
                raise ValueError("transport timeout differs from planned allowance")
            if prompt != read(folder / (arm + "-request.json")):
                raise ValueError("sent prompt differs from saved request")
            receipts.append((author / "receipt.json").stat().st_mtime_ns)
            if receipt["eligible"]:
                if sha(author / "response.json") != receipt["response_sha256"]:
                    raise ValueError("author response identity changed")
            build = result.get("build")
            if build and build["returncode"] == 0:
                project = folder / arm / "project"
                binary = Path(build["command"][build["command"].index("-o") + 1])
                if (
                    sha(project / "cpu_region.c") != build["source_sha256"]
                    or sha(binary) != build["binary_sha256"]
                ):
                    raise ValueError("compiled source or binary identity changed")
                if (project / "cpu_region.c").read_text() != result["proposal"]["source"]:
                    raise ValueError("compiled source differs from author response")
                snapshot = folder / arm / "source.json"
                if snapshot.exists() and tree(project) != read(snapshot):
                    raise ValueError("candidate project changed after measurement")
                compile_settings.append(
                    [token.replace(str(project), "<project>") for token in build["command"]]
                )
        if len(compile_settings) == 2 and compile_settings[0] != compile_settings[1]:
            raise ValueError(f"unequal compiler settings in pair {row['number']}")
        measured = [
            p.stat().st_mtime_ns
            for arm in ("control", "diagnostics")
            for p in (folder / arm / "correctness").glob("**/result.json")
        ]
        if measured and min(measured) < max(receipts):
            raise ValueError(f"pair {row['number']} measured before both author responses")
        observations.append(
            dict(
                number=row["number"],
                common_context_equal=True,
                settings_equal=True,
                identities_intact=True,
                both_responses_before_measurement=bool(measured),
            )
        )
    return dict(
        pairs=20,
        passed=True,
        observations=observations,
        scope="Saved protocol and artifact boundary audit, not performance or equivalence proof",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    write(args.output / "author-audit.json", result)
    stamp(f"AUTHOR AUDIT passed pairs=20 result={args.output}/author-audit.json")


if __name__ == "__main__":
    main()
