"""Three-arm manual-Codex experiment with frozen evaluation and equal candidate caps."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

from lossless._native import engine, operators
from lossless._native.common import stamp, write, read
from lossless import recipes


def specification(repeat, cache):
    # Evaluation shapes are committed before proposals and never used for refinement.
    return {
        "operator": "softmax",
        "comparator": "numpy_buffered",
        "cases": [
            {
                "id": f"{split}_{layout}",
                "split": split,
                "rows": rows + repeat * 2,
                "columns": cols + repeat * 2,
                "layout": layout,
            }
            for split, rows, cols in [("discovery", 64, 257), ("evaluation", 73, 283)]
            for layout in ["c", "f"]
        ],
        "budget_seconds": 90,
        "job_timeout_seconds": 8,
        "compile_timeout_seconds": 8,
        "target_ns": 800000,
        "screening_blocks": 5,
        "confirmation_blocks": 13,
        "max_proposals": 3,
        "min_speedup": 1.02,
        "cache": {"directory": str(cache), "reuse_validation": True},
    }


def pool(arm, proposals):
    base = operators.baseline_source("softmax")
    retained = {p["id"]: p["source"] for p in recipes.native_candidates("softmax")}
    if arm == "retained":
        return {"baseline_control": base, **retained}
    if arm == "enumerated":
        return {
            f"unroll_{n}": base.replace("sums[8]", f"sums[{n}]")
            .replace("j+8<=cols;j+=8", f"j+{n}<=cols;j+={n}")
            .replace("k<8", f"k<{n}")
            for n in [1, 4, 32]
        }
    return {p.stem: p.read_text() for p in sorted(proposals.glob("*.c"))}


def run(out, phase):
    plan = read(out / "plan.json")
    for repeat in range(3):
        for arm in ["retained", "enumerated", "manual_codex"]:
            run_dir = out / f"{arm}_{repeat}"
            if phase == "controls" and arm == "manual_codex":
                continue
            if phase == "manual" and arm != "manual_codex":
                continue
            if phase in ["controls", "manual"]:
                spec_path = out / f"spec_{repeat}.json"
                engine.initialize(spec_path, run_dir)
                sources = pool(arm, out / "manual_proposals")
                assert len(sources) == 3
                for name, source in sources.items():
                    folder = out / "submissions" / arm / name
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "kernel.c").write_text(source)
                    write(
                        folder / "proposal.json",
                        {
                            "schema_version": 1,
                            "operator": "softmax",
                            "id": name,
                            "source_file": "kernel.c",
                            "hypothesis": f"{arm} frozen candidate; generated before evaluation",
                        },
                    )
                    engine.submit(run_dir, folder / "proposal.json")
                engine.execute(run_dir, "discovery")
                engine.seal(run_dir)
            elif phase == "evaluate":
                engine.execute(run_dir, "evaluation")
                engine.report(run_dir)
            stamp(f"SEARCH phase={phase} arm={arm} replicate={repeat + 1}/3 result={run_dir}")
    if phase == "evaluate":
        results = []
        for path in sorted(out.glob("*_[012]")):
            summary = read(path / "summary.json")
            state = engine.verify(path)
            selected = summary["frozen_global_choice"]
            selected_summary = summary["frozen_choice_evaluation"]
            records = [
                read(path / v["result"])
                for v in state["jobs"].values()
                if v["kind"] == "benchmark"
                and v["stage"] == "evaluation"
                and v["candidate"] == selected
                and v["status"] == "ok"
            ]
            accepted = (
                bool(records)
                and selected_summary["complete"]
                and selected_summary["geomean_vs_comparator"] >= 1.02
                and all(r["ci95_vs_comparator"][0] > 1 for r in records)
            )
            results.append(
                {
                    "run": path.name,
                    "selected": selected,
                    "accepted": accepted,
                    "evaluation": selected_summary,
                    "worker_seconds": state["spent_seconds"],
                    "invalid_proposals": len(summary["rejected"]),
                }
            )
        write(
            out / "summary.json",
            {
                "plan": plan,
                "results": results,
                "provider_cost_usd": None,
                "scope": "Manual Codex proposals, one shared discovery-guided authoring session; three workload replications, not three independent model draws. No hosted provider billing or causal model superiority claim.",
            },
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=["prepare", "controls", "manual", "evaluate"], required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    started = time.monotonic()
    stamp(
        f"START phase={a.phase} estimated=1-3min result={out} log={out.parent / (out.name + '_' + a.phase + '.log')}"
    )
    if a.phase == "prepare":
        out.mkdir(parents=True, exist_ok=False)
        (out / "manual_proposals").mkdir()
        import shutil

        for proposal in (Path(__file__).parent / "manual_proposals").glob("*.c"):
            shutil.copy2(proposal, out / "manual_proposals" / proposal.name)
        for repeat in range(3):
            write(out / f"spec_{repeat}.json", specification(repeat, out / "cache"))
        write(
            out / "plan.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "arms": ["retained", "enumerated", "manual_codex"],
                "candidate_cap": 3,
                "wall_budget_per_arm_seconds": 90,
                "replications": 3,
                "promotion": "frozen winner; evaluation geomean >=1.02 and every case lower paired bootstrap bound >1",
                "spec_hashes": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in out.glob("spec_*.json")
                },
                "authoring_time": "recorded separately; not included in controller budget",
                "cost": "unavailable for manual assistant session",
            },
        )
    else:
        run(out, a.phase)
    stamp(
        f"COMPLETE phase={a.phase} result={out} elapsed={time.monotonic() - started:.2f}s log={out.parent / (out.name + '_' + a.phase + '.log')}"
    )


if __name__ == "__main__":
    main()
