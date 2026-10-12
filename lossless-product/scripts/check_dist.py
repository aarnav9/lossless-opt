"""Inspect actual built distributions, not merely ignore rules."""

import json
from pathlib import Path
import sys
import tarfile
import zipfile


def main(directory):
    records = {}
    archives = sorted(Path(directory).glob("*.whl")) + sorted(Path(directory).glob("*.tar.gz"))
    if not any(p.suffix == ".whl" for p in archives) or not any(
        p.name.endswith(".tar.gz") for p in archives
    ):
        raise ValueError("both wheel and source archive are required")
    for archive in archives:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as stream:
                names = stream.namelist()
                file_bytes = sum(info.file_size for info in stream.infolist())
        else:
            with tarfile.open(archive) as stream:
                names = stream.getnames()
                file_bytes = sum(info.size for info in stream)
        for name in names:
            parts = Path(name).parts
            if any(
                p
                in {
                    "research",
                    ".venv",
                    ".lake",
                    ".lossless",
                    "lossless-runs",
                    ".env",
                    "__pycache__",
                }
                for p in parts
            ) or name.endswith((".safetensors", ".gguf", ".so", ".dylib", ".pyc")):
                raise ValueError(f"unexpected artifact content: {name}")
        for suffix in [
            "lossless/cli.py",
            "lossless/_native/deployment.py",
            "lossless/codex_provider.py",
            "lossless/applications/worker.py",
            "lossless/applications/replay.py",
            "lossless/applications/profile_worker.py",
            "lossless/applications/assessment.py",
            "lossless/applications/search.py",
            "lossless/applications/comparison.py",
            "lossless/applications/patches.py",
            "lossless/applications/native.py",
            "lossless/applications/architecture.py",
            "lossless/applications/cpu_catalog.py",
            "lossless/applications/retained.py",
            "lossless/applications/automatic.py",
            "lossless/applications/bundle.py",
            "lossless/applications/workflow.py",
            "lossless/execution.py",
            "lossless/_native/call_policy.py",
            "lossless/retained/catalog.json",
            "lossless/retained/PROVENANCE.json",
            "lossless/retained/modpoly.json",
            "lossless/retained/modpoly.c",
            "lossless/retained/torch_native.cpp",
            "lossless/retained/mps_native.mm",
            "lossless/retained/mps_kernels.metal",
            "lossless/retained/onnx_rms.c",
            "lossless/retained/PYBASELINES_LICENSE.txt",
            "lossless/retained/KERNELBENCH_LICENSE",
            "lossless/retained/pybaselines.py",
            "lossless/retained/mlx_full_request.py",
            "lossless/resources/rms_candidates_004.c",
            "lossless/resources/softmax_fortran_046.c",
            "lossless/resources/softmax_dispatch_033.c",
            "experiments/application_intake/PYBASELINES_LICENSE.txt",
            "lossless/profiling.py",
            "lossless/constraints.py",
            "lossless/_profile_worker.py",
            "lossless/resources/proofs/Bounds.lean",
            "lossless/resources/proofs/Fusion.lean",
            "lossless/resources/proofs/Movement.lean",
            "lossless/resources/theorems/index.jsonl.gz",
            "lossless/resources/theorems/MATHLIB_LICENSE",
            "lossless/adapters/mlx/MLX_LM_LICENSE",
            "/LICENSE",
            "NOTICE",
        ]:
            if not any(n.endswith(suffix) for n in names):
                raise ValueError(f"missing package resource: {suffix}")
        if archive.suffix != ".whl":
            for suffix in [
                "examples/softmax-fortran/recipe.py",
                "examples/softmax-fortran/kernel.c",
                "examples/application/project/app.py",
                "experiments/application_intake/frozen-candidates.json",
                "experiments/application_intake/qualify_study.py",
                "experiments/application_intake/broader_study.py",
                "experiments/application_intake/spectrum_workload.py",
                "experiments/application_intake/qualify_broader.py",
                "experiments/application_intake/continue_broader.py",
                "experiments/application_intake/broader-candidate.json",
                "experiments/application_intake/summarize_broader.py",
                "experiments/application_intake/summarize_continuation.py",
                "experiments/application_intake/qualify_continuation.py",
                "experiments/application_intake/continuation-candidate.json",
                "experiments/application_intake/checklist_study.py",
                "experiments/application_intake/compare_checklist.py",
                "experiments/application_intake/summarize_checklist.py",
                "experiments/application_intake/cpu-control-candidate.json",
                "experiments/application_intake/cpu-assisted-candidate.json",
                "experiments/cpu_100/study.py",
                "experiments/cpu_100/audit_authoring.py",
                "experiments/cpu_100/summarize.py",
                "docs/cpu-100-study.md",
                "docs/research-integration.md",
                "docs/evidence/research-integration.json.gz",
                "scripts/check_retained_application.py",
                "scripts/check_cpu_project.py",
                "docs/cpu-projects.md",
                "docs/cpu-project-workflow-study.md",
                "docs/evidence/cpu-project-workflow.json.gz",
                "scripts/summarize_cpu_projects.py",
                "examples/cpu-project/project/signals.py",
                "scripts/check_retained_cpu.py",
                "scripts/check_retained_onnx.py",
                "scripts/check_retained_mps.py",
                "scripts/check_retained_mlx_vision.py",
                "scripts/check_retained_mlx_request.py",
                "scripts/check_retained_install.py",
                "docs/cpu-100-experiments.md",
                "docs/evidence/cpu-100.json.gz",
                "docs/cpu-investigation.md",
                "docs/benchmarking.md",
                "docs/cpu-checklist-study.md",
                "docs/evidence/cpu-checklist-pybaselines.json.gz",
                "docs/application-broader-study.md",
                "docs/application-continuation-study.md",
                "docs/evidence/application-broader-pybaselines.json.gz",
                "docs/evidence/application-continuation-pybaselines.json.gz",
                "examples/application/cases.json",
                "experiments/kernelbench_049/upstream/LICENSE",
            ]:
                if not any(n.endswith(suffix) for n in names):
                    raise ValueError(f"missing source-distribution resource: {suffix}")
        records[archive.name] = {
            "archive_bytes": archive.stat().st_size,
            "file_bytes": file_bytes,
            "entries": len(names),
        }
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
