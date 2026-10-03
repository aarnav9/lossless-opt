"""Bounded optimization and export of the retained, guarded MLX recipe."""

import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from ...jobs import fields, positive, read_json
from ..._native.common import write, sha, stamp


def resolve_workload(work, base):
    if "cases" in work:
        raise ValueError("MLX takes a request input file, not native tensor cases")
    fields(
        work.get("parameters"),
        {"model", "sampler", "fixed_count", "repeats", "graph_recipes"},
        "MLX parameters",
        {"model", "fixed_count"},
    )
    params = work["parameters"]
    graphs = params.setdefault("graph_recipes", [])
    if (
        not isinstance(graphs, list)
        or any(g not in {"decoder_026", "fullhead_027"} for g in graphs)
        or len(set(graphs)) != len(graphs)
    ):
        raise ValueError("graph_recipes must be unique supported recipe names")
    if params["fixed_count"] is not True or params.get("sampler", "greedy") != "greedy":
        raise ValueError("MLX optimization currently requires fixed_count=true and greedy sampling")
    params.setdefault("sampler", "greedy")
    params.setdefault("repeats", 5)
    if type(params["repeats"]) is not int or not 3 <= params["repeats"] <= 50:
        raise ValueError("MLX repeats must be between 3 and 50")
    model = (base / params["model"]).resolve()
    if not model.is_dir():
        raise ValueError(
            "workload.parameters.model must name an existing local model directory; no implicit download"
        )
    params["model"] = str(model)
    source = (base / work.get("inputs", "")).resolve()
    requests = read_json(source)
    if not isinstance(requests, list) or not requests:
        raise ValueError("MLX inputs must be a JSON array of requests")
    splits = {"discovery": [], "evaluation": []}
    for request in requests:
        fields(
            request, {"text", "max_tokens", "split"}, "MLX request", {"text", "max_tokens", "split"}
        )
        if (
            request["split"] not in splits
            or not isinstance(request["text"], str)
            or not request["text"].strip()
        ):
            raise ValueError("each MLX request needs nonempty text and discovery/evaluation split")
        if positive(request["max_tokens"], "max_tokens", True) > 256:
            raise ValueError("max_tokens exceeds alpha limit of 256")
        splits[request["split"]].append(request)
    if not all(splits.values()) or any(len(rows) > 64 for rows in splits.values()):
        raise ValueError("each MLX split needs 1–64 requests")
    if {r["text"] for r in splits["discovery"]} & {r["text"] for r in splits["evaluation"]}:
        raise ValueError("MLX discovery and evaluation texts must be distinct")
    work["inputs"] = requests


def optimize(workload, resolved, *, llm=None, output=None, resume=False):
    from ...api import Result, save_report
    from datetime import datetime, timezone

    if llm is not None:
        raise ValueError(
            "MLX worker needs a configured file/module callback or command; use job.llm for isolated GPU execution"
        )
    if sys.platform != "darwin" or __import__("platform").machine() != "arm64":
        raise ValueError("the MLX adapter currently requires Apple Silicon macOS")
    for name, version in {"mlx": "0.32.3", "mlx-lm": "0.32.0"}.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            raise ValueError(
                f"{name}=={version} required; install 'lossless-opt[mlx]' in a compatible environment"
            )
    run = (
        Path(output).resolve()
        if output
        else Path(resolved["output"]["directory"])
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    if resume:
        if not output:
            raise ValueError("resume requires the existing run directory via output")
        if read_json(run / "resolved.json") != resolved:
            raise ValueError("resume configuration differs from frozen MLX job")
        if (run / "report.json").exists():
            verify_run(run)
            return Result(run, read_json(run / "report.json"))
        raise ValueError(
            "interrupted MLX jobs retain evidence but cannot be resumed in this alpha; use a new bounded job"
        )
    run.mkdir(parents=True, exist_ok=False)
    write(run / "resolved.json", resolved)
    write(run / "controller.json", {"base": str(workload.base)})
    started = time.monotonic()
    stamp(
        f"START MLX worker result={run} log={run / 'run.log'} budget={resolved['budget']['wall_time_seconds']}s"
    )
    # Only the explicitly configured provider may receive its credential variables.
    names = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        *(resolved.get("llm") or {}).get("credential_env", []),
    }
    env = {k: v for k, v in os.environ.items() if k in names}
    env.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    # PYTHONPATH is used only for an editable source install; an installed wheel
    # is found through the interpreter's ordinary site-packages.
    package_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(package_root)
    with (run / "run.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "lossless.adapters.mlx.worker", str(run)],
            env=env,
            cwd=run,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=resolved["budget"]["wall_time_seconds"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            summary = {
                "schema_version": 1,
                "adapter": "mlx.fixed_count",
                "status": "incomplete",
                "selected": "reference",
                "wall_time_seconds": time.monotonic() - started,
                "reason": "worker budget exhausted or interrupted; no candidate admitted",
            }
            save_report(run, summary)
            return Result(run, summary)
    if process.returncode or not (run / "report.json").exists():
        raise RuntimeError(f"MLX worker failed; inspect {run / 'run.log'}")
    summary = read_json(run / "report.json")
    stamp(
        f"COMPLETE MLX outcome={summary['status']} result={run / 'report.json'} log={run / 'run.log'}"
    )
    return Result(run, summary)


def verify_run(run):
    receipt = read_json(run / "sealed.json")
    if not {
        "report.json",
        "resolved.json",
        "model_identity.json",
        "runtime_identity.json",
        "hardware.json",
    } <= set(receipt.get("hashes", {})):
        raise ValueError("incomplete MLX evidence receipt")
    for name, digest in receipt["hashes"].items():
        if (run / name).resolve().parent != run.resolve() or sha(run / name) != digest:
            raise ValueError(f"MLX run evidence changed: {name}")


def export(run, destination):
    report = read_json(run / "report.json")
    if report["status"] not in {"accepted", "reference_retained"}:
        raise ValueError("MLX job did not complete acceptance")
    verify_run(run)
    destination.mkdir(parents=True, exist_ok=False)
    for name in [
        "report.json",
        "resolved.json",
        "hardware.json",
        "model_identity.json",
        "runtime_identity.json",
        "sealed.json",
    ]:
        shutil.copy2(run / name, destination / name)
    manifest = {
        "schema_version": 1,
        "adapter": "mlx.fixed_count",
        "selected": report["selected"],
        "options": report.get("options", {}),
        "model_identity": read_json(run / "model_identity.json"),
        "runtime_identity": read_json(run / "runtime_identity.json"),
        "runtime": {"mlx": "0.32.3", "mlx-lm": "0.32.0"},
        "hashes": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
        "scope": "Exclusive process, fixed-count greedy requests. Weights unchanged. EOS/cancellation use stock serial.",
    }
    write(destination / "manifest.json", manifest)
    (destination / "USAGE.txt").write_text(
        "import lossless\noperation = lossless.load('PATH_TO_THIS_DIRECTORY')\n# Use a dedicated process with no other MLX work. Model weights are referenced, not bundled.\nrow, caches, probabilities = operation.generate(['Hello'], [8], capture=True)\n# capture returns required scores; caches expose final active state.\n"
    )
    return destination


def load(path, manifest):
    return MLXOperation(path, manifest)


class MLXOperation:
    """Explicit single-worker integration; use only in a dedicated MLX process."""

    def __init__(self, path, manifest):
        self.path, self.manifest = path, manifest
        self._model = None
        self.fallback_reason = None

    def open(self, model=None):
        for name, version in self.manifest["runtime"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(
                    f"artifact requires {name}=={version}; use a compatible environment"
                )
        import mlx.core as mx
        from mlx_lm import load
        from .identity import fingerprint
        from .runtime import admission, runtime_identity
        from .model import GroupedHead

        source = Path(
            model or read_json(self.path / "resolved.json")["workload"]["parameters"]["model"]
        )
        if not source.is_dir():
            raise ValueError(
                "local model directory missing; call operation.open(model=...) with the relocated weights"
            )
        model_identity = fingerprint(source)
        allowed, reason = admission(
            source, mx.device_info()["device_name"], model_identity=model_identity
        )
        if model_identity != self.manifest["model_identity"]:
            allowed = False
            reason = "model identity differs from deployment artifact"
        if runtime_identity() != self.manifest["runtime_identity"]:
            allowed = False
            reason = "Lossless runtime differs from validated artifact"
        self._model, self._tokenizer = load(str(source.resolve()))
        mx.eval(self._model.parameters())
        self._grouped = GroupedHead(self._model, 4) if allowed else None
        self._allowed = allowed and self.manifest["selected"] != "reference"
        self._graph = None
        recipe = self.manifest.get("options", {}).get("execution_recipe", "retained")
        if self._allowed and recipe != "retained":
            from .graph import GraphRecipe

            self._graph = GraphRecipe(
                self._model, recipe, self.manifest["options"].get("graph_cache_entries", 64)
            )
        self.fallback_reason = reason
        return self

    def generate(self, texts, counts, *, capture=False, stop_ids=None, cancel_after=None):
        from .runtime import generate

        if self._model is None:
            self.open()
        return generate(
            self._model,
            self._grouped,
            self._tokenizer,
            texts,
            counts,
            optimized=self._allowed,
            options=self.manifest.get("options", {}),
            capture=capture,
            stop_ids=stop_ids,
            cancel_after=cancel_after,
            fallback_reason=self.fallback_reason,
            graph=self._graph,
        )
