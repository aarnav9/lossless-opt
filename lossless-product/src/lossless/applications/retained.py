"""Local, model-free composition of source recipes on recognized projects."""

import copy
import hashlib
import json
from pathlib import Path


def retained_plan(project, *, workers=(2, 4), compiled=True, budget=240):
    """Build a reviewable search plan for pinned pybaselines ModPoly sources.

    Returns source proposals, not an acceptance decision. Pass the result to
    ``applications.search(config, plan=..., output=...)`` with user-owned fresh
    discovery/evaluation fixtures. The source project is never modified here.
    Pools are offered only for the recognized independent spectrum wrapper.
    Explicit prepared handles use ``retained.pybaselines.ModPolyPlan`` and must
    be benchmarked under their separate setup lifecycle, not hidden in calls.
    """
    project = Path(project).resolve()
    resource = Path(__file__).parents[1] / "retained"
    data = json.loads((resource / "modpoly.json").read_text())
    for name, digest in data["source_files"].items():
        path = project / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"no retained source recipe for this revision: {name}")
    if any(type(n) is not int or n not in (2, 4) for n in workers):
        raise ValueError("retained worker candidates support 2 and 4 workers")
    base = data["source_edits"]
    candidates = [
        dict(
            schema_version=1,
            hypothesis="Compose retained ModPoly scratch, metadata and dispatch edits.",
            edits=base,
        )
    ]
    scope = [{k: e[k] for k in ("path", "symbol")} for e in base]
    helpers = []
    if compiled:
        helpers = [
            dict(
                name="modpoly",
                functions=[
                    dict(
                        name="iterate",
                        returns="int",
                        arguments=["pointer"] * 12 + ["int64"] * 4 + ["int"] * 4 + ["double"] * 2,
                    )
                ],
            )
        ]
        edits = [data["compiled_edit"] if e["symbol"] == "_Polynomial.modpoly" else e for e in base]
        candidates.append(
            dict(
                schema_version=1,
                hypothesis="Stack guarded whole-loop compilation with retained source edits, preserving NumPy BLAS.",
                edits=edits,
                native_sources=[dict(name="modpoly", source=(resource / "modpoly.c").read_text())],
            )
        )
    wrapper = project / "spectrum_workload.py"
    recognized = (
        wrapper.is_file()
        and hashlib.sha256(wrapper.read_bytes()).hexdigest() == data["wrapper_sha256"]
    )
    if recognized:
        scope.extend(
            [
                dict(path="spectrum_workload.py", symbol=s)
                for s in ("Processor.__init__", "Processor.fit", "make_processor")
            ]
        )
        cache_edits = [
            dict(
                path="spectrum_workload.py",
                symbol="Processor.__init__",
                body="from lossless.retained.pybaselines import ModPolyCache\nself.fitters = {}\nself._plans = ModPolyCache(capacity=2)\n",
            ),
            dict(
                path="spectrum_workload.py",
                symbol="Processor.fit",
                body="return self._plans(x, y, **options)\n",
            ),
        ]
        # Ablations and compositions are all measured against the same original.
        originals = list(candidates)
        for proposal in originals:
            c = copy.deepcopy(proposal)
            c["edits"] += copy.deepcopy(cache_edits)
            c["hypothesis"] += " Add bounded owned metadata plans."
            candidates.append(c)
        for n in dict.fromkeys(workers):
            for proposal in originals:
                c = copy.deepcopy(proposal)
                c["edits"] += copy.deepcopy(cache_edits) + [
                    dict(
                        path="spectrum_workload.py",
                        symbol="make_processor",
                        body=f"from lossless.execution import IndependentMap\nreturn IndependentMap(Processor, workers={n}, minimum_batch=4, warm=True)\n",
                    )
                ]
                c["hypothesis"] += (
                    f" Add bounded plans and {n} persistent workers with one BLAS thread each."
                )
                candidates.append(c)
        # Initialize native binding discovery in the declared factory, including
        # each spawned worker factory. It must not hide in the first signal call.
        for candidate in candidates:
            if candidate.get("native_sources"):
                prefix = "from lossless.retained import modpoly_native\nfrom _lossless_native_modpoly import iterate\n"
                init = next(
                    (
                        e
                        for e in candidate["edits"]
                        if e["path"] == "spectrum_workload.py"
                        and e["symbol"] == "Processor.__init__"
                    ),
                    None,
                )
                if init is None:
                    candidate["edits"].append(
                        dict(
                            path="spectrum_workload.py",
                            symbol="Processor.__init__",
                            body=prefix + "self.fitters = {}\n",
                        )
                    )
                else:
                    init["body"] = prefix + init["body"]
    return dict(
        schema_version=1,
        source_scope=scope,
        native_helpers=helpers,
        candidates=candidates,
        max_candidates=len(candidates),
        architecture=False,
        discovery_resources=True,
        wall_time_seconds=budget,
        final_reserve_seconds=min(60, budget / 3),
        contract="Pinned pybaselines ModPoly; finite exact complete output tree, input mutation and retained output ownership. Source/metadata/compiled/pool combinations are candidates, not defaults. Pool proposals require independent spectra and an observer covering descendant memory. Preparation and startup must be included according to the deployment lifecycle.",
    )
