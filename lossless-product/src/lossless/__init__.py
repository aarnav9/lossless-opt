"""Lossless: explicit contracts, measured optimization, reusable implementations."""

__version__ = "0.1.0a1"


def optimize(workload, *, budget=None, llm=None, output=None, resume=False):
    from .api import optimize as run

    return run(workload, budget=budget, llm=llm, output=output, resume=resume)


def load(path):
    from .artifacts import load as load_artifact

    return load_artifact(path)


def profile(workload, *, artifact=None, repeats=None, budget=None, output=None, analysis=None):
    from .profiling import profile as run

    return run(
        workload,
        artifact=artifact,
        repeats=repeats,
        budget=budget,
        output=output,
        analysis=analysis,
    )


def replay(config, *, output, repeats=None, budget=None, split="discovery"):
    from .applications import replay as run

    return run(config, output=output, repeats=repeats, budget=budget, split=split)


from .jobs import Budget, Workload

__all__ = ["Budget", "Workload", "optimize", "load", "profile", "replay", "__version__"]
