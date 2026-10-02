"""Lossless: explicit contracts, measured optimization, reusable implementations."""

__version__ = "0.1.0a1"


def optimize(workload, *, budget=None, llm=None, output=None, resume=False):
    from .api import optimize as run

    return run(workload, budget=budget, llm=llm, output=output, resume=resume)


def load(path):
    from .artifacts import load as load_artifact

    return load_artifact(path)


from .jobs import Budget, Workload

__all__ = ["Budget", "Workload", "optimize", "load", "__version__"]
