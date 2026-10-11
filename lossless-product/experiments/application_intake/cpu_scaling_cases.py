"""Predeclared larger scientific workloads for the saved-output CPU comparison."""

from broader_study import new_records


SPECIFICATIONS = [
    ("long_spectra", 65536, 4, 4, {"return_coef": True}, "oscillatory"),
    ("large_batch", 4096, 128, 4, {"return_coef": True}, "peaks"),
    (
        "long_changing_weights",
        32768,
        8,
        4,
        {"weight_mode": "changing", "return_coef": True},
        "peaks",
    ),
    (
        "irregular_grids",
        16384,
        8,
        4,
        {"ordering": "shuffled", "irregular": True, "use_original": True},
        "steps",
    ),
    ("sustained_requests", 4096, 8, 64, {"return_coef": True}, "oscillatory"),
    (
        "higher_order_tight_tolerance",
        8192,
        16,
        4,
        {"poly_order": 9, "max_iter": 500, "tol": 1e-8, "return_coef": True},
        "peaks",
    ),
]


def workload_cases(project, seed):
    """Fresh signals per request; frozen outputs are never tuned to these cases."""
    (project / "lossless_fixtures").mkdir(exist_ok=True)
    cases = []
    for index, (name, size, batch, requests, options, pattern) in enumerate(SPECIFICATIONS):
        calls = []
        for request in range(requests):
            records = new_records(
                project,
                f"scaling_{name}_request{request}",
                size,
                batch,
                seed + index * 100000 + request * 1000,
                options,
                pattern=pattern,
            )
            calls.append({"args": [records]})
        cases.append({"id": name, "split": "evaluation", "calls": calls})
    return cases


def description():
    return [
        dict(
            id=name, samples=size, batch=batch, requests=requests, options=options, pattern=pattern
        )
        for name, size, batch, requests, options, pattern in SPECIFICATIONS
    ]
