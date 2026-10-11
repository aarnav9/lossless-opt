"""Extra correctness controls, excluded from the count of performance experiments."""

import argparse
import importlib.util
from pathlib import Path

from common import DEFAULT_RUN, HERE, write, stamp


def main():
    spec = importlib.util.spec_from_file_location("campaign_driver", HERE / "study.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    from broader_study import new_records

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_RUN)
    p.add_argument("--candidate", type=int, required=True)
    p.add_argument("--family", type=int, choices=[1, 5], default=1)
    p.add_argument("--arm", choices=["control", "diagnostics"])
    a = p.parse_args()
    if a.family == 5 and a.arm is None:
        p.error("family 5 requires --arm")
    label = f"f{a.family}_{a.candidate:02d}" + (f"_{a.arm}" if a.family == 5 else "")
    project = a.output / f"f{a.family}_{a.candidate:02d}"
    if a.family == 5:
        project = project / a.arm
    project = project / "project"
    out = a.output / f"conformance_{label}"
    out.mkdir()
    (out / "lossless_fixtures").mkdir()
    specs = [
        ("signed_zero", 259, {}, dict(zeros=True)),
        ("tiny", 263, {}, dict(scale=1e-150)),
        ("large", 271, {}, dict(scale=1e140)),
        ("constant_order", 269, dict(poly_order=0), {}),
        ("one_iteration", 277, dict(max_iter=1, tol=1e-15), {}),
        ("tight_stop", 283, dict(max_iter=500, tol=1e-12), {}),
        ("zero_weights", 293, dict(weight_mode="changing"), dict(zero_weights=True)),
        ("strided", 307, dict(use_original=True), dict(strided=True)),
        ("readonly", 311, dict(return_coef=True), dict(readonly=True)),
        (
            "combined",
            313,
            dict(
                weight_mode="sparse",
                ordering="descending",
                use_original=True,
                mask_initial_peaks=True,
            ),
            dict(strided=True),
        ),
    ]
    checks = []
    for index, (name, size, options, transform) in enumerate(specs):
        records = new_records(out, name, size, 2, 634377 + index, options)
        case = dict(id=name, calls=[dict(args=[records])] * 3, transform=transform)
        values = {}
        for variant, project in [
            ("reference", a.output / "baseline"),
            ("candidate", project),
        ]:
            values[variant] = driver.execute(
                dict(project=str(project.resolve()), fixtures=str(out.resolve()), case=case),
                out / name / variant,
                "correctness",
            )
        check = driver.check_pair(
            values["reference"],
            values["candidate"],
            out / name / "reference/result.npz",
            out / name / "candidate/result.npz",
        )
        checks.append(dict(id=name, **check))
        stamp(f"CONFORMANCE {name} exact={check['exact']} numerical={check['numerical']}")
    write(
        out / "report.json",
        dict(
            exact=all(c["exact"] for c in checks),
            numerical=all(c["numerical"] for c in checks),
            cases=checks,
        ),
    )


if __name__ == "__main__":
    main()
