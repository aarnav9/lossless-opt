"""Fresh-process held-out speed intervals from the archived 048 raw timing blocks."""

import argparse
import gzip
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lossless._native.common import interval


def plot(archive, output):
    bundle = json.loads(gzip.decompress(archive.read_bytes()))
    records = bundle["records"]
    plan = records["<study>/plan.json"]["record"]
    refs = records["<study>/references.json"]["record"]
    summaries = bundle["analysis"]["summary"]
    paths = [f"<study>/workers/final_{i + 1}.json" for i in range(plan["final_processes"])]
    reports = [records[p]["record"] for p in paths if p in records]
    if not reports or any(len(r["records"]) != len(plan["evaluation"]) for r in reports):
        raise ValueError("complete fresh-process records required for this plot")
    colors = ["#168b82", "#7964b5"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#cad3db",
            "text.color": "#253745",
            "axes.labelcolor": "#253745",
            "xtick.color": "#526777",
            "ytick.color": "#253745",
        }
    )
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="#f7fafc")
    ax.set_facecolor("#f7fafc")
    yy = np.arange(len(plan["evaluation"]))
    labels = [c["key"].replace("_", " ") + "  /  " + refs[c["key"]] for c in plan["evaluation"]]
    derived = []
    for process, report in enumerate(reports):
        for i, row in enumerate(report["records"]):
            if not row["exact"].get("candidate") or not row["exact"].get("reference"):
                raise ValueError("do not plot incorrect cases as speed wins")
            reference, candidate = row["samples"]["reference"], row["samples"]["candidate"]
            ratio = statistics.median(reference) / statistics.median(candidate)
            low, high = interval(reference, candidate, 4801)
            y = yy[i] + (-0.13 if process == 0 else 0.13)
            ax.plot([low, high], [y, y], color=colors[process], linewidth=2, alpha=0.75)
            ax.scatter(
                [ratio],
                [y],
                s=42,
                color=colors[process],
                zorder=3,
                label=f"Fresh process {process + 1}" if i == 0 else None,
            )
            derived.append(
                dict(case=row["case"]["key"], process=process + 1, ratio=ratio, ci95=[low, high])
            )
    ax.axvline(1, color="#647887", linewidth=1, linestyle="--")
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Speed relative to the frozen strongest applicable reference (×)")
    ax.grid(axis="x", color="#e2e9ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.legend(loc="best", frameon=False)
    outcome = (
        "PASSED THE FULL GATE"
        if summaries["experimental_gate_passed"]
        else "DID NOT PASS THE FULL GATE"
    )
    fig.suptitle(
        "Whole-request MLX optimization", x=0.055, y=0.97, ha="left", fontsize=20, weight="bold"
    )
    fig.text(0.055, 0.90, f"One Astra author trajectory · {outcome}", fontsize=11, color="#526777")
    fig.text(
        0.055,
        0.055,
        "Full probabilities + active KV captured · Apple M2 · 11 timing blocks per process\nIntervals bootstrap paired blocks; they do not establish repeatability across independent authors.",
        fontsize=9,
        color="#526777",
    )
    fig.subplots_adjust(left=0.34, right=0.96, top=0.83, bottom=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(output.with_suffix(suffix), dpi=180, facecolor=fig.get_facecolor())
    output.with_suffix(".plot.json").write_text(json.dumps(derived, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plot(a.archive, a.output)
