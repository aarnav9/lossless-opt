"""Plot held-out task means from the archived KernelBench-derived MLX study."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(archive, output):
    bundle = json.loads(gzip.decompress(archive.read_bytes()))
    analysis = bundle["analysis"]
    tasks = analysis["tasks"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#cad3db",
            "text.color": "#253745",
            "axes.labelcolor": "#253745",
            "xtick.color": "#526777",
            "ytick.color": "#253745",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#f7fafc")
    ax.set_facecolor("#f7fafc")
    colors = ["#168b82", "#7964b5"]
    derived = []
    for p, final in enumerate(analysis["summary"]["final_assessments"]):
        mapping = {r["task"]: r for r in final["tasks"]}
        for i, row in enumerate(tasks):
            value = mapping[row["task"]]["geomean"]
            y = i + (-0.12 if p == 0 else 0.12)
            if value is not None:
                ax.scatter(
                    value,
                    y,
                    s=34,
                    color=colors[p],
                    zorder=3,
                    label=f"Fresh process {p + 1}" if i == 0 else None,
                )
            else:
                ax.text(1, y, "failed checks", color="#ab543f", fontsize=8)
            derived.append(
                dict(task=row["task"], process=p + 1, geomean=value, confirmed=row["confirmed"])
            )
    labels = [r["task"] + "  " + r["label"] + ("  *" if r["confirmed"] else "") for r in tasks]
    ax.set_yticks(np.arange(len(tasks)), labels, fontsize=9)
    ax.invert_yaxis()
    for y in [7.5, 15.5]:
        ax.axhline(y, color="#dbe4eb", linewidth=1)
    ax.axvline(1, color="#647887", linestyle="--", linewidth=1)
    ax.grid(axis="x", color="#e2e9ee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel("Speed relative to the frozen eager / compiled MLX reference (×)", labelpad=10)
    ax.legend(loc="best", frameon=False)
    fig.suptitle(
        "KernelBench-derived MLX pilot", x=0.045, y=0.97, ha="left", fontsize=21, weight="bold"
    )
    fig.text(
        0.045,
        0.922,
        "One Astra trajectory · 20 tasks · new shapes and weights · Apple M2",
        fontsize=11,
        color="#526777",
    )
    fig.text(
        0.045,
        0.893,
        f"{analysis['correct_tasks']}/20 tasks passed final numerical checks; {analysis['confirmed_tasks']}/20 passed discovery + both final speed gates.",
        fontsize=10,
        color="#526777",
    )
    fig.text(
        0.045,
        0.042,
        "Each point combines two cases; 11 paired timing blocks per case. * = task passed its full gate.\nFloat32 tolerance: atol = rtol = 1e-4. M2-sized MLX ports, not official KernelBench CUDA scores.\nOne author trajectory does not establish repeatability across independent searches.",
        fontsize=9,
        color="#526777",
    )
    fig.subplots_adjust(left=0.44, right=0.96, top=0.845, bottom=0.145)
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
