"""Plot distribution-specific scoped speedups from the archived raw measurements."""

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


DISTRIBUTIONS = ["gaussian", "uniform", "extreme", "near_ties", "constant", "single_peak"]
LABELS = ["Gaussian", "Uniform random", "Extreme", "Near ties", "Constant rows", "Single peak"]


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def data(path):
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    records = bundle["records"]
    summary = records["<study>/summary.json"]["record"]
    assert summary["confirmation"]["passed"]
    points, observed = [], []
    for distribution in DISTRIBUTIONS:
        for interface in ["bound", "ordinary"]:
            processes = []
            for index in range(3):
                rows = [
                    value["record"]
                    for name, value in records.items()
                    if name.startswith(f"<study>/confirmation_{index}/case_")
                ]
                assert len(rows) == 9 and all(r["correct"] for r in rows)
                gains = []
                for row in rows:
                    timing = next(
                        t
                        for t in row["timings"]
                        if t["interface"] == interface and t["distribution"] == distribution
                    )
                    samples = timing["confirmation"]["samples_us"]
                    gain = median(samples[timing["reference"]]) / median(samples["first_045"])
                    assert math.isclose(
                        gain, timing["comparisons"]["first_045"]["speedup"], rel_tol=1e-12
                    )
                    gains.append(gain)
                    observed.append(
                        {
                            "process": index,
                            "case": row["case"],
                            "interface": interface,
                            "distribution": distribution,
                            "speedup": gain,
                        }
                    )
                processes.append(geomean(gains))
            points.append(
                {
                    "distribution": distribution,
                    "interface": interface,
                    "process_geomeans": processes,
                    "pooled_geomean": geomean(processes),
                }
            )
    return {
        "points": points,
        "observations": observed,
        "scope": "Nine final Fortran shapes; three fresh processes; numerical softmax on Apple M2. Lines span process geometric means, not confidence intervals.",
    }


def render(evidence, output):
    value = data(evidence)
    ink, muted, rule = "#303747", "#677388", "#e5e9ee"
    colors = {"bound": "#3a9e89", "ordinary": "#9b96cf"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "lossless-softmax-046",
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(13.3, 7.1), facecolor="white")
    fig.text(
        0.07,
        0.93,
        "Softmax gains depend on input structure",
        fontsize=22,
        color=ink,
        weight="medium",
    )
    fig.text(
        0.07,
        0.88,
        "One frozen recipe · nine held-out Fortran shapes · three fresh processes · numerical contract",
        fontsize=11,
        color=muted,
    )
    ax = fig.add_axes([0.21, 0.29, 0.57, 0.49])
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8, 9)
    ax.set_ylim(-0.6, 5.6)
    ax.invert_yaxis()
    ax.set_yticks(range(6), LABELS, color=ink)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
    ax.minorticks_off()
    ax.tick_params(length=0, pad=10, labelcolor=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=rule)
    ax.axvline(1, color="#aeb6c3", linewidth=1)
    ax.axhline(3.5, color=rule, linewidth=1)
    for point in value["points"]:
        y = DISTRIBUTIONS.index(point["distribution"]) + (
            -0.14 if point["interface"] == "bound" else 0.14
        )
        process = point["process_geomeans"]
        color = colors[point["interface"]]
        ax.plot([min(process), max(process)], [y, y], color=color, linewidth=3, alpha=0.35)
        ax.scatter(process, [y] * 3, s=16, color=color, alpha=0.55, zorder=3)
        ax.scatter(
            [point["pooled_geomean"]],
            [y],
            marker="D",
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            1.025,
            y,
            f"{point['pooled_geomean']:.3f}×",
            transform=ax.get_yaxis_transform(),
            color=color,
            va="center",
            fontsize=11,
        )
    ax.set_xlabel(
        "Speed relative to separately screened retained/library reference", color=muted, labelpad=14
    )
    handles = [
        Line2D([0], [0], marker="D", linestyle="", color=colors[k], label=label)
        for k, label in [("bound", "Bound calls"), ("ordinary", "Ordinary application calls")]
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.205, 0.855),
        frameon=False,
        ncol=2,
        labelcolor=ink,
    )
    fig.text(
        0.07,
        0.16,
        "Gaussian gain: 15.1% bound / 9.4% ordinary. Structured rows benefit from runtime-detected shortcuts.",
        fontsize=11,
        color=ink,
    )
    fig.text(
        0.07,
        0.11,
        "Diamonds: pooled geometric means. Small dots/ranges: process means, not confidence intervals. Individual calls can regress;",
        fontsize=9.4,
        color=muted,
    )
    fig.text(
        0.07,
        0.078,
        "the lowest recorded ordinary case/distribution was 0.496×. Opt-in only, with finite shape/layout guards and reference fallback.",
        fontsize=9.4,
        color=muted,
    )
    fig.text(
        0.07,
        0.034,
        "LOSSLESS  /  CAMPAIGN 046     Apple M2 · NumPy 1.26.4 · SciPy 1.16.0     Source: softmax-046.json.gz",
        fontsize=8.7,
        color=muted,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        fig.savefig(
            output.with_suffix(suffix),
            dpi=180,
            facecolor="white",
            metadata={"Date": None} if suffix == ".svg" else None,
        )
    output.with_suffix(".json").write_text(json.dumps(value, indent=2) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evidence, args.output)


if __name__ == "__main__":
    main()
