"""Render the README figure from the checked-in evaluation record; no GPU work.

Requires matplotlib (an optional documentation dependency, not a runtime dependency).
From the repository root: python lossless-product/scripts/plot_retention.py
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def render(data_path, output):
    data = json.loads(data_path.read_text())
    evaluation = data["evaluation"]
    checks = evaluation["checks"]
    if not evaluation["exact"] or len(checks) != data["workload"]["requests"]:
        raise ValueError("Figure requires exact evidence for every request")
    if not all(
        c["tokens_equal"] and c["probabilities"]["bitwise"] and c["kv"]["bitwise"] for c in checks
    ):
        raise ValueError("A recorded exact comparison failed")
    samples = evaluation["samples_seconds"]
    repeats = data["measurement"]["repeats"]
    if any(len(samples[k]) != repeats for k in ("reference", "candidate")):
        raise ValueError("Repeat counts do not match the measurement record")
    if not all(math.isfinite(x) and x > 0 for values in samples.values() for x in values):
        raise ValueError("Timing samples must be finite and positive")
    reference, candidate = (median(samples[k]) for k in ("reference", "candidate"))
    speedup = reference / candidate
    low, high = evaluation["ci95"]
    if not math.isclose(speedup, evaluation["speedup"], rel_tol=1e-12):
        raise ValueError("Recorded speedup differs from the ratio of medians")
    if not 0 < low <= speedup <= high:
        raise ValueError("Invalid recorded interval")
    reduction = 100 * (1 - candidate / reference)
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "lossless-retention",
        }
    )
    background, ink, muted = "#ffffff", "#303747", "#677388"
    slate, teal, baseline = "#8297b5", "#3a9e89", "#9b96cf"
    fig = plt.figure(figsize=(12.8, 7.6), facecolor=background)
    fig.text(
        0.5,
        0.93,
        "Lossless: measured MLX inference performance",
        ha="center",
        color=ink,
        size=16,
        weight="medium",
    )
    fig.text(
        0.5,
        0.876,
        f"{data['model']}  |  {data['device']}  |  unchanged 8-bit weights",
        ha="center",
        color=muted,
        size=10,
    )

    ax = fig.add_axes([0.09, 0.325, 0.86, 0.49], facecolor=background)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(0, 2.3)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 0.5, 1, 1.5, 2], ["0", "0.5", "1.0", "1.5", "2.0"])
    ax.tick_params(axis="y", colors=muted, length=3, width=0.6, pad=7, labelsize=9)
    ax.grid(axis="y", color="#e9edf2", linewidth=0.7)
    ax.set_ylabel("Throughput relative to stock serial MLX (×)", color=muted, size=10, labelpad=12)
    for x, height, color in [(0, 1, slate), (1, speedup, teal)]:
        ax.bar(
            x,
            height,
            width=0.48,
            facecolor=to_rgba(color, 0.18),
            edgecolor=to_rgba(color, 0.85),
            linewidth=1.0,
            zorder=3,
        )
    ax.axhline(1, color=baseline, lw=1, ls=(0, (5, 3)), zorder=4)
    ax.text(
        1.66,
        1.045,
        "Stock serial baseline",
        ha="right",
        va="bottom",
        color=baseline,
        size=8.5,
    )
    ax.errorbar(
        1,
        speedup,
        yerr=[[speedup - low], [high - speedup]],
        fmt="none",
        color=teal,
        capsize=5,
        capthick=1.1,
        elinewidth=1.1,
        zorder=5,
    )
    ax.text(0, 1.09, "1.000×", ha="center", color=ink, size=12)
    ax.text(1, high + 0.10, f"{speedup:.3f}×", ha="center", color=ink, size=12)
    ax.set_xticks(
        [0, 1],
        [
            f"Stock serial MLX\n{reference * 1000:.1f} ms / batch",
            f"Lossless alpha · retained recipe\n{candidate * 1000:.1f} ms / batch",
        ],
    )
    ax.tick_params(axis="x", colors=ink, length=0, pad=12, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dce2eb")
        ax.spines[side].set_linewidth(0.7)

    ax.text(
        0.025,
        0.93,
        f"{repeats} paired warm repeats\n"
        f"Median batch time reduction: {reduction:.1f}%\n"
        f"Speedup 95% interval: [{low:.3f}, {high:.3f}]×",
        transform=ax.transAxes,
        va="top",
        color=muted,
        size=9,
        linespacing=1.7,
        bbox={
            "boxstyle": "round,pad=0.6,rounding_size=0.15",
            "facecolor": "#f8fafc",
            "edgecolor": "#e9edf2",
            "linewidth": 0.7,
        },
    )

    fig.text(
        0.09,
        0.206,
        "Exact comparison: tokens, log-probability bits and active KV state match on all eight requests.",
        color=ink,
        size=9.5,
    )
    fig.text(
        0.09,
        0.161,
        f"MLX {data['runtime']['mlx']} / mlx-lm {data['runtime']['mlx-lm']}"
        "  ·  33-token prefixes  ·  4–32 output tokens per request  ·  evaluation split",
        color=muted,
        size=8.5,
    )
    fig.text(
        0.09,
        0.127,
        "Bars: ratio of median batch times. Whisker: descriptive paired bootstrap 95% interval, within-host timing only.",
        color=muted,
        size=8.5,
    )
    fig.text(
        0.09,
        0.093,
        "Loading and first-use compilation excluded. Batching may increase first-token latency.",
        color=muted,
        size=8.5,
    )
    fig.text(
        0.09,
        0.059,
        "One recorded workload; finite validation, not a universal proof. Retained recipe; no live LLM call.",
        color=muted,
        size=8.5,
    )
    for extension in ["png", "svg"]:
        destination = output / f"mlx-retention.{extension}"
        metadata = {"Title": "Lossless installed MLX alpha: recorded exact-retention comparison"}
        if extension == "svg":
            metadata["Date"] = data["recorded_date"]
            metadata["Description"] = (
                f"Stock serial MLX: {reference * 1000:.1f} milliseconds per batch. "
                f"Lossless retained recipe: {candidate * 1000:.1f} milliseconds. "
                f"Speedup {speedup:.3f}x; descriptive 95% interval [{low:.3f}, {high:.3f}]. "
                "All tested tokens, log-probability bits and active KV state match."
            )
        fig.savefig(
            destination,
            dpi=190 if extension == "png" else 72,
            facecolor=background,
            metadata=metadata,
        )
        if extension == "svg":
            destination.write_text(
                "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
            )
        print(destination, flush=True)
    plt.close(fig)
    print(
        f"Validated {repeats} paired samples; measured ratio {speedup:.12f}x.",
        flush=True,
    )


if __name__ == "__main__":
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=assets / "mlx-retention-evaluation.json")
    parser.add_argument("--output", type=Path, default=assets)
    options = parser.parse_args()
    render(options.data, options.output)
