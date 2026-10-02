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
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch


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
    background, ink, muted, mint = "#0b1220", "#f0f5fc", "#a7b5cb", "#66e6cb"
    fig = plt.figure(figsize=(13.6, 7.8), facecolor=background)
    backdrop = fig.add_axes([0, 0, 1, 1], zorder=-2)
    backdrop.set_axis_off()
    yy, xx = np.mgrid[0:1:500j, 0:1:900j]
    for color, center, opacity in [
        (mint, (0.02, 0.65), 0.055),
        ("#8e8cf7", (0.98, 0.95), 0.09),
    ]:
        layer = np.ones((*xx.shape, 4)) * np.array(to_rgba(color))
        layer[:, :, 3] = opacity * np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / 0.15)
        backdrop.imshow(layer, extent=(0, 1, 0, 1), origin="lower", aspect="auto")

    fig.text(0.06, 0.923, "LOSSLESS  /  INSTALLED ALPHA", color=mint, size=10, weight="bold")
    fig.text(0.06, 0.846, "Same batch. Less waiting.", color=ink, size=29, weight="bold")
    fig.text(
        0.06,
        0.787,
        "SmolLM2-135M-Instruct · unchanged 8-bit weights · eight fixed-count requests",
        color=muted,
        size=11,
    )
    fig.text(0.06, 0.58, f"{speedup:.2f}×", color=mint, size=66, weight="bold")
    fig.text(0.065, 0.525, "measured throughput", color=ink, size=17)

    callout = FancyBboxPatch(
        (0.06, 0.355),
        0.343,
        0.106,
        transform=fig.transFigure,
        boxstyle="round,pad=0.009,rounding_size=0.015",
        facecolor=to_rgba(mint, 0.06),
        edgecolor=to_rgba(mint, 0.18),
        linewidth=1,
    )
    fig.add_artist(callout)
    fig.text(
        0.079,
        0.416,
        f"{reduction:.1f}% less batch time",
        color=ink,
        size=15,
        weight="bold",
    )
    fig.text(
        0.079,
        0.38,
        f"95% speedup interval: {low:.3f}–{high:.3f}×",
        color=muted,
        size=10,
    )

    ax = fig.add_axes([0.535, 0.355, 0.385, 0.36], facecolor="none")
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2.3)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 0.5, 1, 1.5, 2], ["0", "0.5×", "1×", "1.5×", "2×"])
    ax.tick_params(axis="y", colors=muted, length=0, pad=8, labelsize=9)
    ax.grid(axis="y", color="#d4e0f3", alpha=0.085, linewidth=0.8)
    ax.set_ylabel("Throughput / stock serial", color=muted, size=10, labelpad=12)
    ax.axhline(0, color="#d4e0f3", alpha=0.20, lw=1)
    for x, height, color, top_alpha in [
        (0, 1, "#9caccc", 0.40),
        (1, speedup, mint, 0.65),
    ]:
        patch = FancyBboxPatch(
            (x - 0.26, 0),
            0.52,
            height,
            boxstyle="round,pad=0,rounding_size=0.035",
            facecolor=to_rgba(color, 0.045),
            edgecolor=to_rgba(color, 0.60),
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(patch)
        gradient = np.ones((250, 1, 4)) * np.array(to_rgba(color))
        gradient[:, :, 3] = np.linspace(0.06, top_alpha, 250)[:, None]
        fill = ax.imshow(
            gradient,
            extent=(x - 0.26, x + 0.26, 0, height),
            origin="lower",
            aspect="auto",
            zorder=2,
        )
        fill.set_clip_path(patch)
    ax.errorbar(
        1,
        speedup,
        yerr=[[speedup - low], [high - speedup]],
        fmt="none",
        color=mint,
        capsize=5,
        capthick=1.4,
        elinewidth=1.4,
        zorder=5,
    )
    ax.text(0, 1.12, "1.000×", ha="center", color=ink, size=14, weight="bold")
    ax.text(
        1,
        high + 0.12,
        f"{speedup:.3f}×",
        ha="center",
        color=mint,
        size=14,
        weight="bold",
    )
    ax.set_xticks(
        [0, 1],
        [
            f"Stock serial MLX\n{reference * 1000:.1f} ms / batch",
            f"Lossless alpha\n{candidate * 1000:.1f} ms / batch",
        ],
    )
    ax.tick_params(axis="x", colors=ink, length=0, pad=12, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.add_artist(
        plt.Line2D(
            [0.06, 0.94],
            [0.256, 0.256],
            transform=fig.transFigure,
            color="#cedbec",
            alpha=0.13,
        )
    )
    for x, title, detail in [
        (0.085, "Tokens", "All eight requests match"),
        (0.38, "Log probabilities", "Bitwise identical on tested cases"),
        (0.705, "Active KV state", "Bitwise identical on tested cases"),
    ]:
        fig.text(x - 0.025, 0.206, "✓", color=mint, size=15, weight="bold")
        fig.text(x, 0.209, title, color=ink, size=12, weight="bold")
        fig.text(x, 0.176, detail, color=muted, size=9)
    fig.text(
        0.06,
        0.115,
        f"Apple M2 · MLX {data['runtime']['mlx']} / mlx-lm {data['runtime']['mlx-lm']} · {repeats} paired warm repeats · descriptive 95% interval",
        color=muted,
        size=9,
    )
    fig.text(
        0.06,
        0.081,
        "33-token prefixes · 4–32 output tokens per request · Loading and first-use compilation excluded",
        color=muted,
        size=9,
    )
    fig.text(
        0.06,
        0.047,
        "One recorded workload; finite validation. Batching may increase first-token latency. Retained recipe; no live LLM call.",
        color=muted,
        size=9,
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
