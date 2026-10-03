"""Render campaign 043's authoring-cap comparison from checked-in evidence.

Requires matplotlib, an optional documentation dependency. No model or benchmark runs.
From the repository root: python lossless-product/scripts/plot_search_time.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


def minutes_seconds(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def evidence(path):
    data = json.loads(path.read_text())
    pilot = data["pilot"]
    assert pilot["plan"]["record"]["author_timeout_seconds"] == 300
    assert data["plan"]["record"]["author_timeout_seconds"] == 1800
    assert not pilot["evaluation_opened"]
    assert len(pilot["authors"]) == len(data["authors"]) == 3
    for author in pilot["authors"]:
        assert author["receipt"]["record"]["timed_out"] and author["response"] is None
    authors = [a["receipt"]["record"] for a in data["authors"]]
    assert all(
        not a["timed_out"]
        and a["returncode"] == 0
        and a["blinding_passed"]
        and 300 < a["elapsed_seconds"] < 1800
        for a in authors
    )
    assert all(len(a["response"]["record"]["proposals"]) == 3 for a in data["authors"])
    summary = data["summary"]["record"]
    arms = sorted((r for r in summary["arms"] if r["arm"] == "llm"), key=lambda r: r["replicate"])
    winners = sorted(
        (r for r in summary["paired_winners"] if r["arm"] == "llm"),
        key=lambda r: r["replicate"],
    )
    assert len(arms) == len(winners) == 3
    case_speeds = []
    for index, (arm, winner) in enumerate(zip(arms, winners)):
        assert arm["replicate"] == winner["replicate"] == index
        assert arm["accepted"] and winner["incremental_passed"] and winner["checks_passed"]
        assert len(winner["records"]) == 9
        speeds = []
        for row in winner["records"]:
            assert row["status"] == "ok" and row["case"]["split"] == "evaluation"
            assert all(
                check["passed"]
                for checks in row["validation"].values()
                for check in checks.values()
            )
            samples = row["confirmation"]["samples_us"]
            assert len(samples["native_baseline"]) == len(samples["proposal"]) == 15
            ratio = median(samples["native_baseline"]) / median(samples["proposal"])
            assert math.isclose(ratio, row["speedup_vs_native"], rel_tol=1e-12)
            assert row["ci95_vs_native"][0] > 1
            speeds.append(ratio)
        assert math.isclose(
            math.exp(sum(math.log(v) for v in speeds) / len(speeds)),
            winner["geomean_vs_retained"],
            rel_tol=1e-12,
        )
        case_speeds.append(speeds)
    checks = [
        check
        for name, run in data["runs"].items()
        if name.startswith("llm_")
        for job in run["job_results"].values()
        for check in job["record"].get("validation", {}).get("proposal", {}).values()
    ]
    assert len(checks) == 810 and all(c["passed"] for c in checks)
    paybacks = [
        p["search_only_break_even_calls"] for w in winners for p in w["payback_vs_retained"]
    ]
    assert len(paybacks) == 27 and all(type(p) is int and p > 0 for p in paybacks)
    return {
        "date": data["plan"]["record"]["created_utc"].split("T")[0],
        "authors": authors,
        "speeds": case_speeds,
        "means": [w["geomean_vs_retained"] for w in winners],
        "total_minutes": sum(a["total_observed_seconds"] for a in arms) / 60,
        "payback_min": min(paybacks),
        "payback_max": max(paybacks),
    }


def render(data_path, output):
    data = evidence(data_path)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "lossless-search-time",
        }
    )
    # Same palette and typography as the existing README retention figure.
    background, ink, muted = "#ffffff", "#303747", "#677388"
    slate, teal, baseline = "#8297b5", "#3a9e89", "#9b96cf"
    rule = "#e9edf2"
    fig = plt.figure(figsize=(12.8, 8.5), facecolor=background)

    def text(x, y, value, size=10, color=ink, **kwargs):
        return fig.text(x, y, value, size=size, color=color, **kwargs)

    def card(x, y, width, height, color):
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                transform=fig.transFigure,
                facecolor=to_rgba(color, 0.075),
                edgecolor=to_rgba(color, 0.3),
                linewidth=0.8,
                zorder=0,
            )
        )

    text(
        0.5, 0.944, "Lossless: search time and kernel performance", 18, ha="center", weight="medium"
    )
    text(
        0.5,
        0.901,
        "Native softmax  |  Apple M2  |  independent gpt-6-astra / xhigh authors",
        10,
        muted,
        ha="center",
    )

    card(0.085, 0.694, 0.385, 0.139, slate)
    card(0.53, 0.694, 0.385, 0.139, teal)
    text(0.106, 0.797, "5-minute cap per author · pilot", 11, slate, weight="medium")
    text(0.106, 0.748, "0 of 3 authors completed", 18, weight="medium")
    text(0.106, 0.710, "All timed out; kernel speed unmeasured.", 9.5, muted)
    text(0.551, 0.797, "30-minute cap per author · follow-up", 11, teal, weight="medium")
    text(0.551, 0.748, "3 of 3 authors completed", 18, weight="medium")
    times = [minutes_seconds(a["elapsed_seconds"]) for a in data["authors"]]
    text(0.551, 0.710, "Actual authoring: " + " / ".join(times), 9.5, muted)

    text(0.085, 0.634, "Performance of the three selected LLM kernels", 12, weight="medium")
    text(
        0.085,
        0.600,
        "Fresh paired measurements against independently selected retained kernels",
        9.5,
        muted,
    )
    ax = fig.add_axes([0.285, 0.350, 0.485, 0.216], facecolor=background)
    ax.set_xlim(0.97, 1.31)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.set_xticks([1, 1.1, 1.2, 1.3], ["1.00×", "1.10×", "1.20×", "1.30×"])
    ax.tick_params(axis="x", colors=muted, length=3, width=0.6, pad=8, labelsize=9)
    ax.grid(axis="x", color=rule, linewidth=0.7)
    ax.axvline(1, color=baseline, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#dce2eb")
    ax.spines["bottom"].set_linewidth(0.7)
    for index, (speeds, mean) in enumerate(zip(data["speeds"], data["means"])):
        y = 2 - index
        ax.axhline(y, color=rule, linewidth=0.7, zorder=0)
        # Fixed offsets distinguish cases; they carry no second measured quantity.
        offsets = [-0.13, 0.10, -0.02, 0.13, -0.08, 0.06, -0.10, 0.02, 0.10]
        ax.scatter(
            speeds,
            [y + o for o in offsets],
            s=33,
            color=to_rgba(teal, 0.45),
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            [mean],
            [y],
            s=94,
            marker="D",
            color=teal,
            edgecolors=background,
            linewidths=1.2,
            zorder=4,
        )
        ax.text(
            -0.075,
            y + 0.05,
            f"Author {index + 1}",
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            color=ink,
            fontsize=11,
        )
        ax.text(
            -0.075,
            y - 0.29,
            f"finished in {times[index]}",
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            color=muted,
            fontsize=9,
        )
        ax.text(
            1.075,
            y + 0.05,
            f"{mean:.3f}×",
            va="center",
            transform=ax.get_yaxis_transform(),
            color=ink,
            fontsize=15,
            weight="medium",
        )
        ax.text(
            1.075,
            y - 0.29,
            f"+{(mean - 1) * 100:.1f}% vs retained",
            va="center",
            transform=ax.get_yaxis_transform(),
            color=teal,
            fontsize=9,
        )
    ax.get_xticklabels()[0].set_color(baseline)
    text(0.525, 0.286, "Relative kernel speed · higher is faster", 9.5, muted, ha="center")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=5,
            color=to_rgba(teal, 0.5),
            label="One held-out shape/layout",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=6,
            color=teal,
            label="Geometric mean of 9 cases",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.525, 0.248),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        labelcolor=muted,
        handletextpad=0.4,
        columnspacing=2.0,
    )

    card(0.085, 0.115, 0.83, 0.082, slate)
    for x in [0.365, 0.650]:
        fig.add_artist(
            Line2D([x, x], [0.12, 0.185], transform=fig.transFigure, color=rule, linewidth=1)
        )
    for x, title, value, detail in [
        (
            0.103,
            "Numerical validation",
            "9 / 9 kernels passed",
            "810 candidate/distribution checks",
        ),
        (
            0.390,
            "Total cost · three LLM runs",
            f"{data['total_minutes']:.1f} minutes",
            "Authoring + search + final evaluation",
        ),
        (
            0.676,
            "Search-only payback",
            f"{data['payback_min'] / 1e6:.1f}M–{data['payback_max'] / 1e9:.2f}B calls",
            "Versus retained; setup not included",
        ),
    ]:
        text(x, 0.180, title, 8.5, muted)
        text(x, 0.149, value, 13, weight="medium")
        text(x, 0.123, detail, 8, muted)

    text(
        0.085,
        0.068,
        "Fresh sessions at each cap. The 30-minute allowance was not fully used; no time–speedup curve was measured.",
        8.5,
        muted,
    )
    text(
        0.085,
        0.039,
        "One CPU operator/device under numerical tolerances; experimental kernels. Cost excludes the pilot and study overhead.",
        8.5,
        muted,
    )

    output.mkdir(parents=True, exist_ok=True)
    description = (
        "Five-minute pilot: 0 of 3 authors completed, kernel speed unmeasured. "
        f"Thirty-minute cap: 3 of 3 completed in {', '.join(times)}. "
        f"Selected kernels measured {', '.join(f'{m:.3f}x' for m in data['means'])} retained speed across nine held-out cases each. "
        f"All nine proposed kernels passed numerical checks. Search cost {data['total_minutes']:.1f} minutes across three LLM runs; "
        f"search-only payback {data['payback_min'] / 1e6:.1f} million to {data['payback_max'] / 1e9:.2f} billion calls. Fresh sessions, not a measured scaling curve."
    )
    for extension in ["png", "svg"]:
        destination = output / f"search-time-043.{extension}"
        metadata = {
            "Title": "Lossless: authoring time limits and measured kernel performance",
            "Description": description,
        }
        if extension == "svg":
            metadata["Date"] = data["date"]
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
        "Validated pilot receipts, completed authors, 810 candidate checks and all 27 paired case timings.",
        flush=True,
    )


if __name__ == "__main__":
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=assets / "search-043.json")
    parser.add_argument("--output", type=Path, default=assets)
    args = parser.parse_args()
    render(args.data, args.output)
