"""Plot campaign 044's actual author times and independently measured frozen winners."""

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
from matplotlib.ticker import FuncFormatter, MaxNLocator


def read(path):
    return json.loads(path.read_text())


def load(root):
    bundle = json.loads(gzip.decompress(root.read_bytes())) if root.is_file() else None
    data = bundle["summary"]["record"] if bundle else read(root / "summary.json")
    for condition in data["conditions"]:
        name = condition["condition"]
        authors = (
            [a["receipt"]["record"] for a in bundle["authors"][name]]
            if bundle
            else [read(root / name / f"author_{i}" / "receipt.json") for i in range(3)]
        )
        condition_summary = (
            bundle["condition_summaries"][name]["record"]
            if bundle
            else read(root / name / "summary.json")
        )
        assert authors == condition["authors"]
        assert condition["completed_authors"] == sum(a["eligible"] for a in authors)
        paired = [
            p
            for p in condition_summary["paired_winners"]
            if p["arm"] == "llm" and p["status"] == "measured" and p["checks_passed"]
        ]
        for pair in paired:
            assert len(pair["records"]) == 9
            ratios = []
            for row in pair["records"]:
                assert row["status"] == "ok" and row["case"]["split"] == "evaluation"
                assert all(
                    v["passed"] for checks in row["validation"].values() for v in checks.values()
                )
                samples = row["confirmation"]["samples_us"]
                ratio = median(samples["native_baseline"]) / median(samples["proposal"])
                assert math.isclose(ratio, row["speedup_vs_native"], rel_tol=1e-12)
                ratios.append(ratio)
            assert math.isclose(
                math.exp(sum(map(math.log, ratios)) / 9), pair["geomean_vs_retained"], rel_tol=1e-12
            )
        assert [p["geomean_vs_retained"] for p in paired] == condition["paired_gains_vs_retained"]
        condition["paired"] = paired
    return data


def render(root, output):
    data = load(root)
    ink, muted, rule = "#303747", "#677388", "#e5e9ee"
    colors = ["#8297b5", "#3a9e89", "#9b96cf", "#8297b5"]
    labels = {
        "announced_5m": "Announced · 5 min",
        "announced_30m": "Announced · 30 min",
        "announced_60m": "Announced · 60 min",
        "unannounced_60m": "Unannounced · 60 min",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "lossless-timers-044",
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(13.3, 7.1), facecolor="white")
    fig.text(
        0.075,
        0.92,
        "Does a longer author allowance produce faster kernels?",
        fontsize=21,
        color=ink,
        weight="medium",
    )
    fig.text(
        0.075,
        0.87,
        "Announced deadlines versus an unannounced control · 3 independent authors per condition",
        fontsize=11,
        color=muted,
    )
    left = fig.add_axes([0.255, 0.29, 0.285, 0.46])
    right = fig.add_axes([0.64, 0.29, 0.29, 0.46])
    for ax in [left, right]:
        ax.set_ylim(-0.6, 3.6)
        ax.invert_yaxis()
        ax.set_yticks([])
        for side in ["left", "right", "top"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(rule)
        ax.grid(axis="x", color=rule, linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", length=0, pad=10)
    left.set_title("Actual time used", loc="left", fontsize=13, color=ink, pad=18)
    left.set_xlim(-1, 65)
    left.set_xticks([0, 15, 30, 45, 60])
    left.set_xlabel("Minutes · line ends at allowed maximum", labelpad=12, fontsize=9)
    right.set_title("Held-out kernel performance", loc="left", fontsize=13, color=ink, pad=18)
    gains = [g for c in data["conditions"] for g in c["paired_gains_vs_retained"]]
    right.set_xlim(min([0.96] + [g * 0.96 for g in gains]), max([1.08] + [g * 1.04 for g in gains]))
    right.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    right.set_xlabel("Speed relative to retained · higher is faster", labelpad=12, fontsize=9)
    right.axvline(1, color="#9b96cf", linewidth=1.4, zorder=1)
    offsets = [-0.12, 0, 0.12]
    for y, condition in enumerate(data["conditions"]):
        color = colors[y]
        left.text(
            -0.10,
            y - 0.06,
            labels[condition["condition"]],
            transform=left.get_yaxis_transform(),
            ha="right",
            va="center",
            color=ink,
            fontsize=11,
            weight="medium",
        )
        left.text(
            -0.10,
            y + 0.21,
            f"{condition['completed_authors']}/3 authors completed",
            transform=left.get_yaxis_transform(),
            ha="right",
            va="center",
            color=muted,
            fontsize=9,
        )
        left.plot(
            [0, condition["seconds"] / 60],
            [y, y],
            color=color,
            alpha=0.20,
            linewidth=9,
            solid_capstyle="round",
            zorder=2,
        )
        left.plot(
            [condition["seconds"] / 60] * 2,
            [y - 0.2, y + 0.2],
            color=color,
            linewidth=1.4,
            zorder=3,
        )
        for index, author in enumerate(condition["authors"]):
            left.scatter(
                author["elapsed_seconds"] / 60,
                y + offsets[index],
                marker="o" if author["eligible"] else "x",
                s=42,
                color=color,
                linewidths=1.5,
                zorder=4,
            )
        for pair in condition["paired"]:
            right.scatter(
                pair["geomean_vs_retained"],
                y + offsets[pair["replicate"]],
                marker="D",
                s=43,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
        if not condition["paired"]:
            right.text(
                0.5,
                y,
                "No completed proposals"
                if not condition["completed_authors"]
                else "No selected native winner",
                color=muted,
                transform=right.get_yaxis_transform(),
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="o", color=muted, label="Completed author"),
            Line2D([], [], linestyle="none", marker="x", color=muted, label="Failed / timed out"),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="D",
                color=muted,
                label="Frozen winner · geometric mean of 9 cases",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.54, 0.17),
        ncol=3,
        frameon=False,
        labelcolor=muted,
        fontsize=9,
    )
    total = sum(c["search_seconds_total"] for c in data["conditions"]) / 60
    fig.text(
        0.075,
        0.095,
        f"Total LLM-arm cost: {total:.1f} minutes · actual authoring + discovery + evaluation",
        color=ink,
        fontsize=10,
    )
    fig.text(
        0.075,
        0.052,
        "Exploratory, n=3 per condition. Numerical softmax on Apple M2. Limits may be unused; missing winners have no measured speed.",
        color=muted,
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        path = output / f"timers-044.{extension}"
        metadata = {"Title": "Lossless campaign 044: author allowances and kernel performance"}
        if extension == "svg":
            metadata["Date"] = data["plan"]["created_utc"].split("T")[0]
        fig.savefig(path, dpi=190, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        print(path, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--study", type=Path, help="Completed local study directory")
    inputs.add_argument("--evidence", type=Path, help="Curated .json.gz evidence bundle")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.study or args.evidence, args.output)
