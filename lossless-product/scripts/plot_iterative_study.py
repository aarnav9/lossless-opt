"""Render campaign 045's discovery trajectory and separately held-out comparisons."""

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
from matplotlib.ticker import FuncFormatter


def load(path):
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    records = bundle["records"]
    result = records["summary.json"]["record"]
    loop = result["discovery"]["loop"]
    first_prompt = records["rounds/round_00/prompt.json"]["record"]
    initial = first_prompt["feedback"]["candidates"]
    points = [
        {
            "minutes": loop["seed_seconds"] / 60,
            "gain": max(
                p["geomean_speedup_vs_comparator"] for p in initial.values() if p["complete"]
            ),
            "round": 0,
        }
    ]
    for index, round_record in enumerate(loop["rounds"]):
        feedback = records[f"rounds/round_{index:02d}/feedback.json"]["record"]
        choice = round_record["choice"]
        gain = (
            feedback["candidates"][choice]["geomean_speedup_vs_comparator"]
            if choice != "deployment_reference"
            else 1.0
        )
        points.append(
            {
                "minutes": round_record["elapsed_seconds"] / 60,
                "gain": gain,
                "round": index + 1,
                "eligible": round_record["eligible"],
                "timed_out": round_record["timed_out"],
            }
        )
    for pair in result["pairs"]:
        if pair["status"] != "measured" or not pair["checks_passed"]:
            continue
        ratios = []
        for row in pair["records"]:
            assert row["case"]["split"] == "evaluation"
            assert all(v["passed"] for group in row["validation"].values() for v in group.values())
            samples = row["confirmation"]["samples_us"]
            ratio = median(samples["native_baseline"]) / median(samples["proposal"])
            assert math.isclose(ratio, row["speedup_vs_native"], rel_tol=1e-12)
            ratios.append(ratio)
        assert math.isclose(
            math.exp(sum(map(math.log, ratios)) / len(ratios)), pair["geomean"], rel_tol=1e-12
        )
    return result, points


def render(evidence, output, watermark=None):
    result, points = load(evidence)
    ink, muted, rule = "#303747", "#677388", "#e5e9ee"
    teal, slate, purple = "#3a9e89", "#8297b5", "#9b96cf"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "lossless-iterate-045",
            "font.size": 10,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.labelcolor": muted,
        }
    )
    fig = plt.figure(figsize=(13.3, 7.1), facecolor="white")
    fig.text(
        0.07,
        0.93,
        "Does measured feedback improve the next kernel?",
        fontsize=22,
        color=ink,
        weight="medium",
    )
    fig.text(
        0.07,
        0.88,
        "One-hour proposal → benchmark → revision pilot · numerical softmax on Apple M2",
        fontsize=11,
        color=muted,
    )
    left = fig.add_axes([0.09, 0.30, 0.34, 0.44])
    right = fig.add_axes([0.63, 0.30, 0.30, 0.44])
    for axis in [left, right]:
        for edge in ["top", "right", "left"]:
            axis.spines[edge].set_visible(False)
        axis.spines["bottom"].set_color(rule)
        axis.tick_params(length=0, pad=8)
        axis.grid(axis="x", color=rule, linewidth=0.8, zorder=0)
    left.set_title("Discovery incumbent", loc="left", color=ink, fontsize=13, pad=18)
    times, gains = [p["minutes"] for p in points], [p["gain"] for p in points]
    elapsed = result["discovery"]["loop"]["search_seconds"] / 60
    left.step(
        [0, *times, elapsed], [gains[0], *gains, gains[-1]], where="post", color=teal, linewidth=2
    )
    for p in points:
        completed = p.get("eligible", True)
        left.scatter(
            p["minutes"],
            p["gain"],
            color=teal if completed else slate,
            marker="o" if completed else "x",
            s=40,
            zorder=4,
        )
        label = "Seed" if not p["round"] else f"R{p['round']}"
        if not completed:
            label += " timeout" if p["timed_out"] else " failed"
        left.annotate(
            label,
            (p["minutes"], p["gain"]),
            xytext=(-3 if p["minutes"] > 50 else 3, 10 if completed else -20),
            textcoords="offset points",
            ha="right" if p["minutes"] > 50 else "left",
            color=ink,
            fontsize=9,
        )
    left.set_xlim(0, max(60, result["plan"]["search_seconds"] / 60))
    left.set_xticks([0, 15, 30, 45, 60])
    padding = max(0.05, (max(gains) - min(gains)) * 0.25)
    left.set_ylim(min(gains) - padding, max(gains) + padding * 1.4)
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    left.set_xlabel("Elapsed search minutes", labelpad=12)
    left.set_ylabel("Discovery estimate versus NumPy", labelpad=10)
    right.set_title(
        "Frozen final choice · held-out speed", loc="left", color=ink, fontsize=13, pad=18
    )
    labels = {
        "final_over_first": "Versus first round",
        "final_over_prior": "Versus prior 044",
        "final_over_retained": "Versus retained",
        "final_over_enumerated": "Versus enumeration",
    }
    colors = [slate, purple, teal, slate]
    all_values = [1.0]
    for y, pair in enumerate(result["pairs"]):
        right.text(
            -0.055,
            y,
            labels[pair["label"]],
            transform=right.get_yaxis_transform(),
            ha="right",
            va="center",
            color=ink,
            fontsize=10,
        )
        if pair.get("checks_passed"):
            for index, row in enumerate(pair["records"]):
                offset = (index - (len(pair["records"]) - 1) / 2) * 0.035
                right.scatter(
                    row["speedup_vs_native"], y + offset, color=colors[y], alpha=0.5, s=20, zorder=3
                )
                all_values.append(row["speedup_vs_native"])
            right.scatter(
                pair["geomean"],
                y,
                color=colors[y],
                marker="D",
                s=70,
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
        else:
            right.text(
                0.5,
                y,
                "No valid native comparison",
                transform=right.get_yaxis_transform(),
                ha="center",
                va="center",
                color=muted,
                fontsize=9,
            )
    right.set_yticks([])
    right.set_ylim(3.65, -0.65)
    span = max(0.1, max(all_values) - min(all_values))
    right.set_xlim(min(all_values) - span * 0.12, max(all_values) + span * 0.12)
    right.axvline(1, color=purple, linewidth=1.3)
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}×"))
    right.set_xlabel("Paired speed ratio · higher is faster", labelpad=12)
    fig.legend(
        handles=[
            Line2D(
                [], [], linestyle="none", marker="o", color=slate, alpha=0.5, label="One final case"
            ),
            Line2D(
                [],
                [],
                linestyle="none",
                marker="D",
                color=teal,
                label="Geometric mean of final cases",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.75, 0.19),
        ncol=2,
        frameon=False,
        labelcolor=muted,
        fontsize=9,
    )
    loop = result["discovery"]["loop"]
    fig.text(
        0.07,
        0.12,
        f"{sum(r['eligible'] for r in loop['rounds'])} completed rounds + {sum(r['timed_out'] for r in loop['rounds'])} timeout · {loop['attempted_slots']} attempted new slots · {elapsed:.1f} min search · final validation outside budget",
        color=ink,
        fontsize=10,
    )
    fig.text(
        0.07,
        0.06,
        "One exploratory trajectory. Discovery estimates guide selection; only untouched final cases assess generalization. No default promotion.",
        color=muted,
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    if watermark:
        fig.text(
            0.5,
            0.5,
            watermark,
            ha="center",
            va="center",
            rotation=20,
            color="#bb3344",
            alpha=0.5,
            fontsize=28,
            weight="bold",
        )
    for extension in ["png", "svg"]:
        path = output / f"iterate-045.{extension}"
        metadata = {"Title": "Lossless campaign 045: iterative search"}
        if extension == "svg":
            metadata["Date"] = result["plan"]["created_utc"].split("T")[0]
        fig.savefig(path, dpi=190, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        print(path, flush=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watermark", help="Label synthetic layout previews")
    args = parser.parse_args()
    render(args.evidence, args.output, args.watermark)
