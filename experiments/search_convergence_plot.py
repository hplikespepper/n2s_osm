#!/usr/bin/env python3
"""Plot search convergence from the component-history summary CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("objective", r"Best-so-far objective $\mathcal{J}$"),
    ("distance", r"Distance $D$ of best-$\mathcal{J}$ solution"),
    ("makespan", r"Makespan $M$ of best-$\mathcal{J}$ solution"),
)
SERIES_STYLES = (
    {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
)


def parse_formats(value):
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        raise argparse.ArgumentTypeError(
            "unsupported format(s): {}; choose png, pdf, or svg".format(
                ", ".join(unsupported)
            )
        )
    if not formats:
        raise argparse.ArgumentTypeError("at least one output format is required")
    return formats


def load_rows(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("history summary CSV is empty: {}".format(path))

    required = {"variant", "graph_size", "num_vehicles", "step", "instances"}
    for metric, _ in METRICS:
        required.add("mean_{}".format(metric))
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(
            "history summary CSV is missing columns: {}".format(", ".join(missing))
        )

    grouped = {}
    for row in rows:
        variant = row["variant"]
        item = {
            "graph_size": int(row["graph_size"]),
            "num_vehicles": int(row["num_vehicles"]),
            "step": int(row["step"]),
            "instances": int(row["instances"]),
        }
        for metric, _ in METRICS:
            item["mean_{}".format(metric)] = float(row["mean_{}".format(metric)])
        grouped.setdefault(variant, []).append(item)

    for variant, values in grouped.items():
        values.sort(key=lambda item: item["step"])
        steps = [item["step"] for item in values]
        if len(steps) != len(set(steps)):
            raise ValueError("variant {} contains duplicate search steps".format(variant))
    return grouped


def variant_sort_key(item):
    _, rows = item
    return rows[0]["num_vehicles"]


def draw_metric(axis, grouped, metric, ylabel):
    sorted_groups = sorted(grouped.items(), key=variant_sort_key)
    axis.set_facecolor("#FAFAFA")
    for style, (variant, rows) in zip(SERIES_STYLES, sorted_groups):
        steps = [item["step"] for item in rows]
        means = [item["mean_{}".format(metric)] for item in rows]
        label = "$K={}$".format(rows[0]["num_vehicles"])
        markevery = max(1, len(steps) // 6)
        axis.plot(
            steps,
            means,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=3.8,
            markevery=markevery,
            linewidth=1.9,
            label=label,
        )
    axis.set_xlabel("Search step", fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        title="Fleet size",
        frameon=False,
        loc="upper right",
        fontsize=14,
        title_fontsize=14,
    )

    inset = axis.inset_axes([0.23, 0.38, 0.54, 0.47])
    inset.set_facecolor("#FFFFFF")
    inset_steps = []
    for style, (variant, rows) in zip(SERIES_STYLES, sorted_groups):
        post_initial = [item for item in rows if item["step"] > 0]
        if not post_initial:
            continue
        baseline = post_initial[0]["mean_{}".format(metric)]
        if baseline == 0:
            continue
        steps = [item["step"] for item in post_initial]
        inset_steps.extend(steps)
        improvement = [
            100.0 * (baseline - item["mean_{}".format(metric)]) / abs(baseline)
            for item in post_initial
        ]
        inset.plot(
            steps,
            improvement,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=2.8,
            markevery=max(1, len(steps) // 5),
            linewidth=1.35,
        )
    if not inset_steps:
        raise ValueError("convergence data must contain at least one step after step 0")
    inset.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.7)
    zoom_start, zoom_end = min(inset_steps), max(inset_steps)
    inset.set_xlim(zoom_start, zoom_end)
    inset.set_xticks([zoom_start, (zoom_start + zoom_end) // 2, zoom_end])
    inset.set_title(
        "Improvement after step {}".format(zoom_start), fontsize=14, pad=3
    )
    inset.set_ylabel("Improvement (%)", fontsize=14)
    inset.set_xlabel("Search step", fontsize=14, labelpad=1)
    inset.grid(True, color="#E0E0E0", linestyle=":", linewidth=0.6)
    inset.tick_params(axis="both", labelsize=14, length=3)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.7)


def add_caption(axis, label, caption):
    axis.text(
        0.5,
        -0.23,
        "({}) {}".format(label, caption),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=16,
    )


def save_figure(figure, output_dir, stem, formats, dpi):
    outputs = []
    for extension in formats:
        output = output_dir / "{}.{}".format(stem, extension)
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = dpi
        figure.savefig(str(output), **kwargs)
        outputs.append(output)
    return outputs


def plot(grouped, output_dir, formats, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    figure, axes = plt.subplots(1, 3, figsize=(20.0, 6.5), constrained_layout=True)
    for label, axis, (metric, ylabel) in zip("abc", axes, METRICS):
        draw_metric(axis, grouped, metric, ylabel)
        add_caption(axis, label, ylabel)
    outputs.extend(save_figure(figure, output_dir, "search_convergence", formats, dpi))
    plt.close(figure)

    for label, (metric, ylabel) in zip("abc", METRICS):
        figure, axis = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)
        draw_metric(axis, grouped, metric, ylabel)
        add_caption(axis, label, ylabel)
        outputs.extend(
            save_figure(
                figure,
                output_dir,
                "convergence_{}_vs_step".format(metric),
                formats,
                dpi,
            )
        )
        plt.close(figure)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="history_summary.csv")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: INPUT parent/figures)"
    )
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not args.input.is_file():
        parser.error("input CSV does not exist: {}".format(args.input))
    output_dir = args.output_dir or args.input.parent / "figures"
    try:
        outputs = plot(load_rows(args.input), output_dir, args.formats, args.dpi)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
