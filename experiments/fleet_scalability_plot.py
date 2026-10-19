#!/usr/bin/env python3
"""Plot the Fleet Scalability summary produced by the evaluation script."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("distance", "Total distance $D$", "#0072B2"),
    ("makespan", "Makespan $M$", "#D55E00"),
    ("proposed", r"Proposed objective $\mathcal{J}$", "#009E73"),
    ("utilization", r"Active-vehicle ratio $r_{\mathrm{active}}$", "#CC79A7"),
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
        raise ValueError("summary CSV is empty: {}".format(path))

    required = {"graph_size", "num_vehicles", "instances"}
    for metric, _, _ in METRICS:
        required.add("mean_{}".format(metric))
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("summary CSV is missing columns: {}".format(", ".join(missing)))

    parsed = []
    for row in rows:
        item = {
            "graph_size": int(row["graph_size"]),
            "num_vehicles": int(row["num_vehicles"]),
            "instances": int(row["instances"]),
        }
        for metric, _, _ in METRICS:
            item["mean_{}".format(metric)] = float(row["mean_{}".format(metric)])
        parsed.append(item)
    parsed.sort(key=lambda item: item["num_vehicles"])

    vehicle_counts = [item["num_vehicles"] for item in parsed]
    if len(vehicle_counts) != len(set(vehicle_counts)):
        raise ValueError("summary CSV contains duplicate vehicle counts")
    return parsed


def style_axis(axis, vehicle_counts, ylabel):
    axis.set_facecolor("#FAFAFA")
    axis.set_xlabel("Number of vehicles $K$", fontsize=16)
    axis.set_ylabel(ylabel, fontsize=16)
    axis.set_xticks(vehicle_counts)
    axis.tick_params(axis="both", labelsize=16)
    axis.grid(True, color="#D9D9D9", linestyle="--", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_metric(axis, rows, metric, ylabel, color):
    vehicle_counts = [item["num_vehicles"] for item in rows]
    means = [item["mean_{}".format(metric)] for item in rows]
    axis.plot(
        vehicle_counts,
        means,
        color=color,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.5,
        markersize=5,
        linewidth=1.8,
    )
    style_axis(axis, vehicle_counts, ylabel)
    if metric == "utilization":
        axis.set_ylim(0.0, 1.05)


def add_caption(axis, label, caption):
    axis.text(
        0.5,
        -0.25,
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


def plot(rows, output_dir, formats, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)
    for label, axis, (metric, ylabel, color) in zip("abcd", axes.flat, METRICS):
        draw_metric(axis, rows, metric, ylabel, color)
        add_caption(axis, label, ylabel)
    outputs.extend(
        save_figure(figure, output_dir, "fleet_scalability", formats, dpi)
    )
    plt.close(figure)

    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Fleet summary.csv")
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
