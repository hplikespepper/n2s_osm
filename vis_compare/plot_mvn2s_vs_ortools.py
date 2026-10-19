#!/usr/bin/env python3
"""Create a 2x3 route comparison for MvN2S and OR-Tools full.

For each problem size, the script matches records by ``instance_id``, verifies
that both solvers used identical coordinates, and selects the instance with
the largest absolute objective improvement from MvN2S.  OR-Tools solutions
are placed in the top row and the matching MvN2S solutions in the bottom row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result_vis_compare"


@dataclass(frozen=True)
class Comparison:
    label: str
    graph_size: int
    num_vehicles: int
    mvn2s_path: Path
    ortools_path: Path
    min_ortools_vehicles: int = 0


COMPARISONS = (
    Comparison(
        "20 nodes / 10 pairs / 2 vehicles",
        20,
        2,
        PROJECT_ROOT
        / "n2s_mv/result_n2s/run_20261005_202445_6h2JWG/n2s_20_mv2/"
        "mvpdtsp_results_20261005_202624.json",
        PROJECT_ROOT
        / "n2s_mv/result_ortools_full/run_20261006_094739_QEMUhn/"
        "ortools_20_mv2_t15.json",
        2,
    ),
    Comparison(
        "50 nodes / 25 pairs / 5 vehicles",
        50,
        5,
        PROJECT_ROOT
        / "n2s_mv/result_n2s/run_20260917_155458_ql1nI7/n2s_50_mv5/"
        "mvpdtsp_results_20260917_155947.json",
        PROJECT_ROOT
        / "n2s_mv/result_ortools_full/run_20260928_194414_lHjWzt/"
        "ortools_50_mv5_t240.json",
    ),
    Comparison(
        "100 nodes / 50 pairs / 5 vehicles",
        100,
        5,
        PROJECT_ROOT
        / "n2s_mv/result_n2s/run_20260917_155458_ql1nI7/n2s_100_mv5/"
        "mvpdtsp_results_20260917_160341.json",
        PROJECT_ROOT
        / "n2s_mv/result_ortools_full/run_20260925_152619_Z69T4t/"
        "ortools_100_mv5.json",
        4,
    ),
)

VEHICLE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def load_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def index_instances(result: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    records = result.get("instances")
    if not isinstance(records, list):
        raise ValueError(f"Missing 'instances' list in {path}")
    indexed = {int(item["instance_id"]): item for item in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate instance_id found in {path}")
    return indexed


def select_instance(spec: Comparison) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mvn2s_result = load_results(spec.mvn2s_path)
    ortools_result = load_results(spec.ortools_path)
    mvn2s = index_instances(mvn2s_result, spec.mvn2s_path)
    ortools = index_instances(ortools_result, spec.ortools_path)
    common_ids = sorted(mvn2s.keys() & ortools.keys())
    if not common_ids:
        raise ValueError(f"No matching instance_id for {spec.label}")

    candidates: list[tuple[float, int]] = []
    total_mvn2s_wins = 0
    for instance_id in common_ids:
        mv = mvn2s[instance_id]
        ot = ortools[instance_id]
        mv_coords = np.asarray(mv["coordinates"], dtype=float)
        ot_coords = np.asarray(ot["coordinates"], dtype=float)
        if mv_coords.shape != ot_coords.shape or not np.allclose(
            mv_coords, ot_coords, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                f"Coordinate mismatch for {spec.label}, instance {instance_id}"
            )
        improvement = float(ot["best_cost"]) - float(mv["best_cost"])
        if np.isfinite(improvement) and improvement > 1e-9:
            total_mvn2s_wins += 1
        active_ortools_vehicles = sum(
            any(int(node) >= spec.num_vehicles for node in route)
            for route in ot["vehicle_routes"]
        )
        meets_vehicle_requirement = (
            active_ortools_vehicles >= spec.min_ortools_vehicles
        )
        if np.isfinite(improvement) and improvement > 1e-9 and meets_vehicle_requirement:
            candidates.append((improvement, instance_id))

    if not candidates:
        raise ValueError(f"MvN2S is not cheaper on any matched instance for {spec.label}")

    improvement, selected_id = max(candidates, key=lambda item: (item[0], -item[1]))
    mv = mvn2s[selected_id]
    ot = ortools[selected_id]
    metadata = {
        "configuration": spec.label,
        "graph_size": spec.graph_size,
        "num_vehicles": spec.num_vehicles,
        "matched_instances": len(common_ids),
        "mvn2s_wins": total_mvn2s_wins,
        "eligible_candidates": len(candidates),
        "selection_rule": "largest absolute best_cost improvement",
        "selection_constraint": (
            f"OR-Tools must use at least {spec.min_ortools_vehicles} vehicles"
            if spec.min_ortools_vehicles
            else "none"
        ),
        "ortools_active_vehicles": sum(
            any(int(node) >= spec.num_vehicles for node in route)
            for route in ot["vehicle_routes"]
        ),
        "instance_id": selected_id,
        "ortools_cost": float(ot["best_cost"]),
        "mvn2s_cost": float(mv["best_cost"]),
        "absolute_improvement": improvement,
        "relative_improvement_percent": 100.0 * improvement / float(ot["best_cost"]),
        "mvn2s_source": str(spec.mvn2s_path.relative_to(PROJECT_ROOT)),
        "ortools_source": str(spec.ortools_path.relative_to(PROJECT_ROOT)),
    }
    return ot, mv, metadata


def validate_solution(instance: dict[str, Any], num_vehicles: int) -> None:
    coordinates = instance["coordinates"]
    routes = instance["vehicle_routes"]
    if len(routes) != num_vehicles:
        raise ValueError(
            f"Instance {instance['instance_id']} has {len(routes)} routes, "
            f"expected {num_vehicles}"
        )
    node_count = len(coordinates)
    for vehicle, route in enumerate(routes):
        if not route:
            raise ValueError(f"Vehicle {vehicle} has an empty route")
        if any(int(node) < 0 or int(node) >= node_count for node in route):
            raise ValueError(f"Vehicle {vehicle} contains an invalid node index")


def plot_solution(
    ax: plt.Axes,
    instance: dict[str, Any],
    graph_size: int,
    num_vehicles: int,
    improvement_pct: float | None = None,
) -> None:
    validate_solution(instance, num_vehicles)
    coords = np.asarray(instance["coordinates"], dtype=float)
    pickup_start = num_vehicles
    delivery_start = num_vehicles + graph_size // 2

    # Pickup-delivery pairing is identical in both rows and kept deliberately
    # faint so that the colored vehicle routes remain the visual focus.
    for pickup in range(pickup_start, delivery_start):
        delivery = pickup + graph_size // 2
        ax.plot(
            coords[[pickup, delivery], 0],
            coords[[pickup, delivery], 1],
            color="#8A8A8A",
            linestyle=(0, (2, 3)),
            linewidth=0.55,
            alpha=0.22,
            zorder=1,
        )

    for vehicle, saved_route in enumerate(instance["vehicle_routes"]):
        route = [int(node) for node in saved_route]
        if route[-1] != route[0]:
            route.append(route[0])
        route_coords = coords[route]
        color = VEHICLE_COLORS[vehicle % len(VEHICLE_COLORS)]
        ax.plot(
            route_coords[:, 0],
            route_coords[:, 1],
            color=color,
            linewidth=1.65 if graph_size < 100 else 1.3,
            alpha=0.88,
            zorder=2,
        )
        starts = route_coords[:-1]
        deltas = route_coords[1:] - starts
        nonzero = np.linalg.norm(deltas, axis=1) > 1e-12
        arrow_starts = starts[nonzero] + 0.35 * deltas[nonzero]
        arrow_deltas = 0.30 * deltas[nonzero]
        ax.quiver(
            arrow_starts[:, 0],
            arrow_starts[:, 1],
            arrow_deltas[:, 0],
            arrow_deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0025,
            headwidth=4.0,
            headlength=5.0,
            headaxislength=4.3,
            color=color,
            alpha=0.9,
            zorder=3,
        )

    node_size = 42 if graph_size == 20 else (31 if graph_size == 50 else 22)
    pickups = coords[pickup_start:delivery_start]
    deliveries = coords[delivery_start : delivery_start + graph_size // 2]
    ax.scatter(
        pickups[:, 0], pickups[:, 1], s=node_size, marker="o",
        facecolor="white", edgecolor="#1F1F1F", linewidth=0.75, zorder=5,
    )
    ax.scatter(
        deliveries[:, 0], deliveries[:, 1], s=node_size + 4, marker="^",
        facecolor="#4C78A8", edgecolor="#1F1F1F", linewidth=0.65, zorder=5,
    )
    depot = coords[0]
    ax.scatter(
        [depot[0]], [depot[1]], s=235, marker="*", facecolor="#F2C14E",
        edgecolor="black", linewidth=1.1, zorder=7,
    )

    if graph_size == 20:
        for node in range(pickup_start, len(coords)):
            ax.annotate(
                str(node), coords[node], xytext=(3, 3), textcoords="offset points",
                fontsize=9.5, color="#252525", zorder=8,
            )

    # Round the two objective components first, then reconstruct the displayed
    # objective so the printed values exactly satisfy J = D + (K - 1) M.
    distance = round(float(instance["best_distance_cost"]), 2)
    makespan = round(float(instance["best_makespan_cost"]), 2)
    cost = distance + (num_vehicles - 1) * makespan
    title = f"Obj. {cost:.2f}  |  Distance {distance:.2f}  |  Makespan {makespan:.2f}"
    ax.set_title(title, fontsize=14.0, pad=9, color="#167A3E" if improvement_pct else "#222222")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(-0.035, 1.035)
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.tick_params(labelsize=12, length=3.5, width=0.9)
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#B6B6B6")


def legend_handles(num_vehicles: int) -> list[Line2D]:
    handles = [
        Line2D([], [], marker="*", markersize=18, markerfacecolor="#F2C14E", markeredgecolor="black", linestyle="None", label="Depot"),
        Line2D([], [], marker="o", markersize=9, markerfacecolor="white", markeredgecolor="#1F1F1F", linestyle="None", label="Pickup"),
        Line2D([], [], marker="^", markersize=10, markerfacecolor="#4C78A8", markeredgecolor="#1F1F1F", linestyle="None", label="Delivery"),
        Line2D([], [], color="#8A8A8A", linestyle="--", linewidth=1, label="Pickup-delivery pair"),
    ]
    handles.extend(
        Line2D([], [], color=VEHICLE_COLORS[v], linewidth=2.2, label=f"Vehicle {v + 1}")
        for v in range(num_vehicles)
    )
    return handles


def create_figure(selected: list[tuple[Comparison, dict[str, Any], dict[str, Any], dict[str, Any]]]) -> plt.Figure:
    # A taller canvas lets the equal-aspect route maps fill each grid cell;
    # otherwise Matplotlib shrinks them vertically and leaves wide column gaps.
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 12.3), sharex=True, sharey=True)
    for column, (spec, ortools, mvn2s, metadata) in enumerate(selected):
        axes[0, column].text(
            0.5,
            1.15,
            spec.label,
            transform=axes[0, column].transAxes,
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )
        plot_solution(axes[0, column], ortools, spec.graph_size, spec.num_vehicles)
        plot_solution(
            axes[1, column],
            mvn2s,
            spec.graph_size,
            spec.num_vehicles,
            float(metadata["relative_improvement_percent"]),
        )

    axes[0, 0].set_ylabel("Y", fontsize=14)
    axes[1, 0].set_ylabel("Y", fontsize=14)
    for axis in axes[1, :]:
        axis.set_xlabel("X", fontsize=14, labelpad=7)

    max_vehicles = max(spec.num_vehicles for spec, *_ in selected)
    shared_legend_handles = legend_handles(max_vehicles)
    fig.legend(
        handles=shared_legend_handles,
        loc="lower center",
        ncol=len(shared_legend_handles),
        bbox_to_anchor=(0.5, 0.002),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=15,
        handlelength=2.1,
        handletextpad=0.55,
        columnspacing=1.0,
    )
    fig.text(0.5, 0.542, "OR-Tools", ha="center", va="center", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.062, "JRANS", ha="center", va="center", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.040, right=0.995, top=0.935, bottom=0.125, hspace=0.18, wspace=0.040)
    return fig


def save_individual_figures(
    selected: list[tuple[Comparison, dict[str, Any], dict[str, Any], dict[str, Any]]],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for spec, ortools, mvn2s, metadata in selected:
        slug = f"{spec.graph_size}_nodes_{spec.num_vehicles}_vehicles"
        for solver_slug, solver_name, instance, is_mvn2s in (
            ("ortools", "OR-Tools", ortools, False),
            ("mvn2s", "MvN2S", mvn2s, True),
        ):
            fig, ax = plt.subplots(figsize=(8.8, 9.0))
            plot_solution(
                ax,
                instance,
                spec.graph_size,
                spec.num_vehicles,
                float(metadata["relative_improvement_percent"]) if is_mvn2s else None,
            )
            ax.set_xlabel("X", fontsize=14, labelpad=7)
            ax.set_ylabel("Y", fontsize=14, labelpad=7)
            fig.suptitle(
                f"{spec.label} — {solver_name}",
                fontsize=18,
                fontweight="bold",
                y=0.975,
            )
            fig.legend(
                handles=legend_handles(spec.num_vehicles),
                loc="lower center",
                ncol=3,
                bbox_to_anchor=(0.5, 0.018),
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                edgecolor="black",
                fontsize=14,
                handlelength=2.0,
                handletextpad=0.65,
                columnspacing=1.15,
            )
            fig.subplots_adjust(left=0.105, right=0.98, top=0.89, bottom=0.20)
            output_path = individual_dir / f"{slug}_{solver_slug}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    manifest = []
    for spec in COMPARISONS:
        ortools, mvn2s, metadata = select_instance(spec)
        selected.append((spec, ortools, mvn2s, metadata))
        manifest.append(metadata)
        print(
            f"{spec.label}: instance {metadata['instance_id']}, "
            f"OR-Tools={metadata['ortools_cost']:.6f}, "
            f"MvN2S={metadata['mvn2s_cost']:.6f}, "
            f"improvement={metadata['relative_improvement_percent']:.2f}% "
            f"({metadata['mvn2s_wins']}/{metadata['matched_instances']} MvN2S wins; "
            f"{metadata['eligible_candidates']} meet selection constraints)"
        )

    figure = create_figure(selected)
    png_path = output_dir / "mvn2s_vs_ortools_full.png"
    pdf_path = output_dir / "mvn2s_vs_ortools_full.pdf"
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    individual_paths = save_individual_figures(selected, output_dir, args.dpi)

    manifest_path = output_dir / "selected_instances.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    for path in individual_paths:
        print(f"Saved {path}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
