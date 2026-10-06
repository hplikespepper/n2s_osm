"""Summarize baseline results without rerunning solvers.

Usage: python baseline_summary.py result_mc/run_... [result_ortools/run_...]
All durations are seconds; standard deviations are population values (ddof=0).
Existing per-instance JSON files are read only.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def summarize_result(result):
    instances = result["instances"]
    if not instances:
        raise ValueError("Cannot summarize an empty result")
    count = len(instances)
    if result.get("val_size", count) != count:
        raise ValueError("Instance count does not match val_size")
    summary = {"num_instances": count}
    for label, field in (
        ("cost", "best_cost"),
        ("distance", "best_distance_cost"),
        ("makespan", "best_makespan_cost"),
    ):
        values = [instance[field] for instance in instances]
        summary.update({
            "mean_" + label: mean(values),
            "std_" + label: pstdev(values),
            "min_" + label: min(values),
            "max_" + label: max(values),
        })
    summary["mean_instance_solve_time_s"] = mean(x["solve_time"] for x in instances)
    summary["solve_wall_time_s"] = result["solve_wall_time"]
    summary["wall_time_per_instance_s"] = result["solve_wall_time"] / count
    return summary


def summarize_directory(directory):
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "experiment_summary.json":
            continue
        with path.open() as f:
            result = json.load(f)
        if not isinstance(result, dict) or "instances" not in result:
            continue
        row = {key: result.get(key) for key in (
            "method", "graph_size", "num_vehicles", "objective", "makespan_weight",
            "num_workers", "num_mc_samples", "seed", "time_limit", "first_solution_only",
            "pair_relocate", "lkh_version", "num_starts", "start_workers",
            "time_limit_per_start", "scale",
            "max_trials", "max_total_trials",
        )}
        row.update(summarize_result(result))
        row["result_file"] = path.name
        rows.append(row)
    if not rows:
        raise ValueError(f"No baseline results found in {directory}")
    rows.sort(key=lambda row: (row["graph_size"], row["num_vehicles"], row["result_file"]))
    json_path = directory / "experiment_summary.json"
    csv_path = directory / "experiment_summary.csv"
    # Replace complete summaries so repeated backfills never duplicate rows.
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with tmp_json.open("w") as f:
        json.dump(rows, f, indent=2)
    with tmp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp_json.replace(json_path)
    tmp_csv.replace(csv_path)
    print(f"Saved {len(rows)} experiment summaries: {csv_path} and {json_path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        summarize_directory(directory)
