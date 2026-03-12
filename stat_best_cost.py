#!/usr/bin/env python3
import argparse
import json
import math
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("No values provided")
    return sum(values) / len(values)


def std(values: Iterable[float], avg: float) -> float:
    values = list(values)
    if not values:
        raise ValueError("No values provided")
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mean and standard deviation of best_cost values"
    )
    parser.add_argument(
        "--json_path",
        nargs="?",
        default="./results/mvpdtsp_results_ortools_20260208_231845.json",
        help="Path to results JSON file",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data.get("instances", [])
    best_costs = [item["best_cost"] for item in instances if "best_cost" in item]

    if not best_costs:
        raise ValueError("No best_cost values found in instances")

    avg = mean(best_costs)
    sd = std(best_costs, avg)

    print(f"count: {len(best_costs)}")
    print(f"mean_best_cost: {avg}")
    print(f"std_best_cost: {sd}")


if __name__ == "__main__":
    main()
