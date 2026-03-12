#!/usr/bin/env python3
"""Analyze MVPDTSP results JSON files."""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze MVPDTSP results JSON")
    parser.add_argument("json_path", type=str, help="Path to results JSON file")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    instances = data["instances"]
    n = len(instances)

    costs = np.array([inst["best_cost"] for inst in instances])
    dist_costs = np.array([inst["best_distance_cost"] for inst in instances])
    makespan_costs = np.array([inst["best_makespan_cost"] for inst in instances])

    print(f"File: {args.json_path}")
    print(f"Method: {data.get('method', 'N/A')}")
    print(f"Instances: {n}")
    print(f"Graph size: {data.get('graph_size', 'N/A')}")
    print(f"Num vehicles: {data.get('num_vehicles', 'N/A')}")
    print()
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    for name, vals in [
        ("best_cost", costs),
        ("best_distance_cost", dist_costs),
        ("best_makespan_cost", makespan_costs),
    ]:
        print(f"{name:<25} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.min():>10.4f} {vals.max():>10.4f}")

    # per-vehicle stats if available
    if "best_vehicle_distance_costs" in instances[0]:
        num_v = len(instances[0]["best_vehicle_distance_costs"])
        for v in range(num_v):
            v_costs = np.array([inst["best_vehicle_distance_costs"][v] for inst in instances])
            print(f"  vehicle_{v}_distance     {v_costs.mean():>10.4f} {v_costs.std():>10.4f} {v_costs.min():>10.4f} {v_costs.max():>10.4f}")

    # solve time if available
    if "solve_time" in instances[0]:
        times = np.array([inst["solve_time"] for inst in instances])
        print(f"\nSolve time (s):  mean={times.mean():.2f}  total={times.sum():.1f}")


if __name__ == "__main__":
    main()
