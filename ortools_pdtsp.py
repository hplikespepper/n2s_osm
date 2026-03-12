#!/usr/bin/env python3
"""
OR-Tools baseline for PDTSP.
Outputs results JSON compatible with N2S PDTSP format.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "OR-Tools is required. Install with: pip install ortools"
    ) from exc

from problems.problem_pdtsp import PDTSP, PDPDataset


def build_distance_matrix(coords: np.ndarray, scale: float) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return np.rint(dist * scale).astype(np.int64)


def solve_instance(
    coords: np.ndarray,
    time_limit: int,
    scale: float,
    log_search: bool,
    first_solution_only: bool,
) -> List[int]:
    num_nodes = coords.shape[0]
    num_vehicles = 1
    starts = [0]
    ends = [0]

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    dist_mat = build_distance_matrix(coords, scale)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_mat[from_node, to_node])

    transit_callback = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

    max_distance = int(dist_mat.max() * num_nodes * 4 + 1)
    routing.AddDimension(transit_callback, 0, max_distance, True, "Distance")
    distance_dim = routing.GetDimensionOrDie("Distance")

    num_pairs = (num_nodes - 1) // 2
    pickup_start = 1
    delivery_start = 1 + num_pairs

    for p in range(num_pairs):
        pickup = pickup_start + p
        delivery = delivery_start + p
        pickup_index = manager.NodeToIndex(pickup)
        delivery_index = manager.NodeToIndex(delivery)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        routing.solver().Add(distance_dim.CumulVar(pickup_index) <= distance_dim.CumulVar(delivery_index))

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if not first_solution_only:
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(time_limit)
    search_params.log_search = log_search

    start_solve = time.perf_counter()
    solution = routing.SolveWithParameters(search_params)
    elapsed = time.perf_counter() - start_solve
    if solution is None:
        raise RuntimeError(f"OR-Tools failed to find a solution (elapsed {elapsed:.2f}s).")

    route: List[int] = []
    index = routing.Start(0)
    while True:
        node = manager.IndexToNode(index)
        route.append(node)
        if routing.IsEnd(index):
            break
        index = solution.Value(routing.NextVar(index))

    rec = [0 for _ in range(num_nodes)]
    for i in range(len(route) - 1):
        rec[route[i]] = route[i + 1]

    return rec


def compute_cost(problem: PDTSP, coords: np.ndarray, rec: List[int]) -> float:
    rec_tensor = torch.as_tensor(rec, dtype=torch.long).unsqueeze(0)
    coords_tensor = torch.as_tensor(coords, dtype=torch.float).unsqueeze(0)
    batch = {"coordinates": coords_tensor}
    cost = problem.get_costs(batch, rec_tensor)
    return float(cost.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OR-Tools baseline for PDTSP")
    parser.add_argument("--val_dataset", type=str, default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--T_max", type=int, default=1500)
    parser.add_argument("--time_limit", type=int, default=30, help="Seconds per instance")
    parser.add_argument("--log_search", action="store_true", help="Enable OR-Tools search logging")
    parser.add_argument(
        "--first_solution_only",
        action="store_true",
        help="Skip local search and return the first solution found",
    )
    parser.add_argument("--scale", type=float, default=1e6, help="Distance scale for OR-Tools")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[Stage] Loading dataset...")
    dataset = PDPDataset(
        filename=args.val_dataset,
        size=args.graph_size,
        num_samples=args.val_size,
    )

    print("[Stage] Initializing problem...")
    problem = PDTSP(
        p_size=args.graph_size,
        with_assert=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "problem": "pdtsp",
        "graph_size": args.graph_size,
        "T_max": args.T_max,
        "val_size": min(args.val_size, len(dataset)),
        "instances": [],
    }

    total_instances = results_data["val_size"]
    print(
        f"[Stage] Solving {total_instances} instances with OR-Tools... "
        f"(time_limit={args.time_limit}s, first_solution_only={args.first_solution_only})"
    )

    if tqdm is not None:
        iterator = tqdm(range(total_instances), desc="Solving", unit="inst")
    else:
        iterator = range(total_instances)

    for i in iterator:
        if tqdm is None:
            print(f"[Stage] Solving instance {i + 1}/{total_instances}")
        instance = dataset[i]
        coords = instance["coordinates"].cpu().numpy()

        rec = solve_instance(
            coords=coords,
            time_limit=args.time_limit,
            scale=args.scale,
            log_search=args.log_search,
            first_solution_only=args.first_solution_only,
        )

        if tqdm is None:
            print(f"[Stage] Computing costs for instance {i + 1}/{total_instances}")

        cost = compute_cost(problem, coords, rec)

        instance_data = {
            "instance_id": i,
            "best_cost": cost,
            "best_path": rec,
            "path_length": len(rec),
            "coordinates": coords.tolist(),
        }
        results_data["instances"].append(instance_data)

        if tqdm is None and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{total_instances} instances")

    results_dir = os.path.dirname(args.output) if args.output else "results"
    os.makedirs(results_dir, exist_ok=True)

    print("[Stage] Writing results...")
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(results_dir, f"pdtsp_results_ortools_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
