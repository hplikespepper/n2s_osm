#!/usr/bin/env python3
"""
OR-Tools baseline for MVPDTSP.
Outputs results JSON compatible with N2S MVPDTSP format.
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

from problems.problem_mvpdtsp import MVPDTSP, MVPDPDataset


def build_distance_matrix(coords: np.ndarray, scale: float) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return np.rint(dist * scale).astype(np.int64)


def solve_instance(
    coords: np.ndarray,
    num_vehicles: int,
    time_limit: int,
    objective: str,
    scale: float,
    log_search: bool,
    first_solution_only: bool,
) -> Tuple[List[int], List[List[int]]]:
    num_nodes = coords.shape[0]
    starts = list(range(num_vehicles))
    ends = list(range(num_vehicles))

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

    if objective in {"distance+makespan", "makespan"}:
        distance_dim.SetGlobalSpanCostCoefficient(1)

    num_pairs = (num_nodes - num_vehicles) // 2
    pickup_start = num_vehicles
    delivery_start = num_vehicles + num_pairs

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

    vehicle_routes: List[List[int]] = []
    for v in range(num_vehicles):
        index = routing.Start(v)
        route: List[int] = []
        while True:
            node = manager.IndexToNode(index)
            route.append(node)
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))
        vehicle_routes.append(route)

    rec = [0 for _ in range(num_nodes)]
    visited = set()
    for route in vehicle_routes:
        if len(route) == 1:
            rec[route[0]] = route[0]
            visited.add(route[0])
            continue
        for i in range(len(route) - 1):
            rec[route[i]] = route[i + 1]
            visited.add(route[i])
        visited.add(route[-1])

    for i in range(num_nodes):
        if i not in visited:
            rec[i] = i

    return rec, vehicle_routes


def compute_costs(problem: MVPDTSP, coords: np.ndarray, rec: List[int]) -> Tuple[float, float, List[float]]:
    rec_tensor = torch.as_tensor(rec, dtype=torch.long).unsqueeze(0)
    coords_tensor = torch.as_tensor(coords, dtype=torch.float).unsqueeze(0)
    batch = {"coordinates": coords_tensor}
    distance, makespan = problem.compute_cost_components(batch, rec_tensor)
    vehicle_costs = problem._get_route_lengths(batch, rec_tensor)
    return (
        float(distance.item()),
        float(makespan.item()),
        vehicle_costs[0].cpu().tolist(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OR-Tools baseline for MVPDTSP")
    parser.add_argument("--val_dataset", type=str, default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1500)
    parser.add_argument("--time_limit", type=int, default=30, help="Seconds per instance")
    parser.add_argument(
        "--objective",
        type=str,
        default="distance+makespan",
        choices=["distance", "makespan", "distance+makespan"],
    )
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
    dataset = MVPDPDataset(
        filename=args.val_dataset,
        size=args.graph_size,
        num_samples=args.val_size,
        num_vehicles=args.num_vehicles,
    )

    print("[Stage] Initializing problem...")
    problem = MVPDTSP(
        p_size=args.graph_size,
        num_vehicles=args.num_vehicles,
        with_assert=False,
        use_makespan=(args.objective != "distance"),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "problem": "mvpdtsp",
        "graph_size": args.graph_size,
        "T_max": args.T_max,
        "val_size": min(args.val_size, len(dataset)),
        "instances": [],
        "num_vehicles": args.num_vehicles,
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

        rec, vehicle_routes = solve_instance(
            coords=coords,
            num_vehicles=args.num_vehicles,
            time_limit=args.time_limit,
            objective=args.objective,
            scale=args.scale,
            log_search=args.log_search,
            first_solution_only=args.first_solution_only,
        )

        if tqdm is None:
            print(f"[Stage] Computing costs for instance {i + 1}/{total_instances}")

        distance, makespan, vehicle_costs = compute_costs(problem, coords, rec)
        if args.objective == "distance":
            best_cost = distance
        elif args.objective == "makespan":
            best_cost = makespan
        else:
            best_cost = distance + makespan

        instance_data = {
            "instance_id": i,
            "best_cost": best_cost,
            "best_distance_cost": distance,
            "best_makespan_cost": makespan,
            "best_vehicle_distance_costs": vehicle_costs,
            "best_vehicle_completion_times": vehicle_costs,
            "best_rec": rec,
            "vehicle_routes": vehicle_routes,
            "route_lengths": [len(r) for r in vehicle_routes],
            "total_nodes": len(coords),
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
        output_path = os.path.join(results_dir, f"mvpdtsp_results_ortools_{timestamp}.json")

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
