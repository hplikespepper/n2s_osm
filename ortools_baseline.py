#!/usr/bin/env python3
"""
OR-Tools baseline for MVPDTSP.
Outputs results JSON compatible with N2S MVPDTSP format.
The combined objective is total distance + (num_vehicles - 1) * makespan.
Use --num_workers for parallel solving across instances. Timing fields are seconds:
Use --pair_relocate full|light to select the pickup-delivery relocation neighborhood.
solve_time includes model construction, search and route extraction per instance;
solve_wall_time includes worker startup/shutdown and cost reporting, but excludes
dataset loading and JSON writing.
"""

import argparse
import json
import multiprocessing as mp
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
    from ortools.util import optional_boolean_pb2
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "OR-Tools is required. Install with: pip install ortools"
    ) from exc

from problems.problem_mvpdtsp import MVPDTSP, MVPDPDataset
from baseline_summary import summarize_result


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
    pair_relocate: str = "full",
) -> Tuple[List[int], List[List[int]]]:
    if objective not in {"distance", "distance+makespan"}:
        raise ValueError(f"Unsupported objective: {objective}")
    if pair_relocate not in {"full", "light"}:
        raise ValueError(f"Unsupported pair relocation mode: {pair_relocate}")
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

    if objective == "distance+makespan":
        # All routes start at distance zero, so the global span is makespan.
        # Match MVPDTSP.makespan_weight used by N2S.
        distance_dim.SetGlobalSpanCostCoefficient(num_vehicles - 1)

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
    # OR-Tools skips the light operator when the full operator is enabled.
    operators = search_params.local_search_operators
    operators.use_relocate_pair = (
        optional_boolean_pb2.BOOL_TRUE if pair_relocate == "full"
        else optional_boolean_pb2.BOOL_FALSE
    )
    operators.use_light_relocate_pair = optional_boolean_pb2.BOOL_TRUE
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if first_solution_only:
        search_params.solution_limit = 1
    else:
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


def _init_worker():
    # Avoid nested Torch thread pools when running multiple CPU processes.
    torch.set_num_threads(1)


def _solve_task(task):
    i, coords, args = task
    start = time.perf_counter()
    try:
        rec, vehicle_routes = solve_instance(
            coords, args.num_vehicles, args.time_limit, args.objective,
            args.scale, args.log_search, args.first_solution_only,
            args.pair_relocate,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Instance {i}: {exc}") from exc
    elapsed = time.perf_counter() - start
    problem = MVPDTSP(
        args.graph_size,
        num_vehicles=args.num_vehicles,
        use_makespan=(args.objective == "distance+makespan"),
    )
    distance, makespan, vehicle_costs = compute_costs(problem, coords, rec)
    return {
        "instance_id": i,
        "worker_pid": os.getpid(),
        "solve_time": elapsed,
        "best_cost": distance + problem.makespan_weight * makespan,
        "best_distance_cost": distance,
        "best_makespan_cost": makespan,
        "makespan_weight": problem.makespan_weight,
        "best_vehicle_distance_costs": vehicle_costs,
        "best_vehicle_completion_times": vehicle_costs,
        "best_rec": rec,
        "vehicle_routes": vehicle_routes,
        "route_lengths": [len(r) for r in vehicle_routes],
        "total_nodes": len(coords),
        "coordinates": coords.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OR-Tools baseline for MVPDTSP")
    parser.add_argument("--val_dataset", type=str, default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1500, help="Metadata only; does not limit search")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="CPU worker processes across instances (1=serial, 0=available CPUs)")
    parser.add_argument("--time_limit", type=int, default=30, help="Seconds per instance")
    parser.add_argument(
        "--pair_relocate", choices=["full", "light"], default="full",
        help="Pickup-delivery relocation neighborhood (default: full)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="distance+makespan",
        choices=["distance", "distance+makespan"],
        help="distance, or distance + (num_vehicles - 1) * makespan (N2S objective)",
    )
    parser.add_argument("--log_search", action="store_true", help="Enable OR-Tools search logging")
    parser.add_argument(
        "--first_solution_only",
        action="store_true",
        help="Skip local search and return the first solution found",
    )
    parser.add_argument("--scale", type=float, default=1e6, help="Distance scale for OR-Tools")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (optional)")
    args = parser.parse_args()
    if args.graph_size <= 0 or args.graph_size % 2:
        parser.error("--graph_size must be positive and even")
    if args.num_vehicles < 1 or args.val_size < 1 or args.time_limit < 1:
        parser.error("--num_vehicles, --val_size and --time_limit must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    return args


def main() -> None:
    args = parse_args()

    print("[Stage] Loading dataset...")
    dataset = MVPDPDataset(
        filename=args.val_dataset,
        size=args.graph_size,
        num_samples=args.val_size,
        num_vehicles=args.num_vehicles,
    )

    if not len(dataset):
        raise ValueError("Validation dataset is empty")
    expected_nodes = args.graph_size + args.num_vehicles
    if any(item["coordinates"].shape != (expected_nodes, 2) for item in dataset):
        raise ValueError("Dataset node count does not match --graph_size")
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    num_workers = min(args.num_workers or available_cpus, len(dataset))
    _init_worker()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "problem": "mvpdtsp",
        "method": "ortools",
        "graph_size": args.graph_size,
        "T_max": args.T_max,
        "val_size": min(args.val_size, len(dataset)),
        "instances": [],
        "num_vehicles": args.num_vehicles,
        "objective": args.objective,
        "makespan_weight": args.num_vehicles - 1 if args.objective == "distance+makespan" else 0,
        "num_workers": num_workers,
        "time_limit": args.time_limit,
        "first_solution_only": args.first_solution_only,
        "pair_relocate": args.pair_relocate,
        "scale": args.scale,
    }

    total_instances = results_data["val_size"]
    print(
        f"[Stage] Solving {total_instances} instances with OR-Tools... "
        f"(time_limit={args.time_limit}s, first_solution_only={args.first_solution_only}, "
        f"pair_relocate={args.pair_relocate})"
    )

    print(f"[Stage] Using {num_workers} CPU worker(s), one Torch thread per worker")
    tasks = ((i, dataset[i]["coordinates"].cpu().numpy(), args) for i in range(total_instances))
    wall_start = time.perf_counter()

    def collect(iterator):
        if tqdm is not None:
            iterator = tqdm(iterator, total=total_instances, desc="Solving", unit="inst")
        for result in iterator:
            results_data["instances"].append(result)
            if tqdm is None:
                print(f"[Stage] Completed {len(results_data['instances'])}/{total_instances} instances")

    if num_workers == 1:
        collect(map(_solve_task, tasks))
    else:
        with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker) as pool:
            collect(pool.imap_unordered(_solve_task, tasks, chunksize=1))
    results_data["solve_wall_time"] = time.perf_counter() - wall_start
    results_data["instances"].sort(key=lambda result: result["instance_id"])
    print(f"[Stage] Solve wall time: {results_data['solve_wall_time']:.3f}s")

    results_dir = (os.path.dirname(args.output) or ".") if args.output else "results"
    os.makedirs(results_dir, exist_ok=True)

    print("[Stage] Writing results...")
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(results_dir, f"mvpdtsp_results_ortools_{timestamp}.json")

    results_data["summary"] = summarize_result(results_data)
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
