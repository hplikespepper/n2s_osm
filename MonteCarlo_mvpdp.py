#!/usr/bin/env python3
"""
Monte Carlo baseline for MVPDTSP.
Randomly samples feasible solutions and keeps the best.
Outputs results JSON compatible with N2S MVPDTSP / OR-Tools format.
The default objective is D + (num_vehicles - 1) * M, matching N2S.
Use --num_workers N to solve instances in separate CPU processes. Seeds are
assigned per instance so serial and parallel runs sample the same candidates.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from problems.problem_mvpdtsp import MVPDTSP, MVPDPDataset
from baseline_summary import summarize_result


def _build_rec(num_vehicles: int, num_pairs: int, vehicle_routes: List[List[int]]) -> List[int]:
    """Convert vehicle routes to rec (next-node array)."""
    total_nodes = num_vehicles + 2 * num_pairs
    rec = list(range(total_nodes))
    for route in vehicle_routes:
        if len(route) <= 2:
            continue
        for i in range(len(route) - 1):
            rec[route[i]] = route[i + 1]
    return rec


def _random_interleaved_route(
    depot: int,
    pairs: List[int],
    pickup_start: int,
    delivery_start: int,
) -> List[int]:
    """Build a route with random interleaved pickup/delivery ordering.

    For each pair (in random order), insert pickup at a random position,
    then insert delivery at a random position after the pickup.
    This allows carpooling (carrying multiple items simultaneously).
    """
    body: List[int] = []
    shuffled = pairs[:]
    random.shuffle(shuffled)
    for p_idx in shuffled:
        pickup = pickup_start + p_idx
        delivery = delivery_start + p_idx
        # insert pickup at a random position
        p_pos = random.randint(0, len(body))
        body.insert(p_pos, pickup)
        # insert delivery at a random position AFTER pickup
        d_pos = random.randint(p_pos + 1, len(body))
        body.insert(d_pos, delivery)
    return [depot] + body + [depot]


def generate_random_solution(
    num_vehicles: int,
    num_pairs: int,
) -> Tuple[List[int], List[List[int]]]:
    """Generate a random feasible solution with interleaved P-D ordering.

    Randomly assigns each P-D pair to a vehicle, then builds a random
    interleaved route (allows carpooling: multiple pickups before deliveries).
    """
    pickup_start = num_vehicles
    delivery_start = num_vehicles + num_pairs

    # randomly assign each pair to a vehicle
    assignments = [random.randint(0, num_vehicles - 1) for _ in range(num_pairs)]

    vehicle_pairs: List[List[int]] = [[] for _ in range(num_vehicles)]
    for p_idx, v in enumerate(assignments):
        vehicle_pairs[v].append(p_idx)

    vehicle_routes: List[List[int]] = []
    for v in range(num_vehicles):
        route = _random_interleaved_route(
            v, vehicle_pairs[v], pickup_start, delivery_start
        )
        vehicle_routes.append(route)

    rec = _build_rec(num_vehicles, num_pairs, vehicle_routes)
    return rec, vehicle_routes


def generate_greedy_solution(
    coords: np.ndarray,
    num_vehicles: int,
    num_pairs: int,
    perturb_ratio: float = 0.3,
) -> Tuple[List[int], List[List[int]]]:
    """Greedy nearest-neighbor construction with random perturbation.

    1. Greedily assign each pair to the vehicle whose current position
       is nearest to the pickup point.
    2. Within each vehicle, order pairs by nearest-neighbor from depot.
    3. Randomly perturb: swap `perturb_ratio` fraction of pairs between
       vehicles, and shuffle a portion of the within-vehicle ordering.
    """
    pickup_start = num_vehicles
    delivery_start = num_vehicles + num_pairs

    # --- greedy assignment: assign each pair to nearest vehicle ---
    vehicle_pos = [coords[v].copy() for v in range(num_vehicles)]
    vehicle_pairs: List[List[int]] = [[] for _ in range(num_vehicles)]
    pair_order = list(range(num_pairs))
    random.shuffle(pair_order)  # randomize insertion order for diversity

    for p_idx in pair_order:
        pickup_coord = coords[pickup_start + p_idx]
        best_v = 0
        best_dist = float("inf")
        for v in range(num_vehicles):
            d = float(np.sum((vehicle_pos[v] - pickup_coord) ** 2))
            if d < best_dist:
                best_dist = d
                best_v = v
        vehicle_pairs[best_v].append(p_idx)
        # update vehicle position to delivery location
        vehicle_pos[best_v] = coords[delivery_start + p_idx].copy()

    # --- greedy ordering within each vehicle (nearest neighbor) ---
    for v in range(num_vehicles):
        if len(vehicle_pairs[v]) <= 1:
            continue
        remaining = vehicle_pairs[v][:]
        ordered = []
        current_pos = coords[v].copy()
        while remaining:
            best_idx = 0
            best_dist = float("inf")
            for j, p_idx in enumerate(remaining):
                d = float(np.sum((current_pos - coords[pickup_start + p_idx]) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            current_pos = coords[delivery_start + chosen].copy()
        vehicle_pairs[v] = ordered

    # --- random perturbation ---
    # swap some pairs between vehicles
    num_swaps = int(num_pairs * perturb_ratio) if num_vehicles > 1 else 0
    for _ in range(num_swaps):
        v1, v2 = random.sample(range(num_vehicles), 2)
        if vehicle_pairs[v1]:
            idx = random.randrange(len(vehicle_pairs[v1]))
            pair = vehicle_pairs[v1].pop(idx)
            pos = random.randint(0, len(vehicle_pairs[v2]))
            vehicle_pairs[v2].insert(pos, pair)

    # --- build interleaved routes (allows carpooling) ---
    vehicle_routes: List[List[int]] = []
    for v in range(num_vehicles):
        route = _random_interleaved_route(
            v, vehicle_pairs[v], pickup_start, delivery_start
        )
        vehicle_routes.append(route)

    rec = _build_rec(num_vehicles, num_pairs, vehicle_routes)
    return rec, vehicle_routes


def solve_instance(
    coords: np.ndarray,
    num_vehicles: int,
    num_samples: int,
    objective: str,
    greedy: bool = False,
    perturb_ratio: float = 0.3,
) -> Tuple[List[int], List[List[int]], float]:
    """Run Monte Carlo sampling for one instance."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    num_nodes = coords.shape[0]
    if num_vehicles < 1 or num_nodes <= num_vehicles or (num_nodes - num_vehicles) % 2:
        raise ValueError("Expected vehicle depots followed by pickup-delivery pairs")
    num_pairs = (num_nodes - num_vehicles) // 2

    problem = MVPDTSP(
        p_size=num_pairs * 2,
        num_vehicles=num_vehicles,
        with_assert=False,
        use_makespan=(objective != "distance"),
    )

    coords_tensor = torch.as_tensor(coords, dtype=torch.float).unsqueeze(0)
    batch = {"coordinates": coords_tensor}

    best_cost = float("inf")
    best_rec = None
    best_routes = None

    for _ in range(num_samples):
        if greedy:
            rec, routes = generate_greedy_solution(
                coords, num_vehicles, num_pairs, perturb_ratio
            )
        else:
            rec, routes = generate_random_solution(num_vehicles, num_pairs)
        rec_tensor = torch.as_tensor(rec, dtype=torch.long).unsqueeze(0)
        distance, makespan = problem.compute_cost_components(batch, rec_tensor)
        cost = objective_cost(distance.item(), makespan.item(), objective, num_vehicles)

        if cost < best_cost:
            best_cost = cost
            best_rec = rec
            best_routes = [r[:-1] for r in routes]  # remove trailing depot duplicate

    return best_rec, best_routes, best_cost


def compute_costs(
    problem: MVPDTSP, coords: np.ndarray, rec: List[int]
) -> Tuple[float, float, List[float]]:
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


def objective_cost(distance, makespan, objective, num_vehicles):
    """Use the same objective for candidate selection and result reporting."""
    if objective == "distance":
        return distance
    if objective == "makespan":
        return makespan
    if objective == "distance+makespan":
        return distance + (num_vehicles - 1) * makespan
    raise ValueError(f"Unknown objective: {objective}")


def _init_worker():
    # Each process handles one instance at a time; avoid nested CPU thread pools.
    torch.set_num_threads(1)


def _solve_task(task):
    i, coords, args = task
    instance_seed = (args.seed + i) % (2 ** 32)
    random.seed(instance_seed)
    np.random.seed(instance_seed)
    torch.manual_seed(instance_seed)
    t0 = time.perf_counter()
    rec, vehicle_routes, best_cost = solve_instance(
        coords, args.num_vehicles, args.num_mc_samples, args.objective,
        args.greedy, args.perturb_ratio,
    )
    elapsed = time.perf_counter() - t0
    problem = MVPDTSP(
        args.graph_size,
        num_vehicles=args.num_vehicles,
        use_makespan=(args.objective != "distance"),
    )
    distance, makespan, vehicle_costs = compute_costs(problem, coords, rec)
    assert best_cost == objective_cost(distance, makespan, args.objective, args.num_vehicles)
    return {
        "instance_id": i,
        "instance_seed": instance_seed,
        "worker_pid": os.getpid(),
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
        "solve_time": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo baseline for MVPDTSP")
    parser.add_argument("--val_dataset", type=str, default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1500, help="Metadata only; does not limit sampling")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="CPU worker processes across instances (1=serial, 0=available CPUs)")
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=10000,
        help="Number of random solutions to sample per instance",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="distance+makespan",
        choices=["distance", "makespan", "distance+makespan"],
        help="distance+makespan uses D + (num_vehicles - 1) * M, matching N2S",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy construction + random perturbation instead of pure random",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=0.3,
        help="Pair transfer attempts / number of pairs in greedy mode; routes remain randomized",
    )
    args = parser.parse_args()
    if args.graph_size <= 0 or args.graph_size % 2:
        parser.error("--graph_size must be positive and even")
    if args.num_vehicles < 1 or args.val_size < 1 or args.num_mc_samples < 1:
        parser.error("--num_vehicles, --val_size and --num_mc_samples must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if not 0 <= args.perturb_ratio <= 1:
        parser.error("--perturb_ratio must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    method = "monte_carlo_greedy" if args.greedy else "monte_carlo"
    results_data = {
        "timestamp": timestamp,
        "problem": "mvpdtsp",
        "method": method,
        "graph_size": args.graph_size,
        "T_max": args.T_max,
        "val_size": min(args.val_size, len(dataset)),
        "num_mc_samples": args.num_mc_samples,
        "instances": [],
        "num_vehicles": args.num_vehicles,
        "objective": args.objective,
        "makespan_weight": args.num_vehicles - 1 if args.objective == "distance+makespan" else (1 if args.objective == "makespan" else 0),
        "seed": args.seed,
        "num_workers": num_workers,
    }

    total_instances = results_data["val_size"]
    print(
        f"[Stage] Solving {total_instances} instances with Monte Carlo "
        f"({args.num_mc_samples} samples/instance)..."
    )

    print(f"[Stage] Using {num_workers} CPU worker(s), one Torch thread per worker")
    tasks = ((i, dataset[i]["coordinates"].cpu().numpy(), args) for i in range(total_instances))
    wall_start = time.perf_counter()

    def collect(iterator):
        if tqdm is not None:
            iterator = tqdm(iterator, total=total_instances, desc="Solving", unit="inst")
        for result in iterator:
            results_data["instances"].append(result)

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

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            results_dir, f"mvpdtsp_results_mc_{timestamp}.json"
        )

    print("[Stage] Writing results...")
    results_data["summary"] = summarize_result(results_data)
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # print summary
    costs = [inst["best_cost"] for inst in results_data["instances"]]
    label = "Monte Carlo + Greedy" if args.greedy else "Monte Carlo"
    print(f"\n{label} Results ({args.num_mc_samples} samples/instance):")
    print(f"  Mean cost: {np.mean(costs):.4f}")
    print(f"  Std cost:  {np.std(costs):.4f}")
    print(f"  Min cost:  {np.min(costs):.4f}")
    print(f"  Max cost:  {np.max(costs):.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
