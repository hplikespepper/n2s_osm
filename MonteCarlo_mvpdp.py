#!/usr/bin/env python3
"""
Monte Carlo baseline for MVPDTSP.
Randomly samples feasible solutions and keeps the best.
Outputs results JSON compatible with N2S MVPDTSP / OR-Tools format.
"""

import argparse
import json
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
    num_swaps = max(1, int(num_pairs * perturb_ratio))
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
    num_nodes = coords.shape[0]
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
        cost = problem.get_costs(batch, rec_tensor).item()

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo baseline for MVPDTSP")
    parser.add_argument("--val_dataset", type=str, default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1500)
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
        help="Fraction of solution to perturb in greedy mode (0=pure greedy, 1=nearly random)",
    )
    return parser.parse_args()


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

    print("[Stage] Initializing problem...")
    problem = MVPDTSP(
        p_size=args.graph_size,
        num_vehicles=args.num_vehicles,
        with_assert=False,
        use_makespan=(args.objective != "distance"),
    )

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
    }

    total_instances = results_data["val_size"]
    print(
        f"[Stage] Solving {total_instances} instances with Monte Carlo "
        f"({args.num_mc_samples} samples/instance)..."
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

        t0 = time.perf_counter()
        rec, vehicle_routes, _ = solve_instance(
            coords=coords,
            num_vehicles=args.num_vehicles,
            num_samples=args.num_mc_samples,
            objective=args.objective,
            greedy=args.greedy,
            perturb_ratio=args.perturb_ratio,
        )
        elapsed = time.perf_counter() - t0

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
            "solve_time": elapsed,
        }
        results_data["instances"].append(instance_data)

        if tqdm is None and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{total_instances} instances")

    results_dir = os.path.dirname(args.output) if args.output else "results"
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            results_dir, f"mvpdtsp_results_mc_{timestamp}.json"
        )

    print("[Stage] Writing results...")
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
