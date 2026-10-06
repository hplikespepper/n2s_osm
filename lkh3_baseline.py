#!/usr/bin/env python3
"""LKH-3 PDPTW multi-start baseline for the MVN2S MVPDTSP problem.

LKH minimizes total distance.  Each independent LKH solution is therefore a
candidate; candidates are re-scored with the exact MVN2S objective

    total distance + (number of vehicles - 1) * makespan

and the best candidate under that objective is saved.  Dataset loading, node
numbering, cost computation, and JSON fields match the existing MVN2S and
OR-Tools baselines.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from baseline_summary import summarize_result
from problems.problem_mvpdtsp import MVPDPDataset, MVPDTSP


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LKH = SCRIPT_DIR.parent / "LKH3" / "LKH-3.0.14" / "LKH"
SOLUTION_ROUTE_RE = re.compile(r"^\s*(.*?)\s*\(#\d+\)\s+Cost:")


def build_distance_matrix(coords: np.ndarray, scale: float) -> np.ndarray:
    """Use the same rounded Euclidean matrix construction as OR-Tools."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return np.rint(dist * scale).astype(np.int64)


def lkh_coordinates(coords: np.ndarray, num_vehicles: int) -> np.ndarray:
    """Remove MVN2S depot clones; LKH creates its own vehicle depots."""
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (n, 2), got {coords.shape}")
    if num_vehicles < 1 or len(coords) <= num_vehicles:
        raise ValueError("Invalid number of vehicles for the supplied coordinates")
    if (len(coords) - num_vehicles) % 2:
        raise ValueError("Pickup+delivery node count must be even")
    if not np.allclose(coords[:num_vehicles], coords[0], rtol=0, atol=1e-7):
        raise ValueError("MVN2S depot clones do not have identical coordinates")
    return np.concatenate((coords[:1], coords[num_vehicles:]), axis=0)


def write_pdptw_problem(
    path: Path,
    coords: np.ndarray,
    num_vehicles: int,
    scale: float,
    route_limit: int = None,
) -> None:
    """Write one MVN2S instance in LKH's TSPLIB-like PDPTW format."""
    compact = lkh_coordinates(coords, num_vehicles)
    matrix = build_distance_matrix(compact, scale)
    num_pairs = (len(compact) - 1) // 2
    # A route cannot be longer than (number of edges * largest edge).  The
    # loose factor leaves room for waiting/service changes while staying well
    # inside the exact range of a double used by LKH for time windows.
    horizon = int(max(1, matrix.max()) * (len(compact) + 1) * 4 + 1)

    lines = [
        f"NAME : {path.stem}",
        "TYPE : PDPTW",
        f"DIMENSION : {len(compact)}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        f"CAPACITY : {num_pairs}",
        f"VEHICLES : {num_vehicles}",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend(" ".join(map(str, row)) for row in matrix.tolist())
    lines.append("PICKUP_AND_DELIVERY_SECTION")
    depot_latest = horizon if route_limit is None else int(route_limit)
    if depot_latest <= 0:
        raise ValueError("route_limit must be positive")
    lines.append(f"1 0 0 {depot_latest} 0 0 0")
    for pair in range(num_pairs):
        pickup = 2 + pair
        delivery = 2 + num_pairs + pair
        lines.append(f"{pickup} 1 0 {horizon} 0 0 {delivery}")
    for pair in range(num_pairs):
        pickup = 2 + pair
        delivery = 2 + num_pairs + pair
        lines.append(f"{delivery} -1 0 {horizon} 0 {pickup} 0")
    lines.extend(("DEPOT_SECTION", "1", "-1", "EOF"))
    path.write_text("\n".join(lines) + "\n")


def write_parameter_file(
    path: Path,
    problem_path: Path,
    solution_path: Path,
    initial_tour_path: Path,
    seed: int,
    max_trials: int,
) -> None:
    """Write parameters for one fixed-trial LKH run with no time limit."""
    lines = [
        f"PROBLEM_FILE = {problem_path}",
        f"MTSP_SOLUTION_FILE = {solution_path}",
        f"INITIAL_TOUR_FILE = {initial_tour_path}",
        "RUNS = 1",
        f"MAX_TRIALS = {max_trials}",
        f"SEED = {seed}",
        "MTSP_MIN_SIZE = 0",
        "TRACE_LEVEL = 0",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_initial_routes(coords: np.ndarray, num_vehicles: int) -> List[List[int]]:
    """Build deterministic, balanced, precedence-feasible MVN2S routes."""
    num_pairs = (len(coords) - num_vehicles) // 2
    pickup_start = num_vehicles
    delivery_start = num_vehicles + num_pairs
    depot = coords[0]
    pickup_coords = coords[pickup_start:delivery_start]
    angles = np.arctan2(pickup_coords[:, 1] - depot[1], pickup_coords[:, 0] - depot[0])
    pair_groups = np.array_split(np.argsort(angles), num_vehicles)
    routes = []
    for vehicle, group in enumerate(pair_groups):
        unpicked = set(int(pair) for pair in group)
        carried = set()
        body = []
        current = vehicle
        while unpicked or carried:
            candidates = [(pickup_start + pair, pair, True) for pair in unpicked]
            candidates.extend((delivery_start + pair, pair, False) for pair in carried)
            node, pair, is_pickup = min(
                candidates,
                key=lambda item: (float(np.linalg.norm(coords[current] - coords[item[0]])), item[0]),
            )
            body.append(node)
            current = node
            if is_pickup:
                unpicked.remove(pair)
                carried.add(pair)
            else:
                carried.remove(pair)
        routes.append([vehicle, *body, vehicle])
    return routes


def write_initial_tour(
    path: Path,
    routes: Sequence[Sequence[int]],
    num_vehicles: int,
    graph_size: int,
) -> None:
    """Write a feasible mTSP tour, including LKH's extra depot nodes."""
    base_dimension = graph_size + 1
    tour = []
    for vehicle, route in enumerate(routes):
        depot_node = 1 if vehicle == 0 else base_dimension + vehicle
        tour.append(depot_node)
        tour.extend(2 + node - num_vehicles for node in route[1:-1])
    dimension = base_dimension + num_vehicles - 1
    if len(tour) != dimension or sorted(tour) != list(range(1, dimension + 1)):
        raise ValueError("Initial LKH tour does not contain every transformed node once")
    lines = [
        f"NAME : {path.stem}",
        "TYPE : TOUR",
        f"DIMENSION : {dimension}",
        "TOUR_SECTION",
        *(str(node) for node in tour),
        "-1",
        "EOF",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_mtsp_solution(path: Path) -> List[List[int]]:
    """Parse the route prefix of every line in MTSP_SOLUTION_FILE."""
    routes = []
    for line in path.read_text().splitlines():
        match = SOLUTION_ROUTE_RE.match(line)
        if not match:
            continue
        route = [int(token) for token in match.group(1).split()]
        if not route:
            continue
        if route[0] != 1:
            route.insert(0, 1)
        if route[-1] != 1:
            route.append(1)
        routes.append(route)
    if not routes:
        raise ValueError(f"No vehicle routes found in {path}")
    return routes


def map_lkh_routes(
    routes: Sequence[Sequence[int]],
    num_vehicles: int,
    num_pairs: int,
) -> List[List[int]]:
    """Map LKH's one-depot, 1-based nodes to MVN2S depot clones."""
    if len(routes) != num_vehicles:
        raise ValueError(f"Expected {num_vehicles} LKH routes, found {len(routes)}")

    def map_customer(node: int) -> int:
        if 2 <= node <= num_pairs + 1:
            return num_vehicles + node - 2
        if num_pairs + 2 <= node <= 2 * num_pairs + 1:
            return num_vehicles + num_pairs + node - (num_pairs + 2)
        raise ValueError(f"Unexpected LKH customer node {node}")

    mapped = []
    for vehicle, route in enumerate(routes):
        body = [map_customer(node) for node in route if node != 1]
        mapped.append([vehicle, *body, vehicle])
    expected = list(range(num_vehicles, num_vehicles + 2 * num_pairs))
    actual = sorted(node for route in mapped for node in route[1:-1])
    if actual != expected:
        raise ValueError("LKH routes do not cover every pickup/delivery node exactly once")
    return mapped


def build_rec(num_vehicles: int, num_pairs: int, routes: Sequence[Sequence[int]]) -> List[int]:
    """Convert explicit routes to MVN2S's successor-array representation."""
    rec = list(range(num_vehicles + 2 * num_pairs))
    for vehicle, route in enumerate(routes):
        if len(route) < 2 or route[0] != vehicle or route[-1] != vehicle:
            raise ValueError(f"Invalid route for vehicle {vehicle}: {route}")
        for current, nxt in zip(route, route[1:]):
            rec[current] = nxt
    return rec


def compute_costs(
    problem: MVPDTSP,
    coords: np.ndarray,
    rec: Sequence[int],
) -> Tuple[float, float, List[float]]:
    rec_tensor = torch.as_tensor(rec, dtype=torch.long).unsqueeze(0)
    coords_tensor = torch.as_tensor(coords, dtype=torch.float).unsqueeze(0)
    batch = {"coordinates": coords_tensor}
    problem.check_feasibility(rec_tensor)
    distance, makespan = problem.compute_cost_components(batch, rec_tensor)
    vehicle_costs = problem._get_route_lengths(batch, rec_tensor)
    return float(distance.item()), float(makespan.item()), vehicle_costs[0].tolist()


def objective_cost(distance: float, makespan: float, num_vehicles: int) -> float:
    return distance + (num_vehicles - 1) * makespan


def run_lkh_candidate(
    coords: np.ndarray,
    num_vehicles: int,
    lkh_path: Path,
    scale: float,
    seed: int,
    max_trials: int,
    workdir: Path,
    route_limit: int = None,
) -> Tuple[List[int], List[List[int]], str]:
    """Run LKH once and return a feasible MVN2S solution candidate."""
    problem_path = workdir / "instance.pdptw"
    parameter_path = workdir / "instance.par"
    solution_path = workdir / "solution.txt"
    initial_tour_path = workdir / "initial.tour"
    initial_routes = build_initial_routes(coords, num_vehicles)
    write_pdptw_problem(problem_path, coords, num_vehicles, scale, route_limit)
    write_initial_tour(
        initial_tour_path, initial_routes, num_vehicles,
        len(coords) - num_vehicles,
    )
    write_parameter_file(
        parameter_path, problem_path, solution_path, initial_tour_path,
        seed, max_trials,
    )
    completed = subprocess.run(
        [str(lkh_path), str(parameter_path)],
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(f"LKH exited with code {completed.returncode}:\n{tail}")
    if not solution_path.exists():
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(f"LKH did not create {solution_path}:\n{tail}")
    num_pairs = (len(coords) - num_vehicles) // 2
    routes = map_lkh_routes(parse_mtsp_solution(solution_path), num_vehicles, num_pairs)
    return build_rec(num_vehicles, num_pairs, routes), routes, completed.stdout


def solve_instance(
    coords: np.ndarray,
    graph_size: int,
    num_vehicles: int,
    lkh_path: Path,
    scale: float,
    seeds: Sequence[int],
    max_trials: int,
    start_workers: int = 1,
) -> Tuple[List[int], List[List[int]], float, float, List[float], int,
           int, List[int], List[str], List[Dict[str, Any]]]:
    """Generate independent LKH candidates and re-rank by MVN2S objective."""
    if not seeds:
        raise ValueError("At least one LKH seed is required")
    problem = MVPDTSP(graph_size, num_vehicles=num_vehicles, use_makespan=True)
    num_pairs = graph_size // 2
    compact_matrix = build_distance_matrix(lkh_coordinates(coords, num_vehicles), scale)

    def route_integer_cost(body):
        compact_nodes = [0, *(1 + node - num_vehicles for node in body), 0]
        return sum(int(compact_matrix[a, b]) for a, b in zip(compact_nodes, compact_nodes[1:]))

    # Use a known-feasible balanced initial tour to scale route constraints.
    # The upper diversified limits remain feasible by construction.
    initial_routes = build_initial_routes(coords, num_vehicles)
    initial_route_limit = max(
        route_integer_cost(route[1:-1]) for route in initial_routes
    )
    route_limits = [None]
    if len(seeds) > 1:
        route_limits.extend(
            max(1, int(round(initial_route_limit * ratio)))
            for ratio in np.linspace(0.9, 1.1, len(seeds) - 1)
        )
    best = None
    failures = []
    candidate_details = []
    with tempfile.TemporaryDirectory(prefix="mvn2s_lkh3_") as tmp:
        root = Path(tmp)

        def run_start(item):
            start, (seed, route_limit) = item
            candidate_dir = root / f"start_{start}"
            candidate_dir.mkdir()
            candidate_start = time.perf_counter()
            try:
                rec, routes, _ = run_lkh_candidate(
                    coords, num_vehicles, lkh_path, scale, seed,
                    max_trials, candidate_dir, route_limit,
                )
                elapsed = time.perf_counter() - candidate_start
                return start, seed, route_limit, rec, routes, elapsed, None
            except (AssertionError, OSError, RuntimeError, ValueError) as exc:
                elapsed = time.perf_counter() - candidate_start
                return start, seed, route_limit, None, None, elapsed, exc

        items = list(enumerate(zip(seeds, route_limits)))
        workers = min(max(1, start_workers), len(items))
        if workers == 1:
            candidate_results = map(run_start, items)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            candidate_results = executor.map(run_start, items)
        try:
            for start, seed, route_limit, rec, routes, elapsed, error in candidate_results:
                detail = {
                    "start_index": start + 1,
                    "seed": seed,
                    "route_limit": route_limit,
                    "solve_time": elapsed,
                    "succeeded": False,
                }
                if error is not None:
                    detail["failure"] = str(error)
                    candidate_details.append(detail)
                    failures.append(f"seed {seed}, route_limit {route_limit}: {error}")
                    continue
                try:
                    distance, makespan, vehicle_costs = compute_costs(problem, coords, rec)
                except (AssertionError, RuntimeError, ValueError) as exc:
                    detail["failure"] = str(exc)
                    candidate_details.append(detail)
                    failures.append(f"seed {seed}, route_limit {route_limit}: {exc}")
                    continue
                score = objective_cost(distance, makespan, num_vehicles)
                detail.update({
                    "succeeded": True,
                    "cost": score,
                    "distance": distance,
                    "makespan": makespan,
                })
                candidate_details.append(detail)
                limit_key = route_limit if route_limit is not None else 0
                candidate = (score, distance, makespan, seed, limit_key,
                             rec, routes, vehicle_costs, route_limit)
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
        finally:
            if workers > 1:
                executor.shutdown(wait=True)
    if best is None:
        raise RuntimeError("All LKH starts failed:\n" + "\n".join(failures))
    score, distance, makespan, seed, _, rec, routes, vehicle_costs, selected_limit = best
    for detail in candidate_details:
        detail["selected"] = detail["succeeded"] and detail["seed"] == seed
    return (rec, routes, distance, makespan, vehicle_costs, seed,
            selected_limit, route_limits, failures, candidate_details)


def _init_worker() -> None:
    torch.set_num_threads(1)


def _candidate_seeds(base_seed: int, instance_id: int, num_starts: int) -> List[int]:
    modulus = 2_147_483_647
    first = (base_seed + instance_id * num_starts) % modulus
    return [((first + start) % modulus) or 1 for start in range(num_starts)]


def _solve_task(task):
    instance_id, coords, args = task
    seeds = _candidate_seeds(args.seed, instance_id, args.num_starts)
    start = time.perf_counter()
    (rec, routes, distance, makespan, vehicle_costs, selected_seed,
     selected_limit, route_limits, failures, candidate_details) = solve_instance(
        coords=coords,
        graph_size=args.graph_size,
        num_vehicles=args.num_vehicles,
        lkh_path=Path(args.lkh_path),
        scale=args.scale,
        seeds=seeds,
        max_trials=args.max_trials,
        start_workers=args.start_workers,
    )
    elapsed = time.perf_counter() - start
    return {
        "instance_id": instance_id,
        "candidate_seeds": seeds,
        "selected_seed": selected_seed,
        "candidate_route_limits": [
            None if limit is None else limit / args.scale for limit in route_limits
        ],
        "selected_route_limit": (
            None if selected_limit is None else selected_limit / args.scale
        ),
        "num_candidates": len(seeds) - len(failures),
        "candidate_failures": failures,
        "candidate_results": [
            {
                **detail,
                "route_limit": (
                    None if detail["route_limit"] is None
                    else detail["route_limit"] / args.scale
                ),
            }
            for detail in candidate_details
        ],
        "worker_pid": os.getpid(),
        "solve_time": elapsed,
        "best_cost": objective_cost(distance, makespan, args.num_vehicles),
        "best_distance_cost": distance,
        "best_makespan_cost": makespan,
        "makespan_weight": args.num_vehicles - 1,
        "best_vehicle_distance_costs": vehicle_costs,
        "best_vehicle_completion_times": vehicle_costs,
        "best_rec": rec,
        "vehicle_routes": routes,
        "route_lengths": [len(route) for route in routes],
        "total_nodes": len(coords),
        "coordinates": coords.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val_dataset", default="./datasets/pdp_20.pkl")
    parser.add_argument("--val_size", type=int, default=256,
                        help="Number of leading fixed dataset instances (default: 256, matching MVN2S evaluation)")
    parser.add_argument("--graph_size", type=int, default=20)
    parser.add_argument("--num_vehicles", type=int, default=2)
    parser.add_argument("--T_max", type=int, default=1500, help="Metadata only")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Processes across instances (0=available CPUs)")
    parser.add_argument("--num_starts", type=int, default=4,
                        help="Independent LKH seeds per instance (default: 4)")
    parser.add_argument("--start_workers", type=int, default=4,
                        help="Concurrent LKH starts within each instance (default: 4)")
    parser.add_argument("--max_trials", type=int, default=5000,
                        help="LKH MAX_TRIALS for each independent start; no time limit is used")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1e6)
    parser.add_argument("--lkh_path", default=str(DEFAULT_LKH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.graph_size <= 0 or args.graph_size % 2:
        parser.error("--graph_size must be positive and even")
    if (args.num_vehicles < 1 or args.val_size < 1 or args.num_starts < 1
            or args.start_workers < 1):
        parser.error("--num_vehicles, --val_size, --num_starts and --start_workers must be positive")
    if args.num_workers < 0 or args.max_trials < 1:
        parser.error("--num_workers must be nonnegative and --max_trials must be positive")
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    lkh_path = Path(args.lkh_path).expanduser().resolve()
    if not lkh_path.is_file() or not os.access(lkh_path, os.X_OK):
        parser.error(f"--lkh_path is not an executable file: {lkh_path}")
    args.lkh_path = str(lkh_path)
    return args


def main() -> None:
    args = parse_args()
    print("[Stage] Loading the MVN2S dataset...")
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

    available_cpus = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else (os.cpu_count() or 1))
    num_workers = min(args.num_workers or available_cpus, len(dataset))
    start_workers = min(args.start_workers, args.num_starts)
    max_lkh_processes = num_workers * start_workers
    _init_worker()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "timestamp": timestamp,
        "problem": "mvpdtsp",
        "method": "lkh3_pdptw_multistart",
        "lkh_version": "3.0.14",
        "lkh_path": args.lkh_path,
        "graph_size": args.graph_size,
        "T_max": args.T_max,
        "val_dataset": args.val_dataset,
        "val_size": min(args.val_size, len(dataset)),
        "instances": [],
        "num_vehicles": args.num_vehicles,
        "objective": "distance+makespan",
        "makespan_weight": args.num_vehicles - 1,
        "num_starts": args.num_starts,
        "start_workers": start_workers,
        "seed": args.seed,
        "num_workers": num_workers,
        "max_trials": args.max_trials,
        "max_total_trials": args.max_trials * args.num_starts,
        "time_limit": None,
        "scale": args.scale,
    }

    total_instances = results_data["val_size"]
    print(
        f"[Stage] Solving {total_instances} instances with LKH-3 PDPTW "
        f"({args.num_starts} starts, {start_workers} concurrent, "
        f"MAX_TRIALS={args.max_trials}/start, no time limit)..."
    )
    print(
        f"[Stage] Using {num_workers} instance worker(s), up to "
        f"{max_lkh_processes} concurrent LKH processes"
    )
    if max_lkh_processes > available_cpus:
        print(
            f"[WARN] Requested LKH concurrency ({max_lkh_processes}) exceeds "
            f"the available CPU affinity ({available_cpus})"
        )
    tasks = ((i, dataset[i]["coordinates"].cpu().numpy(), args)
             for i in range(total_instances))
    wall_start = time.perf_counter()

    def collect(iterator):
        if tqdm is not None:
            iterator = tqdm(iterator, total=total_instances, desc="Solving", unit="inst")
        for result in iterator:
            results_data["instances"].append(result)
            if tqdm is None:
                print(f"[Stage] Completed {len(results_data['instances'])}/{total_instances}")

    if num_workers == 1:
        collect(map(_solve_task, tasks))
    else:
        with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker) as pool:
            collect(pool.imap_unordered(_solve_task, tasks, chunksize=1))
    results_data["solve_wall_time"] = time.perf_counter() - wall_start
    results_data["instances"].sort(key=lambda result: result["instance_id"])

    results_dir = (os.path.dirname(args.output) or ".") if args.output else "results"
    os.makedirs(results_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        results_dir, f"mvpdtsp_results_lkh3_{timestamp}.json"
    )
    results_data["summary"] = summarize_result(results_data)
    with open(output_path, "w") as handle:
        json.dump(results_data, handle, indent=2)
    print(f"[Stage] Solve wall time: {results_data['solve_wall_time']:.3f}s")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
