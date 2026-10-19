#!/usr/bin/env python3
"""OR-Tools road-distance baseline; JSON is compatible with vis_osm_real.py.

Default: first 100 osm_val_20.pkl samples, 2 vehicles, objective D + M.
Distances are rounded to integers only for search (default scale=1000).
Reported costs use the same float32 road matrix and cost code as N2S.
No capacity constraint is added, matching the current MVPDTSP_OSM model.
Time limits apply to search per instance; solutions are not optimality proofs.
"""
import argparse
import json
import math
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    from ortools.util import optional_boolean_pb2
except ImportError as exc:
    raise ImportError(
        'Cannot import OR-Tools routing dependencies. Check this Python environment: '
        'python -m pip check (OR-Tools 9.12 requires protobuf>=5.29.3,<6).'
    ) from exc

from data.osm_mvpdp_dataset import OSMMVPDPDataset
from problems.problem_mvpdtsp_osm import MVPDTSP_OSM


ROOT = Path(__file__).resolve().parent


def solve_instance(sample, num_vehicles=2, objective='proposed', time_limit=15,
                   scale=1000, first_solution_only=False, log_search=False):
    """Solve an already expanded sample, returning logical-depot successor cycles."""
    if objective not in ('distance', 'only_makespan', 'proposed'):
        raise ValueError('Unsupported objective')
    if not math.isfinite(scale) or scale <= 0 or time_limit <= 0:
        raise ValueError('scale and time_limit must be positive and finite')
    distances = sample['dist'].detach().cpu().numpy().astype(np.float64)
    n = len(distances)
    if distances.shape != (n, n) or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Expected a finite nonnegative square road distance matrix')
    if not np.all(np.diag(distances) == 0) or not 1 <= num_vehicles < n:
        raise ValueError('Invalid distance diagonal or vehicle count')
    # Leave ample int64 headroom for route sums and the combined objective.
    if distances.max() * scale > (2**62 - 1) // (n * (num_vehicles + 1)):
        raise ValueError('Distance scale is too large for OR-Tools int64 costs')
    matrix = np.rint(distances * scale).astype(np.int64)
    depots = list(range(num_vehicles))
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depots, depots)
    routing = pywrapcp.RoutingModel(manager)

    def transit(a, b):
        return int(matrix[manager.IndexToNode(a), manager.IndexToNode(b)])

    transit_index = routing.RegisterTransitCallback(transit)
    if objective != 'only_makespan':
        routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    routing.AddDimension(transit_index, 0, int(matrix.max()) * (n + 1) + 1,
                         True, 'Distance')
    distance_dim = routing.GetDimensionOrDie('Distance')
    if objective != 'distance':
        distance_dim.SetGlobalSpanCostCoefficient(
            num_vehicles - 1 if objective == 'proposed' else 1)

    # Strict route order works even for co-located tasks and rounded zero arcs.
    order_index = routing.RegisterTransitCallback(lambda a, b: 1)
    routing.AddDimension(order_index, 0, n + 1, True, 'Order')
    order_dim = routing.GetDimensionOrDie('Order')
    for pickup, delivery in sample['pairs']:
        p, d = manager.NodeToIndex(int(pickup)), manager.NodeToIndex(int(delivery))
        routing.AddPickupAndDelivery(p, d)
        routing.solver().Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        routing.solver().Add(order_dim.CumulVar(p) < order_dim.CumulVar(d))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.local_search_operators.use_relocate_pair = optional_boolean_pb2.BOOL_TRUE
    params.time_limit.FromSeconds(time_limit)
    params.log_search = log_search
    if first_solution_only:
        params.solution_limit = 1
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError('OR-Tools found no feasible solution within the time limit')

    routes, rec = [], list(range(n))
    for vehicle in depots:
        index, route = routing.Start(vehicle), [vehicle]
        while not routing.IsEnd(index):
            nxt = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(nxt)
            rec[manager.IndexToNode(index)] = node
            route.append(node)
            index = nxt
        routes.append(route)

    # Check solver objective against the directed integer distances we supplied.
    lengths = [sum(int(matrix[a, b]) for a, b in zip(r, r[1:])) for r in routes]
    integer_cost = (sum(lengths) if objective == 'distance' else max(lengths)
                    if objective == 'only_makespan'
                    else sum(lengths) + (num_vehicles - 1) * max(lengths))
    if integer_cost != solution.ObjectiveValue():
        raise RuntimeError('OR-Tools objective disagrees with extracted routes')
    return rec, routes, integer_cost


def solve_task(task):
    i, dataset_index, sample, config = task
    start = time.perf_counter()
    rec, routes, integer_cost = solve_instance(sample, **config)
    k, n = config['num_vehicles'], len(rec)
    problem = MVPDTSP_OSM(n - k, num_vehicles=k, objective=config['objective'], with_assert=True)
    rec_tensor = torch.tensor([rec], dtype=torch.long)
    problem.check_feasibility(rec_tensor)
    batch = {'coordinates': sample['coordinates'].unsqueeze(0), 'dist': sample['dist'].unsqueeze(0)}
    distance, makespan = problem.compute_cost_components(batch, rec_tensor)
    vehicle_costs = problem._get_route_lengths(batch, rec_tensor)[0].tolist()
    active = sum(cost > 1e-8 for cost in vehicle_costs)
    return {
        'instance_id': i, 'dataset_index': dataset_index,
        'best_cost': problem.objective_cost(distance, makespan).item(),
        'objective': problem.objective,
        'best_distance_cost': distance.item(), 'best_makespan_cost': makespan.item(),
        'proposed_cost': problem.proposed_cost(distance, makespan).item(),
        'makespan_weight': problem.makespan_weight,
        'active_vehicles': active, 'vehicle_utilization': active / k,
        'best_vehicle_distance_costs': vehicle_costs,
        'best_rec': rec, 'vehicle_routes': routes,
        'route_lengths': [len(route) for route in routes], 'total_nodes': n,
        'coordinates': sample['coordinates'].tolist(),
        'original_total_nodes': n - k + 1, 'adapted_total_nodes': n,
        'solve_time': time.perf_counter() - start,
        'solver_integer_objective': integer_cost,
        'solver_scaled_objective': integer_cost / config['scale'],
    }


def init_worker():
    torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val_dataset', type=Path, default=ROOT / 'datasets/osm_val_20.pkl')
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--graph_size', type=int, default=20)
    parser.add_argument('--num_vehicles', type=int, default=2)
    parser.add_argument('--objective', choices=['distance', 'only_makespan', 'proposed'], default='proposed')
    parser.add_argument('--time_limit', type=int, default=15, help='Search seconds per instance')
    parser.add_argument('--scale', type=float, default=1000, help='Integer search units per dataset distance unit')
    parser.add_argument('--num_workers', type=int, default=1, help='CPU processes across instances')
    parser.add_argument('--first_solution_only', action='store_true')
    parser.add_argument('--log_search', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'result_ortools')
    parser.add_argument('--output', type=Path, help='Explicit output JSON path')
    args = parser.parse_args()
    if args.graph_size <= 0 or args.graph_size % 2:
        parser.error('--graph_size must be positive and even')
    if min(args.val_size, args.num_vehicles, args.time_limit, args.num_workers) < 1 or args.offset < 0:
        parser.error('Sizes, vehicle/worker counts and time_limit must be positive; offset must be nonnegative')
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error('--scale must be finite and positive')
    return args


def main():
    args = parse_args()
    init_worker()
    dataset = OSMMVPDPDataset(filename=args.val_dataset, size=args.graph_size,
                             num_samples=args.val_size, offset=args.offset,
                             num_vehicles=args.num_vehicles)
    config = {key: getattr(args, key) for key in (
        'num_vehicles', 'objective', 'time_limit', 'scale', 'first_solution_only', 'log_search')}
    workers = min(args.num_workers, len(dataset))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or args.output_dir / f'mvpdtsp_osm_results_ortools_{timestamp}.json'
    data = {
        'timestamp': timestamp, 'problem': 'mvpdtsp_osm', 'method': 'ortools',
        'graph_size': args.graph_size, 'num_vehicles': args.num_vehicles,
        'objective': args.objective,
        'makespan_weight': (args.num_vehicles - 1 if args.objective == 'proposed'
                            else 1 if args.objective == 'only_makespan' else 0),
        'load_path': None, 'val_dataset': str(args.val_dataset.resolve()),
        'source_dataset': str(args.val_dataset.resolve()), 'val_size': len(dataset),
        'offset': args.offset, 'distance_source': 'osm_directed_shortest_path',
        'makespan_unit': 'distance', 'original_total_nodes': args.graph_size + 1,
        'adapted_total_nodes': args.graph_size + args.num_vehicles,
        'time_limit': args.time_limit, 'scale': args.scale, 'num_workers': workers,
        'first_solution_only': args.first_solution_only, 'capacity_constraint': False,
        'first_solution_strategy': 'PATH_CHEAPEST_ARC',
        'local_search_metaheuristic': 'GUIDED_LOCAL_SEARCH', 'instances': [],
    }
    # OSM distance units are preserved. T_max and N2S initial-cost fields are
    # omitted because OR-Tools has neither N2S rollout steps nor its initial solution.
    print(f'Solving {len(dataset)} instances; {workers} CPU worker(s); {args.time_limit}s per instance', flush=True)
    tasks = ((i, args.offset + i, dataset[i], config) for i in range(len(dataset)))
    start = time.perf_counter()

    def collect(results):
        for result in results:
            data['instances'].append(result)
            print(f"[{len(data['instances'])}/{len(dataset)}] dataset_index={result['dataset_index']} "
                  f"cost={result['best_cost']:.3f}", flush=True)

    if workers == 1:
        collect(map(solve_task, tasks))
    else:
        with mp.get_context('spawn').Pool(workers, initializer=init_worker) as pool:
            collect(pool.imap_unordered(solve_task, tasks, chunksize=1))
    data['solve_wall_time'] = time.perf_counter() - start
    data['instances'].sort(key=lambda item: item['instance_id'])
    data['summary'] = {
        'mean_' + field: float(np.mean([item[field] for item in data['instances']]))
        for field in ('best_cost', 'best_distance_cost', 'best_makespan_cost', 'solve_time')
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    summary = '\n'.join(f'{key}: {value:.6f}' for key, value in data['summary'].items())
    output.with_name(output.stem + '_summary.txt').write_text(
        f"Instances: {len(dataset)}\nObjective: {args.objective}\n{summary}\n"
        f"solve_wall_time: {data['solve_wall_time']:.6f}\n")
    print(f'Saved results: {output}\n{summary}')


if __name__ == '__main__':
    main()
