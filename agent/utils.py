# -*- coding: utf-8 -*-

import time
import io
from contextlib import redirect_stdout
import torch
import os
import json
import pickle
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from utils.logger import log_to_screen, log_to_tb_val
from utils import get_inner_model
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import random
from data.collate import osm_collate_fn, pdp_collate_fn

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(
    rank,
    problem,
    agent,
    val_dataset,
    tb_logger,
    distributed=False,
    _id=None,
    use_collectives=True,
    shard_result_path=None,
    aggregate_result_paths=None,
):
    process_group_initialized = False
    original_actor = agent.actor
            
    # Validate mode
    if rank==0: print('\nValidating...', flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = agent.opts
    using_existing_process_group = distributed and opts.distributed and dist.is_available() and dist.is_initialized()
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
    agent.eval()
    
    # Create validation dataset with problem-specific parameters
    if problem.NAME == 'pdtsp_osm':
        val_dataset = problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.val_size,
            filename=val_dataset,
            osm_place=opts.osm_place,
            capacity=opts.capacity
        )
        collate_fn = osm_collate_fn
    else:
        if problem.NAME == 'mvpdtsp':
            val_dataset = problem.make_dataset(
                size=opts.graph_size,
                num_samples=opts.val_size,
                filename=val_dataset,
                num_vehicles=opts.num_vehicles
            )
        else:
            val_dataset = problem.make_dataset(
                size=opts.graph_size,
                num_samples=opts.val_size,
                filename=val_dataset
            )
        collate_fn = pdp_collate_fn

    if distributed and opts.distributed and not using_existing_process_group:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        opts.device = device
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=opts.world_size,
            rank=rank,
            timeout=timedelta(seconds=opts.dist_timeout),
        )
        process_group_initialized = True
        agent.actor.to(device)
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    elif distributed and opts.distributed and using_existing_process_group:
        torch.cuda.set_device(rank)
        opts.device = torch.device("cuda", rank)
    elif not distributed and opts.distributed and dist.is_available() and dist.is_initialized():
        torch.cuda.set_device(rank)
        opts.device = torch.device("cuda", rank)

    if opts.distributed and dist.is_available() and dist.is_initialized():
        # Validation does not need gradient synchronization. Running inference
        # through DDP can introduce implicit collectives in forward() and make
        # ranks with uneven validation latency timeout at later barriers.
        agent.actor = get_inner_model(agent.actor).to(opts.device)

    
    if distributed and opts.distributed:
        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        local_indices = list(train_sampler)  # dataset indices this rank will process
        val_dataloader = DataLoader(val_dataset, batch_size = opts.val_batch_size // opts.world_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler,
                                    collate_fn=collate_fn)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=False,
                                   num_workers=0,
                                   pin_memory=True,
                                   collate_fn=collate_fn)
    
    s_time = time.time()
    bv = []
    cost_hist = []
    best_hist = []
    r = []
    best_solutions = []  # Store best solutions
    initial_solutions = []
    initial_costs = []
    original_coords = []  # Store original coordinates (not augmented)
    solve_times = []  # Per-instance solve time
    component_histories = []
    component_steps = None
    for batch in tqdm(
        val_dataloader,
        desc='inference',
        disable=opts.no_progress_bar or (opts.distributed and rank != 0),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    ):
        batch_start = time.time()
        (bv_, cost_hist_, best_hist_, r_, solutions_, init_solutions_, init_costs_,
         coords_, component_history_) = agent.rollout(problem,
                                                        opts.val_m,
                                                        batch,
                                                        do_sample = True,
                                                        show_bar = rank==0)
        batch_elapsed = time.time() - batch_start
        batch_size_actual = bv_.size(0)
        solve_times.extend([batch_elapsed / batch_size_actual] * batch_size_actual)
        bv.append(bv_)
        cost_hist.append(cost_hist_)
        best_hist.append(best_hist_)
        r.append(r_)
        best_solutions.append(solutions_)
        initial_solutions.append(init_solutions_)
        initial_costs.append(init_costs_)
        original_coords.append(coords_)
        if component_history_ is not None:
            if component_steps is None:
                component_steps = component_history_['steps']
            elif component_steps != component_history_['steps']:
                raise RuntimeError('Inconsistent component-history steps between batches')
            component_histories.append(component_history_)
    bv = torch.cat(bv, 0)
    cost_hist = torch.cat(cost_hist, 0)
    best_hist = torch.cat(best_hist, 0)
    r = torch.cat(r, 0)
    best_solutions = torch.cat(best_solutions, 0)  # Concatenate best solutions
    initial_solutions = torch.cat(initial_solutions, 0)
    initial_costs = torch.cat(initial_costs, 0)
    original_coords = torch.cat(original_coords, 0)  # Concatenate original coordinates
    component_history = None
    if component_histories:
        component_history = {
            'steps': component_steps,
            'objective': torch.cat([x['objective'] for x in component_histories], 0),
            'distance': torch.cat([x['distance'] for x in component_histories], 0),
            'makespan': torch.cat([x['makespan'] for x in component_histories], 0),
        }

    val_score = None
    if distributed and opts.distributed and not use_collectives:
        assert shard_result_path is not None
        torch.save(
            {
                'rank': int(rank),
                'indices': torch.tensor(local_indices, dtype=torch.long),
                'elapsed': torch.tensor([time.time() - s_time], dtype=torch.float32),
                'bv': bv.detach().cpu(),
                'cost_hist': cost_hist.detach().cpu(),
                'best_hist': best_hist.detach().cpu(),
                'reward': r.detach().cpu(),
                'best_solutions': best_solutions.detach().cpu(),
                'initial_solutions': initial_solutions.detach().cpu(),
                'initial_costs': initial_costs.detach().cpu(),
                'original_coords': original_coords.detach().cpu(),
                'solve_times': torch.tensor(solve_times, dtype=torch.float32),
                'component_history': None if component_history is None else {
                    'steps': component_history['steps'],
                    'objective': component_history['objective'].detach().cpu(),
                    'distance': component_history['distance'].detach().cpu(),
                    'makespan': component_history['makespan'].detach().cpu(),
                },
            },
            shard_result_path,
        )

        if aggregate_result_paths is None:
            agent.actor = original_actor
            return None

        while not all(os.path.exists(path) for path in aggregate_result_paths):
            time.sleep(5.0)

        shard_results = [torch.load(path, map_location='cpu') for path in aggregate_result_paths]
        all_indices = torch.cat([result['indices'] for result in shard_results], 0)
        sort_order = torch.argsort(all_indices)

        initial_cost = torch.cat([result['cost_hist'][:, 0] for result in shard_results], 0)[sort_order]
        time_used = torch.cat([result['elapsed'] for result in shard_results], 0)
        bv = torch.cat([result['bv'] for result in shard_results], 0)[sort_order]
        costs_history = torch.cat([result['cost_hist'] for result in shard_results], 0)[sort_order]
        search_history = torch.cat([result['best_hist'] for result in shard_results], 0)[sort_order]
        reward = torch.cat([result['reward'] for result in shard_results], 0)[sort_order]
        initial_solutions = torch.cat([result['initial_solutions'] for result in shard_results], 0)[sort_order]
        best_solutions = torch.cat([result['best_solutions'] for result in shard_results], 0)[sort_order]
        initial_costs = torch.cat([result['initial_costs'] for result in shard_results], 0)[sort_order]
        original_coords = torch.cat([result['original_coords'] for result in shard_results], 0)[sort_order]
        solve_times_tensor = torch.cat([result['solve_times'] for result in shard_results], 0)[sort_order]
        solve_times = solve_times_tensor.tolist()
        if shard_results[0].get('component_history') is not None:
            component_history = {
                'steps': shard_results[0]['component_history']['steps'],
            }
            for key in ('objective', 'distance', 'makespan'):
                component_history[key] = torch.cat(
                    [result['component_history'][key] for result in shard_results], 0
                )[sort_order]
        val_score = bv.mean().item()
        use_local_aggregation = True
    else:
        use_local_aggregation = False
    local_score_sum = bv.sum()
    local_score_count = torch.tensor([bv.numel()], dtype=torch.float32, device=bv.device)
    
    if use_local_aggregation:
        pass
    elif distributed and opts.distributed:
        global_score_sum = local_score_sum.detach().clone()
        global_score_count = local_score_count.clone()
        dist.all_reduce(global_score_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_score_count, op=dist.ReduceOp.SUM)
        if rank == 0:
            val_score = (global_score_sum / global_score_count).item()

        initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
        time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())
        initial_solutions = gather_tensor_and_concat(initial_solutions.contiguous())
        best_solutions = gather_tensor_and_concat(best_solutions.contiguous())
        initial_costs = gather_tensor_and_concat(initial_costs.contiguous())
        original_coords = gather_tensor_and_concat(original_coords.contiguous())
        if component_history is not None:
            gathered_component_history = {'steps': component_history['steps']}
            for key in ('objective', 'distance', 'makespan'):
                gathered_component_history[key] = gather_tensor_and_concat(
                    component_history[key].contiguous()
                )
            component_history = gathered_component_history
        world_size = dist.get_world_size()
        solve_times = (gather_tensor_and_concat(torch.tensor(solve_times, dtype=torch.float32).cuda()) / world_size).cpu().tolist()
        # Reorder all gathered results back to original dataset index order.
        # DistributedSampler(shuffle=False) with world_size W distributes indices as:
        #   rank 0: [0, W, 2W, ...]  rank 1: [1, W+1, 2W+1, ...]  etc.
        # After gather the order is [rank0_data, rank1_data, ...] i.e. [0, W, 2W, ..., 1, W+1, ...]
        # We must sort by original index to align with OR-Tools/MC sequential ordering.
        all_indices = gather_tensor_and_concat(torch.tensor(local_indices, dtype=torch.long).cuda())
        sort_order = torch.argsort(all_indices)
        initial_cost = initial_cost[sort_order]
        bv = bv[sort_order]
        costs_history = costs_history[sort_order]
        search_history = search_history[sort_order]
        reward = reward[sort_order]
        initial_solutions = initial_solutions[sort_order]
        best_solutions = best_solutions[sort_order]
        initial_costs = initial_costs[sort_order]
        original_coords = original_coords[sort_order]
        if component_history is not None:
            for key in ('objective', 'distance', 'makespan'):
                component_history[key] = component_history[key][sort_order]
        solve_times = [solve_times[i] for i in sort_order.tolist()]
    
    else:
        if rank == 0:
            val_score = (local_score_sum / local_score_count).item()
        initial_cost = cost_hist[:,0] # bs
        time_used = torch.tensor([time.time() - s_time]) # bs
        bv = bv
        costs_history = cost_hist
        search_history = best_hist
        reward = r
        
    init_distance = None
    init_makespan = None
    best_distance = None
    best_makespan = None
    if problem.NAME == 'mvpdtsp':
        # Use original coordinates (not augmented) for consistency with saved results
        # Data augmentation (rotate/flip) preserves distances, so costs should be equivalent
        batch_cost = {'coordinates': original_coords.detach().cpu()}
        best_distance, best_makespan = problem.compute_cost_components(batch_cost, best_solutions.detach().cpu().long())
        init_distance, init_makespan = problem.compute_cost_components(batch_cost, initial_solutions.detach().cpu().long())
        
        # Verify consistency: recomputed cost from original coords should match bv from augmented coords
        # Note: Due to data augmentation (rotate/flip), there may be small numerical differences
        if rank == 0:
            recomputed_cost = problem.objective_cost(best_distance, best_makespan)
            max_diff = (bv.cpu() - recomputed_cost).abs().max().item()
            if max_diff > 0.1:  # Allow small difference due to augmentation
                print(f"WARNING: Cost mismatch detected! Max diff: {max_diff:.6f}")
                print(f"  bv (from augmented): [{bv.min():.4f}, {bv.max():.4f}]")
                print(f"  recomputed (from original): [{recomputed_cost.min():.4f}, {recomputed_cost.max():.4f}]")
                print(f"  This may be due to data augmentation transformations.")
            else:
                print(f"✓ Cost verification passed (max diff: {max_diff:.4f})")

    # Capture exactly the statistics printed to the terminal for evaluation reports.
    screen_summary = ""
    if rank == 0:
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            log_to_screen(time_used,
                                  initial_cost,
                                  bv,
                                  reward,
                                  costs_history,
                                  search_history,
                                  batch_size = opts.val_size,
                                  dataset_size = len(val_dataset),
                                  T = opts.T_max,
                                  init_distance = init_distance,
                                  init_makespan = init_makespan,
                                  best_distance = best_distance,
                                  best_makespan = best_makespan)
        screen_summary = summary_buffer.getvalue()
        print(screen_summary, end="", flush=True)

    # log to tb
    if(not opts.no_tb) and rank == 0:
        log_to_tb_val(tb_logger,
                      time_used, 
                      initial_cost, 
                      bv, 
                      reward, 
                      costs_history,
                      search_history,
                      batch_size = opts.val_size,
                      val_size =  opts.val_size,
                      dataset_size = len(val_dataset), 
                      T = opts.T_max,
                      epoch = _id,
                      init_distance = init_distance,
                      init_makespan = init_makespan,
                      best_distance = best_distance,
                      best_makespan = best_makespan)
    
    if rank == 0 and opts.eval_only:
        print_solution = getattr(opts, 'print_solution', False)

        # Create results directory
        results_dir = getattr(opts, "results_dir", "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created results directory: {results_dir}")

        # Prepare results data (final solutions always saved)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            "timestamp": timestamp,
            "problem": opts.problem,
            "graph_size": opts.graph_size,
            "T_max": opts.T_max,
            "load_path": opts.load_path,
            "val_dataset": opts.val_dataset,
            "val_size": opts.val_size,
            "instances": []
        }
        if problem.NAME == 'mvpdtsp':
            results_data["num_vehicles"] = opts.num_vehicles
            results_data["objective"] = problem.objective
            results_data["makespan_weight"] = problem.makespan_weight
        if component_history is not None:
            results_data["component_history_steps"] = component_history['steps']

        # Use the same evaluation order as rollout for saving (align with best_solutions)
        coords_list = original_coords.detach().cpu().numpy().tolist()

        # Use already computed cost components (avoid redundant calculation)
        dist_costs = best_distance
        makespan_costs = best_makespan
        init_dist_costs = init_distance
        init_makespan_costs = init_makespan
        best_vehicle_costs = None
        if problem.NAME == 'mvpdtsp':
            # Get individual vehicle route lengths for detailed reporting
            # Use original coordinates to match the saved coordinate data
            batch_cost = {'coordinates': original_coords.cpu()}
            best_vehicle_costs = problem._get_route_lengths(batch_cost, best_solutions.cpu().long())

        def decode_vehicle_routes(rec, num_vehicles):
            routes = []
            total_nodes = rec.size(0)
            for v in range(num_vehicles):
                route = [v]
                current = v
                visited = set([v])
                for _ in range(total_nodes):
                    nxt = int(rec[current].item())
                    route.append(nxt)
                    if nxt == v:
                        break
                    if nxt in visited:
                        break
                    visited.add(nxt)
                    current = nxt
                routes.append(route)
            return routes

        if print_solution:
            print("\n" + "="*60)
            print("INITIAL AND FINAL SOLUTIONS:")
            print("="*60)

        for i in range(min(opts.val_size, len(best_solutions))):
            solution = best_solutions[i]
            best_distance_cost = dist_costs[i].item() if dist_costs is not None else None
            best_makespan_cost = makespan_costs[i].item() if makespan_costs is not None else None
            cost = bv[i].item()
            init_solution = initial_solutions[i]
            init_cost = initial_costs[i].item()
            coordinates = torch.as_tensor(coords_list[i]).cpu().numpy().tolist()

            if problem.NAME == 'mvpdtsp':
                vehicle_routes = decode_vehicle_routes(solution, opts.num_vehicles)
                rec_tensor = solution.detach().cpu().long().unsqueeze(0)
                coords_tensor = torch.as_tensor(coordinates).float().unsqueeze(0)
                batch_cost = {"coordinates": coords_tensor}
                inst_distance, inst_makespan = problem.compute_cost_components(batch_cost, rec_tensor)
                inst_vehicle_costs = problem._get_route_lengths(batch_cost, rec_tensor)
                inst_distance_val = inst_distance.item()
                inst_makespan_val = inst_makespan.item()
                inst_vehicle_costs_list = inst_vehicle_costs[0].cpu().tolist()
                inst_objective_val = problem.objective_cost(inst_distance, inst_makespan).item()
                inst_proposed_val = problem.proposed_cost(inst_distance, inst_makespan).item()
                active_vehicles = sum(cost > 1e-8 for cost in inst_vehicle_costs_list)
                instance_data = {
                    "instance_id": i,
                    "best_cost": inst_objective_val,
                    "objective": problem.objective,
                    "best_distance_cost": inst_distance_val,
                    "best_makespan_cost": inst_makespan_val,
                    "proposed_cost": inst_proposed_val,
                    "makespan_weight": problem.makespan_weight,
                    "active_vehicles": active_vehicles,
                    "vehicle_utilization": active_vehicles / float(opts.num_vehicles),
                    "best_vehicle_distance_costs": inst_vehicle_costs_list,
                    "best_vehicle_completion_times": inst_vehicle_costs_list,
                    "best_rec": solution.cpu().numpy().tolist(),
                    "vehicle_routes": vehicle_routes,
                    "route_lengths": [len(r) for r in vehicle_routes],
                    "total_nodes": len(solution),
                    "coordinates": coordinates
                }
                if component_history is not None:
                    instance_data["component_history"] = {
                        "steps": component_history['steps'],
                        "best_cost": component_history['objective'][i].detach().cpu().tolist(),
                        "best_distance_cost": component_history['distance'][i].detach().cpu().tolist(),
                        "best_makespan_cost": component_history['makespan'][i].detach().cpu().tolist(),
                    }
                if print_solution:
                    init_vehicle_routes = decode_vehicle_routes(init_solution, opts.num_vehicles)
                    instance_data["initial_cost"] = init_cost
                    instance_data["initial_rec"] = init_solution.cpu().numpy().tolist()
                    if init_dist_costs is not None and init_makespan_costs is not None:
                        instance_data["initial_distance_cost"] = init_dist_costs[i].item()
                        instance_data["initial_makespan_cost"] = init_makespan_costs[i].item()
                if i < len(solve_times):
                    instance_data["solve_time"] = solve_times[i]
            else:
                instance_data = {
                    "instance_id": i,
                    "best_cost": cost,
                    "best_path": solution.cpu().numpy().tolist(),
                    "path_length": len(solution),
                    "coordinates": coordinates
                }
                if print_solution:
                    instance_data["initial_cost"] = init_cost
                    instance_data["initial_path"] = init_solution.cpu().numpy().tolist()
                if i < len(solve_times):
                    instance_data["solve_time"] = solve_times[i]

            results_data["instances"].append(instance_data)

            if print_solution:
                print(f"\nInstance {i+1}:")
                print(f"  Initial Cost: {init_cost:.6f}")
                print(f"  Best Cost: {cost:.6f}")
                if problem.NAME == 'mvpdtsp':
                    print(f"  Initial Rec: {init_solution.cpu().numpy().tolist()}")
                    print(f"  Final Rec: {solution.cpu().numpy().tolist()}")
                else:
                    print(f"  Initial Path: {init_solution.cpu().numpy().tolist()}")
                    print(f"  Final Path: {solution.cpu().numpy().tolist()}")
                    print(f"  Path Length: {len(solution)}")

        # Save results to JSON file (always)
        results_file = os.path.join(results_dir, f"{opts.problem}_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        summary_file = os.path.splitext(results_file)[0] + "_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(screen_summary)
        print(f"\nResults saved to: {results_file}")
        print(f"Evaluation statistics saved to: {summary_file}")
        if print_solution:
            print("="*60)
    
    if distributed and opts.distributed and process_group_initialized and dist.is_initialized():
        dist.destroy_process_group()

    agent.actor = original_actor

    return val_score
    
