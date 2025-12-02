# -*- coding: utf-8 -*-

import time
import torch
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
from utils.logger import log_to_screen, log_to_tb_val
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import random
from data.collate import osm_collate_fn, pdp_collate_fn

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def convert_adjacency_to_path(rec):
    """
    Convert N2S adjacency list (successor representation) to visit order path.
    
    Args:
        rec: adjacency list where rec[i] = j means "visit node j after node i"
             Can be a list, numpy array, or tensor
    
    Returns:
        path: list of nodes in visit order [node1, node2, ..., nodeN]
              (does not include depot at start/end)
    """
    if isinstance(rec, torch.Tensor):
        rec = rec.cpu().numpy().tolist()
    elif hasattr(rec, 'tolist'):
        rec = rec.tolist()
    
    path = []
    current = 0  # Start at depot
    max_steps = len(rec) + 1
    
    for _ in range(max_steps):
        next_node = rec[current]
        if next_node == 0:  # Return to depot
            break
        path.append(next_node)
        current = next_node
    
    return path

def extract_vehicle_paths_pdtsp2v(rec, vehicle_assignment):
    """
    Extract separate vehicle paths from unified PDTSP_2V solution.
    
    Args:
        rec: adjacency list (successor representation)
        vehicle_assignment: tensor or list indicating which vehicle each node belongs to
                           0=depot, 1=vehicle1, 2=vehicle2
    
    Returns:
        v1_path: list of nodes for vehicle 1 [node1, node2, ...]
        v2_path: list of nodes for vehicle 2 [node1, node2, ...]
    """
    if isinstance(vehicle_assignment, torch.Tensor):
        vehicle_assignment = vehicle_assignment.cpu().numpy().tolist()
    elif hasattr(vehicle_assignment, 'tolist'):
        vehicle_assignment = vehicle_assignment.tolist()
    
    # Convert to visit order first
    complete_path = convert_adjacency_to_path(rec)
    
    # Split by vehicle
    v1_path = []
    v2_path = []
    
    for node in complete_path:
        vehicle = vehicle_assignment[node]
        if vehicle == 1:
            v1_path.append(node)
        elif vehicle == 2:
            v2_path.append(node)
    
    return v1_path, v2_path

def validate(rank, problem, agent, val_dataset, tb_logger, distributed = False, _id = None):
            
    # Validate mode
    if rank==0: print('\nValidating...', flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = agent.opts
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
        val_dataset = problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.val_size,
            filename=val_dataset
        )
        collate_fn = pdp_collate_fn

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        if torch.cuda.device_count() > 1:
            agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                                   device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))

    
    if distributed and opts.distributed:
        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
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
    for batch in tqdm(val_dataloader, desc = 'inference', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        bv_, cost_hist_, best_hist_, r_, solutions_ = agent.rollout(problem,
                                                        opts.val_m,
                                                        batch,
                                                        do_sample = True,
                                                        show_bar = rank==0)
        bv.append(bv_)
        cost_hist.append(cost_hist_)
        best_hist.append(best_hist_)
        r.append(r_)
        best_solutions.append(solutions_)
    bv = torch.cat(bv, 0)
    cost_hist = torch.cat(cost_hist, 0)
    best_hist = torch.cat(best_hist, 0)
    r = torch.cat(r, 0)
    best_solutions = torch.cat(best_solutions, 0)  # Concatenate best solutions
        
    if distributed and opts.distributed: dist.barrier()
    
    if distributed and opts.distributed:
        initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
        time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())
    
    else:
        initial_cost = cost_hist[:,0] # bs
        time_used = torch.tensor([time.time() - s_time]) # bs
        bv = bv
        costs_history = cost_hist
        search_history = best_hist
        reward = r
        
    if distributed and opts.distributed: dist.barrier()
        
    # log to screen  
    if rank == 0: log_to_screen(time_used, 
                                  initial_cost, 
                                  bv, 
                                  reward, 
                                  costs_history,
                                  search_history,
                                  batch_size = opts.val_size, 
                                  dataset_size = len(val_dataset), 
                                  T = opts.T_max)
    
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
                      epoch = _id)
    
    # Print best solutions for each instance (only save to file in eval_only mode)
    if rank == 0 and opts.eval_only:
        print("\n" + "="*60)
        print("BEST SOLUTIONS FOUND:")
        print("="*60)
        
        # Create results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created results directory: {results_dir}")
        
        # Prepare results data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data = {
            "timestamp": timestamp,
            "problem": "pdtsp",
            "graph_size": opts.graph_size,
            "T_max": opts.T_max,
            "val_size": opts.val_size,
            "instances": []
        }
        
        # Load original coordinates for saving
        val_dataset_orig = problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.val_size,
            filename=val_dataset if isinstance(val_dataset, str) else None
        )
        
        for i in range(min(opts.val_size, len(best_solutions))):
            solution = best_solutions[i]
            cost = bv[i].item()
            coordinates = val_dataset_orig[i]['coordinates'].cpu().numpy().tolist()
            
            instance_data = {
                "instance_id": i,
                "best_cost": cost,
                "best_path": solution.cpu().numpy().tolist(),
                "path_length": len(solution),
                "coordinates": coordinates
            }
            
            # For PDTSP_2V: extract and save separate vehicle paths
            if opts.problem == 'pdtsp_2v':
                # Get vehicle assignment from problem
                batch_item = val_dataset_orig[i]
                batch_for_assignment = {
                    'coordinates': batch_item['coordinates'].unsqueeze(0)
                }
                vehicle_assignment, _, _ = problem.split_by_y_coordinate(batch_for_assignment)
                
                # Extract vehicle paths
                v1_path, v2_path = extract_vehicle_paths_pdtsp2v(
                    solution.cpu().numpy().tolist(),
                    vehicle_assignment[0]
                )
                
                # Calculate separate costs
                batch_for_cost = {
                    'coordinates': batch_item['coordinates'].unsqueeze(0),
                    'vehicle_assignment': vehicle_assignment
                }
                rec_for_cost = solution.unsqueeze(0)
                v1_cost, v2_cost, _ = problem.get_costs_separate(batch_for_cost, rec_for_cost)
                
                instance_data["vehicle_1_path"] = v1_path
                instance_data["vehicle_2_path"] = v2_path
                instance_data["vehicle_1_cost"] = v1_cost[0].item()
                instance_data["vehicle_2_cost"] = v2_cost[0].item()
                instance_data["vehicle_assignment"] = vehicle_assignment[0].cpu().numpy().tolist()
                
                print(f"\nInstance {i+1}:")
                print(f"  Best Cost: {cost:.6f}")
                print(f"  Vehicle 1 Cost: {v1_cost[0].item():.6f}, Nodes: {len(v1_path)}")
                print(f"  Vehicle 2 Cost: {v2_cost[0].item():.6f}, Nodes: {len(v2_path)}")
                print(f"  Vehicle 1 Path: {v1_path}")
                print(f"  Vehicle 2 Path: {v2_path}")
            else:
                # For other problems, keep adjacency format (visualization scripts handle conversion)
                print(f"\nInstance {i+1}:")
                print(f"  Best Cost: {cost:.6f}")
                print(f"  Best Path: {solution.cpu().numpy().tolist()}")
                print(f"  Path Length: {len(solution)}")
            
            results_data["instances"].append(instance_data)
        
        # Save results to JSON file
        results_file = os.path.join(results_dir, f"pdtsp_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # # Also save as pickle for easier loading
        # pickle_file = os.path.join(results_dir, f"pdtsp_results_{timestamp}.pkl")
        # with open(pickle_file, 'wb') as f:
        #     pickle.dump(results_data, f)
        # print(f"Results also saved to: {pickle_file}")
        
        print("="*60)
    
    if distributed and opts.distributed: dist.barrier()
    
