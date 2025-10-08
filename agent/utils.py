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
            results_data["instances"].append(instance_data)
            
            print(f"\nInstance {i+1}:")
            print(f"  Best Cost: {cost:.6f}")
            print(f"  Best Path: {solution.cpu().numpy().tolist()}")
            print(f"  Path Length: {len(solution)}")
        
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
    
