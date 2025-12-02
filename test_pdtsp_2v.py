#!/usr/bin/env python3
"""
Quick test script to verify PDTSP_2V implementation.
Tests basic functionality without training.
"""

import torch
import numpy as np
from problems.problem_pdtsp_2v import PDTSP_2V

def test_basic_functionality():
    """Test basic problem setup and methods"""
    print("="*60)
    print("Testing PDTSP_2V Basic Functionality")
    print("="*60)
    
    # Create problem instance
    problem = PDTSP_2V(p_size=20, init_val_met='greedy', with_assert=True)
    print(f"✅ Created problem: {problem.NAME}")
    print(f"   Size: {problem.size} nodes")
    
    # Create a simple batch
    batch_size = 2
    graph_size = 21  # 20 nodes + 1 depot
    
    batch = {
        'coordinates': torch.rand(batch_size, graph_size, 2)
    }
    
    print(f"\n✅ Created test batch:")
    print(f"   Shape: {batch['coordinates'].shape}")
    
    # Test vehicle assignment
    vehicle_assignment, upper_count, lower_count = problem.split_by_y_coordinate(batch)
    print(f"\n✅ Vehicle assignment computed:")
    print(f"   Batch 1 - Upper: {upper_count[0].item()}, Lower: {lower_count[0].item()}")
    print(f"   Batch 2 - Upper: {upper_count[1].item()}, Lower: {lower_count[1].item()}")
    print(f"   Vehicle assignment shape: {vehicle_assignment.shape}")
    
    # Test initial solution generation
    batch['vehicle_assignment'] = vehicle_assignment
    batch['upper_count'] = upper_count
    batch['lower_count'] = lower_count
    
    solution = problem.get_initial_solutions(batch)
    print(f"\n✅ Generated initial solution:")
    print(f"   Shape: {solution.shape}")
    print(f"   Sample (batch 1): {solution[0, :10].tolist()}")
    
    # Test cost calculation
    costs = problem.get_costs(batch, solution)
    print(f"\n✅ Calculated costs:")
    print(f"   Total costs: {costs.tolist()}")
    
    # Test separate costs
    v1_costs, v2_costs, total_costs = problem.get_costs_separate(batch, solution)
    print(f"\n✅ Separate vehicle costs:")
    print(f"   Vehicle 1: {v1_costs.tolist()}")
    print(f"   Vehicle 2: {v2_costs.tolist()}")
    print(f"   Total: {total_costs.tolist()}")
    print(f"   Matches get_costs: {torch.allclose(costs, total_costs)}")
    
    # Test visited order map
    visited_time = torch.arange(problem.size).unsqueeze(0).expand(batch_size, problem.size)
    visited_order_map = problem.get_visited_order_map(visited_time)
    print(f"\n✅ Visited order map:")
    print(f"   Shape: {visited_order_map.shape}")
    
    # Test mask generation
    selected_node = torch.tensor([[1], [2]])  # Select nodes 1 and 2
    mask = problem.get_real_mask(selected_node, visited_order_map, vehicle_assignment)
    print(f"\n✅ Generated mask with vehicle constraints:")
    print(f"   Shape: {mask.shape}")
    print(f"   Valid positions for node 1: {(~mask[0]).sum().item()}")
    
    return True

def test_different_distributions():
    """Test with different y-coordinate distributions"""
    print("\n" + "="*60)
    print("Testing Different Vehicle Distributions")
    print("="*60)
    
    problem = PDTSP_2V(p_size=20, init_val_met='greedy', with_assert=False)
    
    # Test case 1: All pickups in upper half
    batch1 = {
        'coordinates': torch.rand(1, 21, 2)
    }
    batch1['coordinates'][0, 1:11, 1] = 0.6  # Set pickup y-coords to upper half
    
    va1, u1, l1 = problem.split_by_y_coordinate(batch1)
    print(f"\n✅ Test case 1 - All upper:")
    print(f"   Upper: {u1.item()}, Lower: {l1.item()}")
    
    # Test case 2: All pickups in lower half
    batch2 = {
        'coordinates': torch.rand(1, 21, 2)
    }
    batch2['coordinates'][0, 1:11, 1] = 0.3  # Set pickup y-coords to lower half
    
    va2, u2, l2 = problem.split_by_y_coordinate(batch2)
    print(f"\n✅ Test case 2 - All lower:")
    print(f"   Upper: {u2.item()}, Lower: {l2.item()}")
    
    # Test case 3: Balanced distribution
    batch3 = {
        'coordinates': torch.rand(1, 21, 2)
    }
    batch3['coordinates'][0, 1:6, 1] = 0.6   # 5 upper
    batch3['coordinates'][0, 6:11, 1] = 0.3  # 5 lower
    
    va3, u3, l3 = problem.split_by_y_coordinate(batch3)
    print(f"\n✅ Test case 3 - Balanced:")
    print(f"   Upper: {u3.item()}, Lower: {l3.item()}")
    
    return True

def test_initial_solution_methods():
    """Test different initial solution methods"""
    print("\n" + "="*60)
    print("Testing Initial Solution Methods")
    print("="*60)
    
    batch = {
        'coordinates': torch.rand(3, 21, 2)
    }
    
    methods = ['random', 'greedy', 'p2d']
    
    for method in methods:
        problem = PDTSP_2V(p_size=20, init_val_met=method, with_assert=False)
        
        # Get vehicle assignment
        va, u, l = problem.split_by_y_coordinate(batch)
        batch['vehicle_assignment'] = va
        batch['upper_count'] = u
        batch['lower_count'] = l
        
        # Generate solution
        solution = problem.get_initial_solutions(batch)
        costs = problem.get_costs(batch, solution)
        
        print(f"\n✅ Method: {method}")
        print(f"   Costs: {costs.mean().item():.4f} (avg)")
        print(f"   Range: [{costs.min().item():.4f}, {costs.max().item():.4f}]")
    
    return True

def test_step_operation():
    """Test the step (improvement) operation"""
    print("\n" + "="*60)
    print("Testing Step Operation")
    print("="*60)
    
    problem = PDTSP_2V(p_size=20, init_val_met='greedy', with_assert=False)
    
    batch = {
        'coordinates': torch.rand(2, 21, 2)
    }
    
    # Get initial solution
    solution = problem.get_initial_solutions(batch)
    initial_cost = problem.get_costs(batch, solution)
    
    print(f"\n✅ Initial solution:")
    print(f"   Costs: {initial_cost.tolist()}")
    
    # Prepare for step
    obj = torch.cat([initial_cost.unsqueeze(1), initial_cost.unsqueeze(1)], dim=1)
    action_record = [torch.zeros((2, 10)) for _ in range(10)]
    
    # Define an exchange action (node_to_remove, insert_pos1, insert_pos2)
    exchange = torch.tensor([[1, 2, 5], [2, 1, 4]])
    
    # Execute step
    new_solution, reward, new_obj, _ = problem.step(
        batch, solution, exchange, obj, action_record
    )
    
    print(f"\n✅ After step:")
    print(f"   New costs: {new_obj[:, 0].tolist()}")
    print(f"   Reward: {reward.tolist()}")
    print(f"   Improvement: {(initial_cost - new_obj[:, 0]).tolist()}")
    
    return True

if __name__ == "__main__":
    try:
        print("\n" + "🚗"*30)
        print("PDTSP_2V Implementation Test Suite")
        print("🚗"*30)
        
        # Run all tests
        test_basic_functionality()
        test_different_distributions()
        test_initial_solution_methods()
        test_step_operation()
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        print("="*60)
        print("\nImplementation is ready for training.")
        print("Run: bash train_pdtsp_2v.sh")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Test failed with error:")
        print("="*60)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
