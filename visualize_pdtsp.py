#!/usr/bin/env python3
"""
PDTSP Solution Visualization Script

This script reads the saved results from the results directory and 
visualizes the PDTSP solution for the first instance.

Usage:
    python3 visualize_pdtsp.py [--results_file path_to_results.json]
"""

import os
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_latest_results(results_dir="results"):
    """Load the latest results file from the results directory."""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory '{results_dir}' not found!")
    
    # Find the latest JSON file
    json_files = list(Path(results_dir).glob("pdtsp_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No results files found in '{results_dir}'!")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def convert_n2s_path_to_visit_order(n2s_path):
    """
    Convert N2S pointing relationship path to actual visit order.
    
    In N2S format, the path represents pointing relationships:
    - Index i points to node n2s_path[i]
    - We need to follow these pointers starting from depot (node 0)
    
    Args:
        n2s_path: List or numpy array where index i points to node n2s_path[i]
        
    Returns:
        visit_order: List of nodes in the order they are visited
    """
    # Convert to numpy array if it isn't already
    n2s_path = np.array(n2s_path)
    
    if len(n2s_path) == 0:
        return []
    
    visit_order = []
    current_node = 0  # Start from depot
    visited = set()
    
    print(f"DEBUG: Converting N2S path to visit order...")
    print(f"DEBUG: N2S path: {n2s_path}")
    
    while current_node not in visited:
        visit_order.append(current_node)
        visited.add(current_node)
        
        # Follow the pointer from current node
        if current_node < len(n2s_path):
            next_node = n2s_path[current_node]
            print(f"DEBUG: Node {current_node} points to node {next_node}")
            current_node = next_node
        else:
            print(f"DEBUG: Node {current_node} is out of range, stopping")
            break
        
        # Safety check to prevent infinite loops
        if len(visit_order) > len(n2s_path) + 1:
            print("DEBUG: Detected potential infinite loop, breaking")
            break
    
    # Remove the depot from the end if we returned to it
    if len(visit_order) > 1 and visit_order[-1] == 0:
        visit_order = visit_order[1:-1]  # Remove starting and ending depot
    elif len(visit_order) > 0 and visit_order[0] == 0:
        visit_order = visit_order[1:]  # Remove only starting depot
    
    print(f"DEBUG: Final visit order: {visit_order}")
    return visit_order


def plot_pdtsp_solution(coordinates, solution, instance_id=0, cost=None, save_path=None):
    """
    Visualize PDTSP solution
    
    Args:
        coordinates: List of [x, y] coordinates for all nodes
        solution: N2S path format (pointing relationships, not visit order)
        instance_id: Instance identifier
        cost: Solution cost (optional)
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(14, 10))
    
    coordinates = np.array(coordinates)
    solution = np.array(solution)
    
    num_nodes = len(coordinates)
    depot_coord = coordinates[0]  # Depot (node 0)
    
    # Pickup points (nodes 1 to num_nodes//2)
    pickup_coords = coordinates[1:(num_nodes)//2 + 1]
    # Delivery points (nodes (num_nodes//2 + 1) to num_nodes-1)
    delivery_coords = coordinates[(num_nodes)//2 + 1:]
    
    # Plot depot
    plt.scatter(depot_coord[0], depot_coord[1], c='red', s=300, marker='s', 
               label='Depot', zorder=6, edgecolors='black', linewidth=2)
    plt.text(depot_coord[0], depot_coord[1] + 0.03, 'D', ha='center', va='bottom', 
             fontsize=14, fontweight='bold', color='white')
    plt.text(depot_coord[0], depot_coord[1] - 0.04, '0', ha='center', va='top', 
             fontsize=12, fontweight='bold', color='red')
    
    # Plot pickup points
    for i, coord in enumerate(pickup_coords):
        node_id = i + 1  # Pickup nodes start from 1
        plt.scatter(coord[0], coord[1], c='blue', s=200, marker='o', 
                   zorder=5, edgecolors='black', linewidth=1.5, alpha=0.8)
        plt.text(coord[0], coord[1] + 0.025, f'P{i+1}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='white')
        plt.text(coord[0], coord[1] - 0.035, str(node_id), ha='center', va='top', 
                fontsize=10, fontweight='bold', color='blue')
    
    # Plot delivery points
    for i, coord in enumerate(delivery_coords):
        node_id = i + (num_nodes)//2 + 1  # Delivery nodes start after pickup nodes
        plt.scatter(coord[0], coord[1], c='green', s=200, marker='^', 
                   zorder=5, edgecolors='black', linewidth=1.5, alpha=0.8)
        plt.text(coord[0], coord[1] + 0.025, f'D{i+1}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='white')
        plt.text(coord[0], coord[1] - 0.035, str(node_id), ha='center', va='top', 
                fontsize=10, fontweight='bold', color='green')
    
    # Draw pickup-delivery connections with dashed lines
    for i in range(len(pickup_coords)):
        pickup_coord = pickup_coords[i]
        delivery_coord = delivery_coords[i]
        plt.plot([pickup_coord[0], delivery_coord[0]], 
                [pickup_coord[1], delivery_coord[1]], 
                'gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)
    
    # Plot the route
    # Convert N2S path format to actual visit order
    print(f"DEBUG: Original N2S solution path: {solution}")
    visit_order = convert_n2s_path_to_visit_order(solution)
    print(f"DEBUG: Converted visit order: {visit_order}")
    print(f"DEBUG: Coordinates shape: {coordinates.shape}")
    print(f"DEBUG: Depot coord: {depot_coord}")
    
    # Create the complete path starting and ending at depot
    if len(visit_order) > 0:
        # Convert visit order to coordinates
        complete_path_coords = [depot_coord]  # Start at depot
        
        print(f"DEBUG: Processing visit order...")
        for i, node_id in enumerate(visit_order):
            if 0 <= node_id < len(coordinates):
                coord = coordinates[node_id]
                complete_path_coords.append(coord)
                # Determine node type for debugging
                if node_id == 0:
                    node_type = "Depot"
                elif 1 <= node_id <= (num_nodes)//2:
                    node_type = f"Pickup P{node_id}"
                else:
                    node_type = f"Delivery D{node_id - (num_nodes)//2}"
                print(f"  Step {i+1}: Node {node_id} ({node_type}) -> ({coord[0]:.6f}, {coord[1]:.6f})")
            else:
                print(f"Warning: Invalid node_id {node_id} in visit order")
        
        complete_path_coords.append(depot_coord)  # Return to depot
        complete_path_coords = np.array(complete_path_coords)
        
        print(f"DEBUG: Complete path has {len(complete_path_coords)} points")
        
        # Plot route segments
        for i in range(len(complete_path_coords) - 1):
            start = complete_path_coords[i]
            end = complete_path_coords[i + 1]
        
            # Calculate arrow position
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Draw the route line
            plt.plot([start[0], end[0]], [start[1], end[1]], 
                    'purple', linewidth=3, alpha=0.8, zorder=3)
            
            # Add arrow in the middle of the segment
            mid_x = start[0] + 0.7 * dx
            mid_y = start[1] + 0.7 * dy
            plt.arrow(mid_x - 0.1*dx, mid_y - 0.1*dy, 0.1*dx, 0.1*dy,
                     head_width=0.015, head_length=0.015, fc='purple', ec='purple',
                     alpha=0.9, zorder=4)
            
            # Add step number
            step_x = start[0] + 0.3 * dx
            step_y = start[1] + 0.3 * dy
            plt.text(step_x, step_y, str(i+1), ha='center', va='center', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='yellow', alpha=0.8, edgecolor='black'))
    else:
        print("Warning: Empty solution provided")
        visit_order = []
    
    # Create legend
    depot_patch = mpatches.Patch(color='red', label='Depot')
    pickup_patch = mpatches.Patch(color='blue', label='Pickup Points')
    delivery_patch = mpatches.Patch(color='green', label='Delivery Points')
    route_patch = mpatches.Patch(color='purple', label='Optimal Route')
    connection_patch = mpatches.Patch(color='gray', label='P-D Pairs', alpha=0.4)
    
    plt.legend(handles=[depot_patch, pickup_patch, delivery_patch, route_patch, connection_patch], 
              loc='upper right', fontsize=12)
    
    # Set title and labels
    title = f"PDTSP Solution - Instance {instance_id + 1}"
    if cost is not None:
        title += f"\nTotal Cost: {cost:.6f}"
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add route information as text - show both N2S format and visit order
    if visit_order:
        route_text = f"Visit Order: 0 → {' → '.join(map(str, visit_order))} → 0\nN2S Format: {solution}"
    else:
        route_text = f"N2S Format: {solution}"
    plt.figtext(0.5, 0.02, route_text, ha='center', fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize PDTSP solution')
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to specific results JSON file (default: latest in results/)')
    parser.add_argument('--instance_id', type=int, default=0,
                       help='Instance to visualize (default: 0)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the visualization (default: auto-generate)')
    
    args = parser.parse_args()
    
    # Load results
    if args.results_file:
        print(f"Loading results from: {args.results_file}")
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
    else:
        results_data = load_latest_results()
    
    # Check if instance exists
    if args.instance_id >= len(results_data['instances']):
        print(f"Error: Instance {args.instance_id} not found. Available instances: 0-{len(results_data['instances'])-1}")
        return
    
    # ⚠️ IMPORTANT FIX: Load coordinates from original dataset instead of results file
    # The coordinates in results file are incorrect due to data shuffling during N2S processing
    print("Loading original coordinates from dataset...")
    with open('./datasets/pdp_20.pkl', 'rb') as f:
        original_dataset = pickle.load(f)
    
    orig_instance = original_dataset[args.instance_id]
    orig_depot, orig_locations = orig_instance[0], orig_instance[1]
    coordinates = np.vstack([np.array(orig_depot).reshape(1, 2), np.array(orig_locations)])
    
    print(f"Using correct coordinates from original dataset:")
    print(f"  Depot: [{coordinates[0][0]:.6f}, {coordinates[0][1]:.6f}]")
    
    # Get solution and cost from results (these are correct)
    instance = results_data['instances'][args.instance_id]
    solution = instance['best_path']
    cost = instance['best_cost']
    
    print(f"\nVisualizing Instance {args.instance_id}:")
    print(f"  Best Cost: {cost:.6f}")
    print(f"  Path Length: {len(solution)}")
    print(f"  Best Path: {solution}")
    
    # Generate save path if not provided
    if args.save_path is None:
        timestamp = results_data['timestamp']
        args.save_path = f"results/pdtsp_visualization_instance_{args.instance_id}_{timestamp}.png"
    
    # Create visualization
    plot_pdtsp_solution(coordinates, solution, args.instance_id, cost, args.save_path)


if __name__ == "__main__":
    main()
