#!/usr/bin/env python3
"""
MVPDTSP Solution Visualization Script

This script reads saved results from the results directory and
visualizes the MVPDTSP (multi-vehicle) solution for instances.

Usage:
    python3 vis_mapdtsp.py [--results_file path_to_results.json] [--instance_id N]
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def load_latest_results(results_dir="results"):
    """Load the latest MVPDTSP results file from the results directory."""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory '{results_dir}' not found!")
    
    # Find the latest JSON file for MVPDTSP
    json_files = list(Path(results_dir).glob("mvpdtsp_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No MVPDTSP results files found in '{results_dir}'!")
    
    latest_file = max(json_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_mvpdtsp_solution(coordinates, vehicle_routes,
                          instance_id=0, total_cost=None, save_path=None):
    """Visualize MVPDTSP solution with multiple vehicles."""
    plt.figure(figsize=(16, 12))
    
    coordinates = np.array(coordinates)
    
    total_nodes = len(coordinates)
    num_vehicles = len(vehicle_routes)
    num_pairs = (total_nodes - num_vehicles) // 2
    pickup_start = num_vehicles
    delivery_start = num_vehicles + num_pairs

    depot_coords = coordinates[:num_vehicles]
    pickup_coords = coordinates[pickup_start:delivery_start]
    delivery_coords = coordinates[delivery_start:]

    # Plot depots
    for v in range(num_vehicles):
        depot_coord = depot_coords[v]
        plt.scatter(depot_coord[0], depot_coord[1], c='red', s=350, marker='s',
                    label='Depot' if v == 0 else None, zorder=10, edgecolors='black', linewidth=2)
        plt.text(depot_coord[0], depot_coord[1], f'D\n{v}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white')
    
    # Plot pickup points
    for i, coord in enumerate(pickup_coords):
        node_id = i + pickup_start
        plt.scatter(coord[0], coord[1], c='blue', s=200, marker='o', 
                   zorder=5, edgecolors='black', linewidth=1.5, alpha=0.8)
        plt.text(coord[0], coord[1] + 0.025, f'P{i+1}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='white')
        plt.text(coord[0], coord[1] - 0.035, str(node_id), ha='center', va='top', 
                fontsize=9, fontweight='bold', color='blue')
    
    # Plot delivery points
    for i, coord in enumerate(delivery_coords):
        node_id = i + delivery_start
        plt.scatter(coord[0], coord[1], c='darkviolet', s=200, marker='^', 
                   zorder=5, edgecolors='black', linewidth=1.5, alpha=0.8)
        plt.text(coord[0], coord[1] + 0.025, f'D{i+1}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='white')
        plt.text(coord[0], coord[1] - 0.035, str(node_id), ha='center', va='top', 
                fontsize=9, fontweight='bold', color='darkviolet')
    
    # Draw pickup-delivery connections with dashed lines
    for i in range(len(pickup_coords)):
        pickup_coord = pickup_coords[i]
        delivery_coord = delivery_coords[i]
        plt.plot([pickup_coord[0], delivery_coord[0]], 
                [pickup_coord[1], delivery_coord[1]], 
                'gray', linestyle='--', alpha=0.3, linewidth=1, zorder=1)
    
    # Function to plot a vehicle's route
    def plot_vehicle_route(path, color, vehicle_name, zorder_base):
        if not path:
            print(f"Warning: {vehicle_name} has empty path")
            return

        complete_path_coords = []
        print(f"\n{vehicle_name} route:")
        print(f"  Starting from node {path[0]}")
        
        for i, node_id in enumerate(path):
            if 0 <= node_id < len(coordinates):
                coord = coordinates[node_id]
                complete_path_coords.append(coord)
                
                # Determine node type
                if node_id < num_vehicles:
                    node_type = f"Depot D{node_id}"
                elif pickup_start <= node_id < delivery_start:
                    node_type = f"Pickup P{node_id - pickup_start + 1}"
                else:
                    node_type = f"Delivery D{node_id - delivery_start + 1}"
                print(f"  Step {i+1}: Node {node_id} ({node_type})")
            else:
                print(f"  Warning: Invalid node_id {node_id} in {vehicle_name}")

        complete_path_coords = np.array(complete_path_coords)
        
        # Plot route segments
        for i in range(len(complete_path_coords) - 1):
            start = complete_path_coords[i]
            end = complete_path_coords[i + 1]
            
            # Calculate direction
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Draw the route line
            plt.plot([start[0], end[0]], [start[1], end[1]],
                     color=color, linewidth=3, alpha=0.7, zorder=zorder_base)
            
            # Add arrow in the middle of the segment (skip zero-length)
            if dx != 0 or dy != 0:
                mid_x = start[0] + 0.7 * dx
                mid_y = start[1] + 0.7 * dy
                plt.arrow(mid_x - 0.1*dx, mid_y - 0.1*dy, 0.1*dx, 0.1*dy,
                          head_width=0.02, head_length=0.02, fc=color, ec=color,
                          alpha=0.9, zorder=zorder_base + 1)
            
            # Add step number
            step_x = start[0] + 0.3 * dx
            step_y = start[1] + 0.3 * dy
            plt.text(step_x, step_y, str(i+1), ha='center', va='center', 
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                             alpha=0.85, edgecolor=color, linewidth=2))
    
    colors = ['green', 'darkorange', 'dodgerblue', 'crimson', 'goldenrod', 'purple']
    print("\n" + "="*60)
    for v, path in enumerate(vehicle_routes):
        color = colors[v % len(colors)]
        plot_vehicle_route(path, color, f"Vehicle {v}", 3 + v * 2)
        print("="*60)
    
    # Create legend
    depot_patch = mpatches.Patch(color='red', label='Depot')
    pickup_patch = mpatches.Patch(color='blue', label='Pickup Points')
    delivery_patch = mpatches.Patch(color='darkviolet', label='Delivery Points')
    vehicle_patches = [mpatches.Patch(color=colors[v % len(colors)], label=f'Vehicle {v} Route')
                       for v in range(num_vehicles)]
    connection_patch = mpatches.Patch(color='gray', label='P-D Pairs', alpha=0.3)
    
    plt.legend(handles=[depot_patch, pickup_patch, delivery_patch, 
                       *vehicle_patches, connection_patch], 
              loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set title and labels
    title = f"MVPDTSP Solution - Instance {instance_id + 1}"
    if total_cost is not None:
        title += f"\nTotal Cost: {total_cost:.6f}"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add route information as text
    route_lines = [f"Vehicle {v}: {' → '.join(map(str, vehicle_routes[v]))}" for v in range(num_vehicles)]
    route_text = "\n".join(route_lines)
    
    plt.figtext(0.5, 0.02, route_text, ha='center', fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
               wrap=True)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize MVPDTSP solution')
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
    
    # Get instance data
    instance = results_data['instances'][args.instance_id]
    
    coordinates = instance.get('coordinates')
    if coordinates is None:
        print("Error: No coordinates found in results file.")
        return

    vehicle_routes = instance.get('vehicle_routes')
    if vehicle_routes is None:
        print("Error: No vehicle_routes found in results file.")
        return
    total_cost = instance.get('best_cost')
    
    print(f"\n{'='*70}")
    print(f"Visualizing PDTSP_2V Instance {args.instance_id}:")
    print(f"{'='*70}")
    print(f"Total Cost: {total_cost:.6f}")
    for v, route in enumerate(vehicle_routes):
        print(f"Vehicle {v} visits {len(route)} nodes: {route}")
    
    # Generate save path if not provided
    if args.save_path is None:
        timestamp = results_data.get('timestamp', 'unknown')
        args.save_path = f"results/mvpdtsp_visualization_instance_{args.instance_id}_{timestamp}.png"
    
    # Create visualization
    plot_mvpdtsp_solution(
        coordinates,
        vehicle_routes,
        args.instance_id,
        total_cost,
        args.save_path
    )


if __name__ == "__main__":
    main()
