#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize PDTSP solutions on real road networks using OSMnx.

Usage:
    python vis_osm.py --results results/pdtsp_results_20251007_171709.json --index 0
"""

import argparse
import json
import pickle
import os
import matplotlib.pyplot as plt
import osmnx as ox
from data.osm_utils import build_drive_graph

# ==================== Configuration ====================
# Control which instance to visualize
INSTANCE_INDEX = 0  # Change this to visualize different instances

# OSM place (should match the training data)
OSM_PLACE = "Boca Raton, Florida, USA"

# Paths
RESULTS_FILE = "results/pdtsp_results_20251007_171709.json"
VAL_DATASET_FILE = "./datasets/osm_val_20.pkl"
OUTPUT_DIR = "visualizations"
# ======================================================


def plot_solution_on_osm(G, sample, solution_path, coordinates, best_cost, instance_id, output_path):
    """
    Plot a PDTSP solution on the OSM road network.
    
    Args:
        G: OSMnx graph
        sample: Dataset sample containing path_lookup and node2osmid
        solution_path: List of node indices representing the solution
        coordinates: Normalized coordinates [num_nodes, 2]
        best_cost: Best cost of the solution
        instance_id: Instance ID
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the base road network
    ox.plot_graph(G, node_size=2, edge_color='lightgray', edge_linewidth=0.5,
                  show=False, close=False, ax=ax)
    
    # Convert solution path to OSM node IDs and reconstruct the route
    osmid_seq = []
    for u, v in zip(solution_path[:-1], solution_path[1:]):
        # Check if path is precomputed in path_lookup
        if ('path_lookup' in sample) and ((u, v) in sample['path_lookup']):
            seg = sample['path_lookup'][(u, v)]
        else:
            # Compute shortest path if not in lookup
            osmu = sample['node2osmid'][u]
            osmv = sample['node2osmid'][v]
            seg = ox.distance.shortest_path(G, osmu, osmv, weight='length')
        
        # Append segment, avoiding duplicate nodes
        if osmid_seq and seg and seg[0] == osmid_seq[-1]:
            osmid_seq += seg[1:]
        else:
            osmid_seq += seg
    
    # Plot the solution route
    xs = [G.nodes[n]['x'] for n in osmid_seq]
    ys = [G.nodes[n]['y'] for n in osmid_seq]
    ax.plot(xs, ys, linewidth=3, alpha=0.8, color='red', label='Solution Route', zorder=3)
    
    # Plot depot (node 0) as a green star
    depot_osmid = sample['node2osmid'][0]
    depot_x = G.nodes[depot_osmid]['x']
    depot_y = G.nodes[depot_osmid]['y']
    ax.scatter(depot_x, depot_y, s=300, c='green', marker='*', 
               edgecolors='black', linewidths=2, label='Depot', zorder=5)
    
    # Plot pickup/delivery nodes
    num_pairs = (len(solution_path) - 1) // 2
    for i in range(1, num_pairs + 1):
        # Pickup node
        pickup_osmid = sample['node2osmid'][i]
        pickup_x = G.nodes[pickup_osmid]['x']
        pickup_y = G.nodes[pickup_osmid]['y']
        ax.scatter(pickup_x, pickup_y, s=150, c='blue', marker='o',
                   edgecolors='black', linewidths=1.5, alpha=0.8, zorder=4)
        ax.text(pickup_x, pickup_y, f'P{i}', fontsize=8, ha='center', va='center',
                color='white', weight='bold', zorder=6)
        
        # Delivery node
        delivery_osmid = sample['node2osmid'][i + num_pairs]
        delivery_x = G.nodes[delivery_osmid]['x']
        delivery_y = G.nodes[delivery_osmid]['y']
        ax.scatter(delivery_x, delivery_y, s=150, c='orange', marker='s',
                   edgecolors='black', linewidths=1.5, alpha=0.8, zorder=4)
        ax.text(delivery_x, delivery_y, f'D{i}', fontsize=8, ha='center', va='center',
                color='white', weight='bold', zorder=6)
    
    # Add title and legend
    ax.set_title(f'Instance {instance_id} - Best Cost: {best_cost:.2f}\n'
                 f'Route Length: {len(solution_path)} nodes',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main(args):
    # Load results
    print(f"Loading results from: {args.results}")
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Load validation dataset
    print(f"Loading validation dataset from: {args.val_dataset}")
    with open(args.val_dataset, 'rb') as f:
        val_dataset = pickle.load(f)
    
    # Check if instance index is valid
    instance_idx = args.index
    if instance_idx >= len(results['instances']):
        print(f"Error: Instance index {instance_idx} out of range. "
              f"Available instances: 0-{len(results['instances'])-1}")
        return
    
    if instance_idx >= len(val_dataset):
        print(f"Error: Instance index {instance_idx} out of range in validation dataset. "
              f"Available samples: 0-{len(val_dataset)-1}")
        return
    
    # Get instance data
    instance = results['instances'][instance_idx]
    sample = val_dataset[instance_idx]
    
    print(f"\nVisualizing Instance {instance_idx}:")
    print(f"  Best Cost: {instance['best_cost']}")
    print(f"  Path Length: {instance['path_length']}")
    print(f"  Best Path: {instance['best_path']}")
    
    # Build OSM graph
    print(f"\nLoading OSM road network for: {args.osm_place}")
    G = build_drive_graph(args.osm_place)
    print(f"  Graph nodes: {len(G.nodes)}")
    print(f"  Graph edges: {len(G.edges)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    output_file = os.path.join(args.output_dir, 
                               f"instance_{instance_idx}_cost_{instance['best_cost']:.0f}.png")
    
    # Plot solution
    print(f"\nGenerating visualization...")
    plot_solution_on_osm(
        G=G,
        sample=sample,
        solution_path=instance['best_path'],
        coordinates=instance['coordinates'],
        best_cost=instance['best_cost'],
        instance_id=instance_idx,
        output_path=output_file
    )
    
    print(f"\n✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize PDTSP solutions on OSM road networks')
    parser.add_argument('--results', type=str, default=RESULTS_FILE,
                        help='Path to results JSON file')
    parser.add_argument('--val_dataset', type=str, default=VAL_DATASET_FILE,
                        help='Path to validation dataset pickle file')
    parser.add_argument('--index', type=int, default=INSTANCE_INDEX,
                        help='Instance index to visualize')
    parser.add_argument('--osm_place', type=str, default=OSM_PLACE,
                        help='OSM place name (should match training data)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    main(args)
