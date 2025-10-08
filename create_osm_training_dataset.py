#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate training dataset for PDTSP on OSM road networks.

This script pre-generates training data to avoid online generation overhead during training.
Training with pre-generated data can be 100-1000x faster than online generation.

Usage:
    python create_osm_training_dataset.py --graph_size 20 --num_samples 100000 --output ./datasets/osm_train_20_100k.pkl
"""

import argparse
import pickle
import os
from tqdm import tqdm
from data.osm_pdp_dataset import OSMOnlinePDPSDataset


def main():
    parser = argparse.ArgumentParser(description='Generate OSM training dataset')
    parser.add_argument('--graph_size', type=int, default=20, 
                        help='Graph size (number of pickup/delivery pairs)')
    parser.add_argument('--num_samples', type=int, default=100000, 
                        help='Number of samples to generate')
    parser.add_argument('--place', type=str, default='Boca Raton, Florida, USA',
                        help='OSM place string')
    parser.add_argument('--capacity', type=int, default=3,
                        help='Vehicle capacity')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path (e.g., ./datasets/osm_train_20_100k.pkl)')
    parser.add_argument('--disable_geo_aug', action='store_true',
                        help='Disable geometric augmentation')
    parser.add_argument('--multi_start', type=int, default=4,
                        help='Number of multi-start positions')
    
    args = parser.parse_args()
    
    print(f"Generating OSM training dataset:")
    print(f"  Place: {args.place}")
    print(f"  Graph size: {args.graph_size}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Capacity: {args.capacity}")
    print(f"  Output: {args.output}")
    print(f"  Seed: {args.seed}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create online dataset
    print("\nInitializing dataset generator...")
    dataset = OSMOnlinePDPSDataset(
        place=args.place,
        graph_size=args.graph_size,
        capacity=args.capacity,
        size=args.num_samples,
        seed=args.seed,
        disable_geo_aug=args.disable_geo_aug,
        multi_start=args.multi_start,
        pair_permute_aug=True
    )
    
    # Generate all samples
    print(f"\nGenerating {args.num_samples} samples...")
    print("This may take a while (minutes to hours depending on sample size)...")
    
    data_list = []
    for i in tqdm(range(args.num_samples), desc='Generating'):
        sample = dataset[i]
        
        # Convert tensors to numpy for storage
        sample_data = {
            'coordinates': sample['coordinates'].cpu().numpy(),
            'dist': sample['dist'].cpu().numpy(),
            'path_lookup': sample['path_lookup'],
            'node2osmid': sample['node2osmid'],
            'pairs': sample['pairs'],
            'capacity': sample['capacity'],
        }
        data_list.append(sample_data)
    
    # Save to pickle file
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data_list, f)
    
    # Print file size
    file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
    print(f"\n✓ Dataset saved successfully!")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Samples: {len(data_list)}")
    print(f"  Average size per sample: {file_size/len(data_list)*1024:.2f} KB")
    
    print(f"\nTo use this dataset for training, run:")
    print(f"  python run.py --problem pdtsp_osm --graph_size {args.graph_size} \\")
    print(f"    --train_dataset '{args.output}'")


if __name__ == "__main__":
    main()
