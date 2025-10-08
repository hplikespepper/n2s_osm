#!/usr/bin/env python
"""
Generate OSM validation dataset for testing.
"""
import pickle
import torch
from data.osm_pdp_dataset import OSMOnlinePDPSDataset

def create_val_dataset(graph_size=20, num_samples=10, place="Boca Raton, Florida, USA", output_file='./datasets/osm_val.pkl'):
    """Create a validation dataset and save to file."""
    
    print(f"Creating OSM validation dataset...")
    print(f"  Graph size: {graph_size}")
    print(f"  Samples: {num_samples}")
    print(f"  Place: {place}")
    print(f"  Output: {output_file}")
    
    # Create dataset
    dataset = OSMOnlinePDPSDataset(
        place=place,
        graph_size=graph_size,
        capacity=3,
        size=num_samples,
        seed=2025,
        disable_geo_aug=True,
        multi_start=4,
        pair_permute_aug=False  # No augmentation for validation
    )
    
    # Generate all samples
    data = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Convert tensors to numpy for pickle
        sample_dict = {
            'coordinates': sample['coordinates'].numpy(),
            'dist': sample['dist'].numpy(),
            'path_lookup': sample['path_lookup'],
            'node2osmid': sample['node2osmid'],
            'pairs': sample['pairs'],
            'capacity': sample['capacity'],
            'multi_start': sample['multi_start'],
            'disable_geo_aug': sample['disable_geo_aug'],
        }
        data.append(sample_dict)
        print(f"  Generated sample {i+1}/{num_samples}")
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Validation dataset saved to {output_file}")
    print(f"  Total samples: {len(data)}")
    print(f"  Coordinates shape: {data[0]['coordinates'].shape}")
    print(f"  Distance matrix shape: {data[0]['dist'].shape}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_size', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--place', type=str, default='Boca Raton, Florida, USA')
    parser.add_argument('--output', type=str, default='./datasets/osm_val.pkl')
    args = parser.parse_args()
    
    create_val_dataset(args.graph_size, args.num_samples, args.place, args.output)
