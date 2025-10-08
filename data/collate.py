# data/collate.py
"""
Custom collate function for OSM-based datasets.
Handles non-tensor fields like path_lookup, node2osmid, pairs.
"""
import torch
from typing import List, Dict, Any


def osm_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for OSM datasets.
    
    Handles:
    - coordinates: stack into tensor [batch_size, num_nodes, 2]
    - dist: stack into tensor [batch_size, num_nodes, num_nodes]
    - path_lookup: keep as list of dicts (cannot be batched)
    - node2osmid: keep as list of lists
    - pairs: keep as list of lists
    - capacity, multi_start, disable_geo_aug: take first value (should be same for all)
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Batched dictionary
    """
    if len(batch) == 0:
        return {}
    
    # Separate tensor fields from non-tensor fields
    coordinates = torch.stack([item['coordinates'] for item in batch])
    dist = torch.stack([item['dist'] for item in batch])
    
    # Non-tensor fields that vary per instance
    path_lookup = [item.get('path_lookup', None) for item in batch]
    node2osmid = [item.get('node2osmid', None) for item in batch]
    pairs = [item.get('pairs', None) for item in batch]
    
    # Scalar fields (same for all items in batch)
    result = {
        'coordinates': coordinates,
        'dist': dist,
    }
    
    # Only add path_lookup if it exists (for training with real routing)
    if any(pl is not None for pl in path_lookup):
        result['path_lookup'] = path_lookup
    
    if any(n2o is not None for n2o in node2osmid):
        result['node2osmid'] = node2osmid
        
    if any(p is not None for p in pairs):
        result['pairs'] = pairs
    
    # Add scalar fields if they exist
    if 'capacity' in batch[0]:
        result['capacity'] = batch[0]['capacity']
    if 'multi_start' in batch[0]:
        result['multi_start'] = batch[0]['multi_start']
    if 'disable_geo_aug' in batch[0]:
        result['disable_geo_aug'] = batch[0]['disable_geo_aug']
    
    return result


def pdp_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard collate function for regular PDP datasets (non-OSM).
    Uses PyTorch's default collate behavior.
    """
    # For standard PDP, just stack coordinates
    coordinates = torch.stack([item['coordinates'] for item in batch])
    return {'coordinates': coordinates}
