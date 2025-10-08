# problems/problem_pdtsp_osm.py
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
from torch.utils.data import Dataset
from problems.problem_pdtsp import PDTSP
from data.osm_pdp_dataset import OSMOnlinePDPSDataset, OSMFixedPDPSDataset

@dataclass
class PDTSP_OSM_Config:
    place: str = "Boca Raton, Florida, USA"
    capacity: int = 3
    seed: int = 2025
    disable_geo_aug: bool = True
    multi_start: int = 4
    pair_permute_aug: bool = True

class PDTSP_OSM(PDTSP):
    """
    PDTSP on real road network using OSM data.
    Inherits from PDTSP and overrides make_dataset and get_costs to support real road networks.
    """
    NAME = 'pdtsp_osm'
    
    def __init__(self, p_size, init_val_met='random', with_assert=False, osm_place="Boca Raton, Florida, USA", capacity=3):
        """
        Initialize PDTSP_OSM problem.
        
        Args:
            p_size: Number of nodes (customers + depot)
            init_val_met: Method to generate initial solutions ('random' or 'greedy')
            with_assert: Whether to enable assertions for feasibility checking
            osm_place: OSM place string for loading road network
            capacity: Vehicle capacity
        """
        super().__init__(p_size, init_val_met, with_assert)
        self.osm_place = osm_place
        self.capacity = capacity
        print(f'PDTSP_OSM with {self.size} nodes on {osm_place}.')
    
    def get_costs(self, batch, rec):
        """
        Calculate route costs using real road network distances if available.
        Falls back to Euclidean distance if 'dist' matrix is not in batch.
        
        Args:
            batch: Dictionary containing 'coordinates' and optionally 'dist' matrix
            rec: Route representation (successor matrix) [batch_size, size+1]
            
        Returns:
            Total route costs [batch_size]
        """
        batch_size, size = rec.size()
        
        # Check feasibility if enabled
        if self.do_assert:
            self.check_feasibility(rec)
        
        # If real road network distance matrix is available, use it
        if 'dist' in batch and batch['dist'] is not None:
            # dist is [batch_size, num_nodes, num_nodes] where num_nodes = size
            # rec contains indices into the coordinate/distance matrix
            d1_indices = rec.long()  # [batch_size, size]
            d2_indices = torch.arange(size, device=rec.device).unsqueeze(0).expand(batch_size, size)  # [batch_size, size]
            
            # Gather distances: dist[batch_idx, d2_indices[i], d1_indices[i]] gives distance from node i to its successor
            batch_indices = torch.arange(batch_size, device=rec.device).unsqueeze(1).expand(batch_size, size)
            distances = batch['dist'][batch_indices, d2_indices, d1_indices]  # [batch_size, size]
            
            return distances.sum(1)  # [batch_size]
        
        # Fallback to Euclidean distance (same as original PDTSP)
        d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
        d2 = batch['coordinates']
        length = (d1 - d2).norm(p=2, dim=2).sum(1)
        
        return length
    
    @staticmethod
    def make_dataset(*args, **kwargs):
        """
        Create dataset for PDTSP_OSM.
        
        This method handles two calling conventions:
        1. From ppo.py: make_dataset(size=graph_size, num_samples=epoch_size)
        2. Full specification: make_dataset(filename=..., size=..., num_samples=..., offset=...)
        
        Returns:
            Dataset instance (OSMOnlinePDPSDataset for training, OSMFixedPDPSDataset for validation)
        """
        import os
        
        # Extract parameters from kwargs
        filename = kwargs.get('filename', None)
        size = kwargs.get('size', 20)
        num_samples = kwargs.get('num_samples', 10000)
        offset = kwargs.get('offset', 0)
        distribution = kwargs.get('distribution', None)
        
        # OSM-specific parameters
        osm_place = kwargs.get('osm_place', "Boca Raton, Florida, USA")
        capacity = kwargs.get('capacity', 3)
        seed = kwargs.get('seed', 2025)
        disable_geo_aug = kwargs.get('disable_geo_aug', True)
        multi_start = kwargs.get('multi_start', 4)
        pair_permute_aug = kwargs.get('pair_permute_aug', True)
        
        # Check if filename exists and is valid
        if filename is not None and os.path.isfile(filename):
            # Validation/test mode: load from file
            return OSMFixedPDPSDataset(
                filename=filename,
                disable_geo_aug=disable_geo_aug,
                multi_start=multi_start,
                num_samples=num_samples,  # Pass num_samples to limit dataset size
                offset=offset,  # Pass offset for starting index
            )
        else:
            # Training/validation mode: generate online
            # (filename doesn't exist or not provided)
            return OSMOnlinePDPSDataset(
                place=osm_place,
                graph_size=size,
                capacity=capacity,
                size=num_samples,
                seed=seed,
                disable_geo_aug=disable_geo_aug,
                multi_start=multi_start,
                pair_permute_aug=pair_permute_aug,
            )
