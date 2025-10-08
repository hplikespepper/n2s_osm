# data/osm_pdp_dataset.py
import pickle, torch, numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from data.osm_utils import build_drive_graph, sample_task_nodes, project_and_scale_xy, compute_apsp_for_subset
from utils.augment import apply_pair_permutation

class OSMOnlinePDPSDataset(Dataset):
    def __init__(self, place: str, graph_size: int, capacity: int, size: int,
                 seed: int = 2025, disable_geo_aug: bool = True,
                 multi_start: int = 4, pair_permute_aug: bool = True):
        assert graph_size % 2 == 0 and graph_size > 0
        self.place = place
        self.graph_size = graph_size
        self.n_pairs = graph_size // 2
        self.capacity = capacity
        self.size = size
        self.disable_geo_aug = disable_geo_aug
        self.multi_start = multi_start
        self.pair_permute_aug = pair_permute_aug
        self.G = build_drive_graph(place)
        self.rng = np.random.default_rng(seed)

    def __len__(self): return self.size

    def __getitem__(self, idx):
        node2osmid, pairs = sample_task_nodes(self.G, self.n_pairs, self.rng)
        coordinates = project_and_scale_xy(self.G, node2osmid)
        dist, path_lookup = compute_apsp_for_subset(self.G, node2osmid)

        if self.pair_permute_aug:
            coordinates, dist, path_lookup, node2osmid, pairs = apply_pair_permutation(
                coordinates, dist, path_lookup, node2osmid, pairs, self.rng
            )

        return {
            "coordinates": torch.tensor(coordinates, dtype=torch.float32),
            "dist": torch.tensor(dist, dtype=torch.float32),
            "path_lookup": path_lookup,
            "node2osmid": node2osmid,
            "pairs": pairs,
            "capacity": self.capacity,
            "multi_start": self.multi_start,
            "disable_geo_aug": self.disable_geo_aug,
        }

class OSMFixedPDPSDataset(Dataset):
    def __init__(self, filename: str, disable_geo_aug: bool = True, multi_start: int = 4, 
                 num_samples: int = None, offset: int = 0):
        """
        Dataset for loading fixed OSM PDPS instances from file.
        
        Args:
            filename: Path to pickle file containing dataset
            disable_geo_aug: Whether to disable geometric augmentation
            multi_start: Number of starting positions for multi-start search
            num_samples: Number of samples to load (if None, load all)
            offset: Starting index for loading samples
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        # Slice data based on num_samples and offset (similar to PDPDataset)
        if num_samples is not None:
            self.data = data[offset:offset+num_samples]
        else:
            self.data = data[offset:]
            
        self.disable_geo_aug = disable_geo_aug
        self.multi_start = multi_start

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        s["disable_geo_aug"] = self.disable_geo_aug
        s["multi_start"] = self.multi_start
        s["coordinates"] = torch.tensor(s["coordinates"], dtype=torch.float32)
        s["dist"] = torch.tensor(s["dist"], dtype=torch.float32)
        return s
