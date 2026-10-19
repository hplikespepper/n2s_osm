"""Adapt single-physical-depot OSM samples without changing stored datasets."""
from numbers import Integral
from pathlib import Path
import pickle
import torch
from torch.utils.data import Dataset
from data.osm_pdp_dataset import OSMOnlinePDPSDataset


def expand_single_depot(sample, num_vehicles, size):
    if not isinstance(num_vehicles, Integral) or num_vehicles < 1:
        raise ValueError('num_vehicles must be a positive integer')
    if size <= 0 or size % 2:
        raise ValueError('size must be a positive even customer count')
    coords = torch.as_tensor(sample['coordinates'], dtype=torch.float32)
    if 'dist' not in sample or sample['dist'] is None:
        raise ValueError('OSM sample requires dist')
    distances = torch.as_tensor(sample['dist'], dtype=torch.float32)
    n = size + 1
    if coords.shape != (n, 2) or distances.shape != (n, n):
        raise ValueError(f'Expected single-depot coordinates {(n, 2)} and dist {(n, n)}')
    if not torch.isfinite(coords).all() or not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Coordinates/dist must be finite, and distances non-negative')
    if not torch.equal(distances.diag(), torch.zeros(n, device=distances.device)):
        raise ValueError('OSM dist diagonal must be zero')
    mapping = sample['node2osmid']
    if len(mapping) != n:
        raise ValueError('node2osmid length does not match customer count')
    expected = [(i, i + size // 2) for i in range(1, size // 2 + 1)]
    if [tuple(pair) for pair in sample['pairs']] != expected:
        raise ValueError('pairs must follow pickup-then-delivery index layout')
    index_map = [0] * num_vehicles + list(range(1, n))
    idx = torch.tensor(index_map, device=coords.device)
    didx = idx.to(distances.device)
    result = dict(sample)
    result['coordinates'] = coords.index_select(0, idx)
    result['dist'] = distances.index_select(0, didx).index_select(1, didx)
    result['node2osmid'] = [mapping[i] for i in index_map]
    result['pairs'] = [(p + num_vehicles - 1, d + num_vehicles - 1) for p, d in expected]
    lookup = sample.get('path_lookup') or {}
    expanded = {}
    for i, u in enumerate(index_map):
        for j, v in enumerate(index_map):
            if mapping[u] == mapping[v]:
                expanded[i, j] = [mapping[u]]
            elif (u, v) in lookup:
                expanded[i, j] = list(lookup[u, v])
    result['path_lookup'] = expanded
    return result


class OSMMVPDPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0,
                 num_vehicles=2, osm_place='Boca Raton, Florida, USA', capacity=3,
                 seed=2025, disable_geo_aug=True, multi_start=4, pair_permute_aug=True,
                 distribution=None):
        if not isinstance(num_vehicles, Integral) or num_vehicles < 1:
            raise ValueError('num_vehicles must be a positive integer')
        if size <= 0 or size % 2:
            raise ValueError('size must be a positive even customer count')
        if offset < 0 or (num_samples is not None and num_samples < 1):
            raise ValueError('offset must be non-negative and num_samples positive')
        self.size, self.num_vehicles, self.offset = size, num_vehicles, offset
        self.filename = str(Path(filename).resolve()) if filename is not None else None
        if filename is not None:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.source = data[offset:] if num_samples is None else data[offset:offset + num_samples]
        else:
            if num_samples is None or offset:
                raise ValueError('Online generation requires num_samples and offset=0')
            self.source = OSMOnlinePDPSDataset(
                place=osm_place, graph_size=size, capacity=capacity, size=num_samples,
                seed=seed, disable_geo_aug=disable_geo_aug, multi_start=multi_start,
                pair_permute_aug=pair_permute_aug)
        self.disable_geo_aug, self.multi_start = disable_geo_aug, multi_start
        if not len(self.source):
            raise ValueError('Dataset selection is empty')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = expand_single_depot(self.source[index], self.num_vehicles, self.size)
        sample['disable_geo_aug'] = self.disable_geo_aug
        sample['multi_start'] = self.multi_start
        return sample
