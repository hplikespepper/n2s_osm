"""Multi-vehicle PDP on directed OSM road distances, with one physical depot."""
import torch
from problems.problem_mvpdtsp import MVPDTSP
from data.osm_mvpdp_dataset import OSMMVPDPDataset


class MVPDTSP_OSM(MVPDTSP):
    NAME = 'mvpdtsp_osm'
    IS_OSM = True

    def __init__(self, p_size, num_vehicles=2, init_val_met='random', with_assert=False,
                 capacity=None, use_makespan=False, objective=None,
                 osm_place='Boca Raton, Florida, USA'):
        super().__init__(p_size, num_vehicles, init_val_met, with_assert,
                         capacity, use_makespan, objective)
        self.osm_place = osm_place

    def _get_route_lengths(self, batch, rec):
        distances = batch.get('dist')
        bs, total = rec.shape
        if distances is None:
            raise ValueError('mvpdtsp_osm requires dist; Euclidean fallback is disabled')
        if distances.shape != (bs, self.size + self.num_vehicles, self.size + self.num_vehicles) or total != self.size + self.num_vehicles:
            raise ValueError('dist/rec dimensions do not match the multi-vehicle problem')
        arange = torch.arange(bs, device=rec.device)
        lengths = distances.new_zeros((bs, self.num_vehicles))
        for v in range(self.num_vehicles):
            current = torch.full((bs,), v, device=rec.device, dtype=torch.long)
            active = torch.ones(bs, device=rec.device, dtype=torch.bool)
            for _ in range(total):
                nxt = rec[arange, current]
                lengths[:, v] += distances[arange, current, nxt] * active.to(distances.dtype)
                current = nxt
                active &= current != v
                if not active.any():
                    break
            if active.any():
                raise ValueError(f'Vehicle {v} route does not return to its logical depot')
        return lengths

    @staticmethod
    def make_dataset(*args, **kwargs):
        return OSMMVPDPDataset(*args, **kwargs)
