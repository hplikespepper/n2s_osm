# utils/augment.py
import numpy as np
from typing import List, Tuple, Dict

def apply_pair_permutation(
    coordinates, dist, path_lookup, node2osmid, pairs, rng: np.random.Generator
):
    N1 = coordinates.shape[0]
    assert (N1 - 1) % 2 == 0
    n_pairs = (N1 - 1) // 2
    order = rng.permutation(n_pairs)

    new_idx = {0: 0}
    for new_k, old_k in enumerate(order):
        old_p, old_d = 1 + old_k, 1 + n_pairs + old_k
        new_p, new_d = 1 + new_k, 1 + n_pairs + new_k
        new_idx[old_p] = new_p; new_idx[old_d] = new_d

    new_coords = coordinates.copy()
    for old, new in list(new_idx.items())[1:]:
        new_coords[new] = coordinates[old]

    new_dist = dist.copy()
    for (i, j), _ in np.ndenumerate(dist):
        ii = new_idx.get(i, i); jj = new_idx.get(j, j)
        new_dist[ii, jj] = dist[i, j]

    new_path = {}
    for (i, j), path in path_lookup.items():
        ii = new_idx.get(i, i); jj = new_idx.get(j, j)
        new_path[(ii, jj)] = path

    new_node2osmid = [None] * N1
    new_node2osmid[0] = node2osmid[0]
    for old, new in list(new_idx.items())[1:]:
        new_node2osmid[new] = node2osmid[old]

    new_pairs = [(1 + k, 1 + n_pairs + k) for k in range(n_pairs)]
    return new_coords, new_dist, new_path, new_node2osmid, new_pairs
