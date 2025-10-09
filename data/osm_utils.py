# data/osm_utils.py
import osmnx as ox
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict
from pyproj import Transformer, CRS


def build_drive_graph(place: str):
    G = ox.graph_from_place(place, network_type="drive")
    # G = ox.distance.add_edge_lengths(G)
    # 兼容不同版本的 OSMnx
    if hasattr(ox, "add_edge_lengths"):
        # 旧版 osmnx (<=1.x)
        G = ox.add_edge_lengths(G)
    elif hasattr(ox.distance, "add_edge_lengths"):
        # 新版 osmnx (>=2.0)
        G = ox.distance.add_edge_lengths(G)
    else:
        raise RuntimeError("Unsupported osmnx version: cannot find add_edge_lengths")
    
    # 确保图是强连通的，只保留最大强连通分量
    # 这样可以避免采样到不可达的节点
    if not nx.is_strongly_connected(G):
        # 获取所有强连通分量，按大小排序
        sccs = list(nx.strongly_connected_components(G))
        largest_scc = max(sccs, key=len)
        # 只保留最大强连通分量的子图
        G = G.subgraph(largest_scc).copy()
        print(f"Warning: Graph was not strongly connected. Using largest SCC with {len(largest_scc)} nodes out of {sum(len(scc) for scc in sccs)} total nodes.")

    return G

def sample_task_nodes(G, n_pairs: int, rng: np.random.Generator):
    nodes = list(G.nodes)
    assert len(nodes) > 2 * n_pairs + 1, "OSM graph too small"
    depot = int(rng.integers(0, len(nodes)))
    depot_osmid = nodes[depot]
    idxs = rng.choice(len(nodes), size=2 * n_pairs, replace=False)
    picks = [nodes[i] for i in idxs[:n_pairs]]
    drops = [nodes[i] for i in idxs[n_pairs:]]
    node2osmid = [depot_osmid] + picks + drops
    pairs = [(i + 1, i + 1 + n_pairs) for i in range(n_pairs)]
    return node2osmid, pairs

def project_and_scale_xy(G, node_ids: List[int]) -> np.ndarray:
    xs = [G.nodes[n]['x'] for n in node_ids]
    ys = [G.nodes[n]['y'] for n in node_ids]
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True)
    X, Y = tf.transform(xs, ys)
    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
    def mm(z):
        zmin, zmax = z.min(), z.max()
        return np.zeros_like(z) if zmax - zmin < 1e-9 else (z - zmin) / (zmax - zmin)
    return np.stack([mm(X), mm(Y)], axis=1).astype(np.float32)

def compute_apsp_for_subset(G, node2osmid: List[int]) -> Tuple[np.ndarray, Dict[Tuple[int,int], List[int]]]:
    N1 = len(node2osmid)
    dist = np.zeros((N1, N1), dtype=np.float32)
    path_lookup: Dict[Tuple[int, int], List[int]] = {}
    for i, u in enumerate(node2osmid):
        lengths, paths = nx.single_source_dijkstra(G, u, weight="length")
        for j, v in enumerate(node2osmid):
            if i == j:
                continue
            if v not in lengths:
                dist[i, j] = 1e9
                path_lookup[(i, j)] = [u, v]
            else:
                dist[i, j] = float(lengths[v])
                path_lookup[(i, j)] = paths[v]
    return dist, path_lookup
