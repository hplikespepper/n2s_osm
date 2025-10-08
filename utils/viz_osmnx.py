# utils/viz_osmnx.py
import osmnx as ox
import matplotlib.pyplot as plt
import torch

def plot_solution_on_osm(G, sample, seq, out_path):
    fig, ax = plt.subplots(figsize=(10,10))
    ox.plot_graph(G, node_size=2, edge_color='lightgray', show=False, close=False, ax=ax)

    # 把张量/列表统一一下
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().tolist()
    seq = [int(x) for x in seq]

    osmid_seq = []
    for u, v in zip(seq[:-1], seq[1:]):
        if ('path_lookup' in sample) and ((u, v) in sample['path_lookup']):
            seg = sample['path_lookup'][(u, v)]
        else:
            osmu = sample['node2osmid'][u]
            osmv = sample['node2osmid'][v]
            seg = ox.distance.shortest_path(G, osmu, osmv, weight='length')
        if osmid_seq and seg and seg[0] == osmid_seq[-1]:
            osmid_seq += seg[1:]
        else:
            osmid_seq += seg

    xs = [G.nodes[n]['x'] for n in osmid_seq]
    ys = [G.nodes[n]['y'] for n in osmid_seq]
    ax.plot(xs, ys, linewidth=2, alpha=0.9)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
