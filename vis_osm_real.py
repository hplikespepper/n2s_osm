#!/usr/bin/env python
"""Render PDTSP solutions on OSM or Positron tiles using fixed road graphs.

Routes deliberately connect OSM path nodes, as in vis_osm.py; edge geometry
and training objectives are unchanged. See README_vis_osm_real.md.
"""
import argparse
import json
import math
import os
import pickle
import sys
from numbers import Integral
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import contextily as cx
import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer

from data.osm_utils import build_drive_graph


def decode_solution(adj, num_nodes):
    if len(adj) != num_nodes or num_nodes == 0:
        raise ValueError('best_path length must equal the number of task nodes')
    if any(not isinstance(n, Integral) or not 0 <= n < num_nodes for n in adj):
        raise ValueError('best_path contains an invalid task node index')
    route, visited = [0], {0}
    for _ in range(num_nodes):
        nxt = int(adj[route[-1]])
        if nxt == 0:
            if len(visited) != num_nodes:
                raise ValueError('best_path returns to depot before visiting every task node')
            return route + [0]
        if nxt in visited:
            raise ValueError(f'best_path has a non-depot cycle at task node {nxt}')
        visited.add(nxt)
        route.append(nxt)
    raise ValueError('best_path does not close at depot')


def validate_pairs(sample):
    count = len(sample['node2osmid'])
    seen = []
    for pair in sample['pairs']:
        if len(pair) != 2 or any(not isinstance(n, Integral) or not 0 < n < count for n in pair):
            raise ValueError(f'Invalid pickup/delivery pair: {pair}')
        seen.extend(pair)
    if sorted(seen) != list(range(1, count)):
        raise ValueError('pairs must cover each non-depot task node exactly once')


def reconstruct_route(G, sample, route):
    mapping = sample['node2osmid']
    for idx, node in enumerate(mapping):
        if node not in G:
            raise ValueError(f'Task node {idx}: OSM node {node} is missing from graph')
    lookup = sample.get('path_lookup') or {}
    osm_route = []
    for u, v in zip(route[:-1], route[1:]):
        if (u, v) in lookup:
            segment = list(lookup[u, v])
        else:
            try:
                segment = nx.shortest_path(G, mapping[u], mapping[v], weight='length')
            except nx.NetworkXNoPath as exc:
                raise ValueError(f'Task segment {u}->{v} is unreachable') from exc
        if not segment or segment[0] != mapping[u] or segment[-1] != mapping[v]:
            raise ValueError(f'Task segment {u}->{v}: path endpoints do not match node2osmid')
        for node in segment:
            if node not in G:
                raise ValueError(f'Task segment {u}->{v}: OSM node {node} is missing')
        for a, b in zip(segment[:-1], segment[1:]):
            if not G.has_edge(a, b):
                raise ValueError(f'Task segment {u}->{v}: directed OSM edge {a}->{b} is missing')
        osm_route.extend(segment[1:] if osm_route else segment)
    return osm_route


def project_nodes(G, nodes):
    tf = Transformer.from_crs(G.graph.get('crs', 'EPSG:4326'), 'EPSG:3857', always_xy=True)
    xs = [G.nodes[n]['x'] for n in nodes]
    ys = [G.nodes[n]['y'] for n in nodes]
    x, y = tf.transform(xs, ys)
    points = np.column_stack((x, y))
    if not np.isfinite(points).all():
        raise ValueError('Road graph contains invalid coordinates')
    return points


def display_bounds(points):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    span = np.maximum(high - low, 200.0)
    half = span * 0.6  # 10% margin on each side
    return center - half, center + half


def load_graph(graphml, val_dataset, place):
    path = Path(graphml) if graphml else Path(val_dataset).with_suffix('.graphml')
    if graphml or path.exists():
        print(f'Loading fixed road graph: {path}')
        return ox.load_graphml(filepath=path), path, False
    print('Legacy dataset: fixing a road graph for the first time. Matching nodes/edges\n'
          'cannot prove this is the original dataset graph. Existing OSMnx download\n'
          'cache will be reused where available.')
    ox.settings.use_cache = True
    return build_drive_graph(place), path, True


def add_arrows(ax, points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = lengths.sum()
    if total == 0:
        return
    cumulative = np.cumsum(lengths)
    count = min(12, max(1, int(total / 1000)))
    for target in np.linspace(total / (count + 1), total * count / (count + 1), count):
        idx = min(int(np.searchsorted(cumulative, target)), len(lengths) - 1)
        if lengths[idx] == 0:
            continue
        fraction = (target - (cumulative[idx] - lengths[idx])) / lengths[idx]
        direction = (points[idx + 1] - points[idx]) / lengths[idx]
        tip = points[idx] + fraction * (points[idx + 1] - points[idx])
        tail = tip - direction * min(lengths[idx] * 0.3, total * 0.015)
        ax.annotate('', xy=tip, xytext=tail, zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color='red', lw=1.8, mutation_scale=15))


def plot_solution(G, sample, osm_route, cost, index, args, output):
    route_xy = project_nodes(G, osm_route)
    task_xy = project_nodes(G, sample['node2osmid'])
    points = (project_nodes(G, list(G.nodes)) if args.extent == 'place'
              else np.vstack((route_xy, task_xy)))
    low, high = display_bounds(points)
    provider = (cx.providers.OpenStreetMap.Mapnik if args.basemap == 'osm'
                else cx.providers.CartoDB.Positron)
    if args.basemap == 'positron':
        api_key = os.environ.get('CARTO_BASEMAP_API_KEY', '').strip()
        if not api_key:
            raise ValueError('Positron requires CARTO_BASEMAP_API_KEY. Get a key at '
                             'https://www.carto.com/basemaps/apikey/ and set it in your environment.')
        # Explicit current endpoint also supports older xyzservices catalogs.
        provider = provider.copy()
        provider['url'] = 'https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png?key={apikey}'
        provider['apikey'] = api_key
    min_zoom, max_zoom = provider.get('min_zoom', 0), provider['max_zoom']
    if args.zoom != 'auto' and not min_zoom <= args.zoom <= max_zoom:
        raise ValueError(f'Zoom must be between {min_zoom} and {max_zoom} for {args.basemap}')
    fig, ax = plt.subplots(figsize=(12, 12))
    try:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_aspect('equal')
        try:
            cx.add_basemap(ax, source=provider, zoom=args.zoom, crs='EPSG:3857',
                           attribution_size=8, zorder=0)
        except Exception as exc:
            message = str(exc)
            if args.basemap == 'positron':
                message = message.replace(api_key, '[redacted]')
            raise RuntimeError(f'Failed to load {args.basemap} basemap: {message}') from None
        ax.plot(route_xy[:, 0], route_xy[:, 1], color='red', lw=3, alpha=0.9,
                label='Solution Route', zorder=3,
                path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
        add_arrows(ax, route_xy)
        ax.scatter(*task_xy[0], s=300, c='green', marker='*', edgecolors='black',
                   linewidths=1.5, label='Depot', zorder=6)
        for pair_id, (pickup, delivery) in enumerate(sample['pairs'], 1):
            for node, marker, color, label in ((pickup, 'o', 'blue', 'Pick up'),
                                                (delivery, 's', 'darkorange', 'Drop off')):
                ax.scatter(*task_xy[node], s=150, c='white', marker=marker,
                           edgecolors=color, linewidths=2, zorder=5,
                           label=label if pair_id == 1 else None)
                ax.annotate(str(pair_id), task_xy[node], xytext=(0, 0),
                            textcoords='offset points', ha='center', va='center',
                            fontsize=8, weight='bold', zorder=7,
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.set_title(f'Instance {index} - Objective: {cost:.2f}\n'
                     f'Task nodes: {len(task_xy)} (including depot) | Basemap: {args.basemap}',
                     fontsize=14, weight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.set_axis_off()
        fig.savefig(output, dpi=args.dpi, bbox_inches='tight')
    finally:
        plt.close(fig)


def parse_zoom(value):
    if value == 'auto':
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('zoom must be auto or an integer') from exc


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/pdtsp_results_pt_199.json')
    parser.add_argument('--val_dataset', default='./datasets/osm_val_20.pkl')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--osm_place', default='Boca Raton, Florida, USA')
    parser.add_argument('--output_dir', default='visualizations')
    parser.add_argument('--basemap', choices=['osm', 'positron'], default='osm')
    parser.add_argument('--graphml', help='Fixed road graph; otherwise discover dataset sibling .graphml')
    parser.add_argument('--extent', choices=['route', 'place'], default='route')
    parser.add_argument('--zoom', type=parse_zoom, default='auto')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--tile_cache_dir', default='cache/vis_osm_real')
    return parser


def main(args):
    if args.index < 0:
        raise ValueError('Instance index must be non-negative')
    if args.dpi <= 0:
        raise ValueError('dpi must be positive')
    with open(args.results) as f:
        results = json.load(f)
    with open(args.val_dataset, 'rb') as f:
        dataset = pickle.load(f)
    instances = results['instances']
    if args.index >= len(instances) or args.index >= len(dataset):
        raise ValueError(f'Instance index {args.index} out of range: '
                         f'{len(instances)} results, {len(dataset)} dataset samples')
    instance, sample = instances[args.index], dataset[args.index]
    cost = float(instance['best_cost'])
    if not math.isfinite(cost):
        raise ValueError('best_cost must be finite')
    validate_pairs(sample)
    route = decode_solution(instance['best_path'], len(sample['node2osmid']))
    G, graph_file, newly_built = load_graph(args.graphml, args.val_dataset, args.osm_place)
    osm_route = reconstruct_route(G, sample, route)
    # Do not persist a mismatching or malformed road graph.
    project_nodes(G, osm_route + list(sample['node2osmid']))
    if newly_built:
        ox.save_graphml(G, filepath=graph_file)
        print(f'Fixed road graph saved: {graph_file}')
    cache_dir = Path(args.tile_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cx.set_cache_dir(str(cache_dir.resolve()))
    # Identify the application rather than using contextily's generic agent,
    # which OSM can reject with an HTTP-200 policy image.
    cx.tile.USER_AGENT = 'vis_osm_real/1.0 (PDTSP research visualization; Python requests)'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'instance_{args.index}_cost_{cost:.0f}_{args.basemap}.png'
    plot_solution(G, sample, osm_route, cost, args.index, args, output)
    print(f'Saved visualization: {output}')
    return output


if __name__ == '__main__':
    try:
        main(make_parser().parse_args())
    except (ValueError, KeyError, OSError, RuntimeError, nx.NetworkXException) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
