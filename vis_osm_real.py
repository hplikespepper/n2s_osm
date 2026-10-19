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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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


def decode_vehicle_routes(rec, num_nodes, num_vehicles):
    if num_vehicles < 1 or num_vehicles > num_nodes:
        raise ValueError('Invalid vehicle count')
    if len(rec) != num_nodes or any(not isinstance(n, Integral) or not 0 <= n < num_nodes for n in rec):
        raise ValueError('Invalid multi-vehicle successor table')
    if sorted(rec) != list(range(num_nodes)):
        raise ValueError('Multi-vehicle successor table must be a permutation')
    routes, covered = [], set()
    for v in range(num_vehicles):
        route, visited = [v], {v}
        for _ in range(num_nodes):
            nxt = int(rec[route[-1]])
            route.append(nxt)
            if nxt == v:
                break
            if nxt < num_vehicles or nxt in visited or nxt in covered:
                raise ValueError(f'Invalid cycle for vehicle {v}')
            visited.add(nxt)
        else:
            raise ValueError(f'Vehicle {v} does not return to depot')
        covered.update(visited)
        routes.append(route)
    if covered != set(range(num_nodes)):
        raise ValueError('Vehicle routes do not cover all task nodes')
    return routes


def validate_pairs(sample, num_depots=1):
    count = len(sample['node2osmid'])
    seen = []
    for pair in sample['pairs']:
        if len(pair) != 2 or any(not isinstance(n, Integral) or not num_depots <= n < count for n in pair):
            raise ValueError(f'Invalid pickup/delivery pair: {pair}')
        seen.extend(pair)
    if sorted(seen) != list(range(num_depots, count)):
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


def add_arrows(ax, points, color="red"):
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
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8, mutation_scale=15))


def route_labels(sample, task_route):
    """Use map request numbering and name each logical depot Depot."""
    labels = {}
    for pair_id, (pickup, delivery) in enumerate(sample['pairs'], 1):
        labels[pickup], labels[delivery] = f'P{pair_id}', f'D{pair_id}'
    return [labels.get(node, 'Depot') for node in task_route]


def wrap_route(labels, width=36):
    lines, current = [], labels[0]
    for label in labels[1:]:
        extension = f' → {label}'
        if len(current + extension) > width:
            lines.append(current)
            current = f'→ {label}'
        else:
            current += extension
    return '\n'.join(lines + [current])


def plot_solution(G, sample, osm_route, cost, index, args, output, multi=False,
                  task_routes=None, distance=None, makespan=None):
    if task_routes is None:
        raise ValueError('Plotting requires task routes for node ownership and route labels')
    if distance is None or makespan is None:
        route_distances = [sum(float(sample['dist'][a, b]) for a, b in zip(route, route[1:]))
                           for route in task_routes]
        distance, makespan = sum(route_distances), max(route_distances)
    vehicle_by_task = ({node: v for v, route in enumerate(task_routes) for node in route[1:-1]}
                       if multi else {})
    routes_xy = [project_nodes(G, route) for route in osm_route] if multi else [project_nodes(G, osm_route)]
    route_xy = np.vstack(routes_xy)
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
    route_texts = [wrap_route(route_labels(sample, route), width=80) for route in task_routes]
    footer_height = 0.4 + sum(0.32 * (text.count('\n') + 1) + 0.18 for text in route_texts)
    map_width = 11.6
    map_height = map_width * (high[1] - low[1]) / (high[0] - low[0])
    figure_height = map_height + footer_height + 0.9
    fig, ax = plt.subplots(figsize=(12, figure_height))
    ax.set_position([0.2 / 12, (footer_height + 0.15) / figure_height,
                     map_width / 12, map_height / figure_height])
    footer = fig.add_axes([0.2 / 12, 0.15 / figure_height, map_width / 12,
                           footer_height / figure_height])
    footer.set_axis_off()
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
        # Strong colors shared by routes, task outlines, legend and route text.
        colors = ['#542788', '#FF7F0E'] + list(plt.get_cmap('tab10').colors[2:])
        for v, xy in enumerate(routes_xy):
            color = colors[v % len(colors)]
            label = f'Vehicle {v + 1}' if multi else 'Solution Route'
            if multi and np.all(xy == xy[0]):
                label += ' (empty)'
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=3, alpha=1.0,
                    label=label, zorder=3,
                    path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
            add_arrows(ax, xy, color=color)
        ax.scatter(*task_xy[0], s=300, c='green', marker='*', edgecolors='black',
                   linewidths=1.5, label='Depot', zorder=6)
        for pair_id, (pickup, delivery) in enumerate(sample['pairs'], 1):
            for node, marker in ((pickup, 'o'), (delivery, 's')):
                color = colors[vehicle_by_task[node] % len(colors)] if multi else colors[0]
                ax.scatter(*task_xy[node], s=150, c='white', marker=marker,
                           edgecolors=color, linewidths=2, zorder=5)
                ax.annotate(str(pair_id), task_xy[node], xytext=(0, 0),
                            textcoords='offset points', ha='center', va='center',
                            fontsize=8, weight='bold', zorder=7,
                            path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        ax.set_title(f'Objective: {cost:.2f} | Distance: {distance:.2f} | Makespan: {makespan:.2f}\n'
                     f'Request pairs: {len(sample["pairs"])} | Vehicles: {len(routes_xy)}',
                     fontsize=14, weight='bold')
        handles, labels = ax.get_legend_handles_labels()
        for marker, label in (('o', 'Pick up'), ('s', 'Drop off')):
            handles.append(Line2D([], [], linestyle='None', marker=marker,
                                  markerfacecolor='white', markeredgecolor='black',
                                  markeredgewidth=2, markersize=10, label=label))
            labels.append(label)
        legend = ax.legend(handles, labels, loc='best', frameon=True, fancybox=False,
                           facecolor='white', edgecolor='#555555', framealpha=0.95,
                           borderpad=0.55, fontsize=14, handlelength=2.5)
        legend.get_frame().set_linewidth(1.2)
        route_artists = []
        y = 1 - 0.24 / footer_height
        for vehicle, route_text in enumerate(route_texts):
            color = colors[vehicle % len(colors)]
            route_artists.append(footer.text(
                0.02, y, f'Vehicle {vehicle + 1}: {route_text}', color=color,
                fontsize=14, family='DejaVu Sans Mono', linespacing=1.5,
                va='top', transform=footer.transAxes))
            y -= (0.32 * (route_text.count('\n') + 1) + 0.18) / footer_height
        # Fit the footer to rendered text, with compact padding and row spacing.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_heights = [text.get_window_extent(renderer).height / fig.dpi for text in route_artists]
        padding, row_gap = 0.12, 0.08
        footer_height = 2 * padding + sum(text_heights) + row_gap * max(0, len(route_artists) - 1)
        figure_height = map_height + footer_height + 0.9
        fig.set_size_inches(12, figure_height)
        ax.set_position([0.2 / 12, (footer_height + 0.15) / figure_height,
                         map_width / 12, map_height / figure_height])
        footer.set_position([0.2 / 12, 0.15 / figure_height,
                             map_width / 12, footer_height / figure_height])
        offset = padding
        for text, height in zip(route_artists, text_heights):
            text.set_position((0.02, 1 - offset / footer_height))
            offset += height + row_gap
        route_frame = Rectangle((0.005, 0.04 / footer_height), 0.99,
                                1 - 0.08 / footer_height, transform=footer.transAxes,
                                facecolor='white', edgecolor='#555555', linewidth=1.2,
                                zorder=0, clip_on=False)
        footer.add_patch(route_frame)
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
    if args.index >= len(instances):
        raise ValueError(f'Instance index {args.index} out of range: '
                         f'{len(instances)} results, {len(dataset)} dataset samples')
    instance = instances[args.index]
    source_index = instance.get('dataset_index', args.index)
    if not isinstance(source_index, Integral) or not 0 <= source_index < len(dataset):
        raise ValueError(f'Dataset index {source_index} out of range')
    sample = dataset[source_index]
    multi = results.get('problem') == 'mvpdtsp_osm'
    num_vehicles = int(results.get('num_vehicles', 1)) if multi else 1
    if multi:
        from data.osm_mvpdp_dataset import expand_single_depot
        sample = expand_single_depot(sample, num_vehicles, results['graph_size'])
    cost = float(instance['best_cost'])
    if not math.isfinite(cost):
        raise ValueError('best_cost must be finite')
    validate_pairs(sample, num_vehicles)
    if multi:
        routes = decode_vehicle_routes(instance['best_rec'], len(sample['node2osmid']), num_vehicles)
        for pickup, delivery in sample['pairs']:
            if not any(pickup in route and delivery in route and route.index(pickup) < route.index(delivery) for route in routes):
                raise ValueError(f'Invalid pickup/delivery order or vehicle for {pickup}->{delivery}')
    else:
        routes = [decode_solution(instance['best_path'], len(sample['node2osmid']))]
    G, graph_file, newly_built = load_graph(args.graphml, args.val_dataset, args.osm_place)
    osm_routes = [reconstruct_route(G, sample, route) for route in routes]
    osm_route = [node for route in osm_routes for node in route]
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
    suffix = f'_mv{num_vehicles}' if multi else ''
    output = output_dir / f'instance_{args.index}_cost_{cost:.0f}{suffix}_{args.basemap}.png'
    plot_solution(G, sample, osm_routes if multi else osm_route, cost, args.index, args, output,
                  multi=multi, task_routes=routes,
                  distance=instance.get('best_distance_cost'),
                  makespan=instance.get('best_makespan_cost'))
    print(f'Saved visualization: {output}')
    return output


if __name__ == '__main__':
    try:
        main(make_parser().parse_args())
    except (ValueError, KeyError, OSError, RuntimeError, nx.NetworkXException) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
