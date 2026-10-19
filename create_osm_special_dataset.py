#!/usr/bin/env python3
"""Convert manually selected WGS84 CSV points into one fixed-road OSM instance."""
import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import CRS, Geod

from data.osm_utils import compute_apsp_for_subset, project_and_scale_xy
from data.osm_mvpdp_dataset import expand_single_depot

ROOT = Path(__file__).resolve().parent


def read_points(path, graph_size=20):
    """Return depot, pickups, deliveries; CSV pair IDs remain available as labels."""
    if graph_size <= 0 or graph_size % 2:
        raise ValueError('graph_size must be positive and even')
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'node_type', 'pair_id', 'longitude', 'latitude'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV requires columns: {sorted(required)}')
        points = {}
        for line, row in enumerate(reader, 2):
            kind = row['node_type'].strip().lower()
            if kind not in ('depot', 'pickup', 'delivery'):
                raise ValueError(f'Line {line}: invalid node_type {kind!r}')
            pair_id = row['pair_id'].strip()
            if kind == 'depot':
                if pair_id:
                    raise ValueError(f'Line {line}: depot pair_id must be empty')
                pair_id = None
            else:
                pair_id = int(pair_id)
                if pair_id < 1:
                    raise ValueError(f'Line {line}: pair_id must be positive')
            lon, lat = float(row['longitude']), float(row['latitude'])
            if not (math.isfinite(lon) and math.isfinite(lat)
                    and -180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f'Line {line}: invalid WGS84 coordinates')
            key = kind, pair_id
            if key in points:
                raise ValueError(f'Line {line}: duplicate {key}')
            points[key] = dict(node_type=kind, pair_id=pair_id, longitude=lon, latitude=lat)
    pickups = {key[1] for key in points if key[0] == 'pickup'}
    deliveries = {key[1] for key in points if key[0] == 'delivery'}
    if ('depot', None) not in points or pickups != deliveries or len(pickups) != graph_size // 2:
        raise ValueError(f'Expected one depot and {graph_size // 2} complete pickup/delivery pairs')
    ids = sorted(pickups)
    keys = [('depot', None)] + [('pickup', i) for i in ids] + [('delivery', i) for i in ids]
    return [points[key] for key in keys]


def snap_points(graph, points):
    """Find nearest graph nodes by WGS84 geodesic distance, without sklearn."""
    if not CRS(graph.graph.get('crs', 'EPSG:4326')).equals(CRS.from_epsg(4326)):
        raise ValueError('The fixed graph must use WGS84 longitude/latitude coordinates')
    nodes = list(graph.nodes)
    lons = np.array([graph.nodes[node]['x'] for node in nodes], dtype=float)
    lats = np.array([graph.nodes[node]['y'] for node in nodes], dtype=float)
    geod = Geod(ellps='WGS84')
    mapping, report = [], []
    for index, point in enumerate(points):
        _, _, distances = geod.inv(np.full(len(nodes), point['longitude']),
                                   np.full(len(nodes), point['latitude']), lons, lats)
        nearest = int(np.argmin(distances))
        node = int(nodes[nearest])
        mapping.append(node)
        report.append(dict(point, task_index=index, osmid=node,
                           snapped_longitude=float(lons[nearest]),
                           snapped_latitude=float(lats[nearest]),
                           snap_distance_m=float(distances[nearest])))
    return mapping, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'datasets/special_instance.csv')
    parser.add_argument('--graphml', type=Path, default=ROOT / 'datasets/osm_val_20.graphml')
    parser.add_argument('--graph_size', type=int, default=20)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'result_special_case')
    args = parser.parse_args()
    points = read_points(args.csv, args.graph_size)
    graph = ox.load_graphml(args.graphml)
    mapping, report = snap_points(graph, points)
    # Reject unreachable requests before the legacy helper can insert a sentinel cost.
    component_id = {node: i for i, group in enumerate(nx.strongly_connected_components(graph)) for node in group}
    if len({component_id[node] for node in mapping}) != 1:
        raise ValueError('Selected road nodes are not mutually reachable in the fixed directed graph')
    distances, lookup = compute_apsp_for_subset(graph, mapping)
    half = args.graph_size // 2
    sample = {
        'coordinates': project_and_scale_xy(graph, mapping), 'dist': distances,
        'node2osmid': mapping, 'pairs': [(i, i + half) for i in range(1, half + 1)],
        'path_lookup': lookup, 'capacity': 3,
        'multi_start': 1, 'disable_geo_aug': True,
        'source_csv': str(args.csv.resolve()), 'requested_points': report,
    }
    expand_single_depot(sample, 2, args.graph_size)
    if not np.isfinite(distances).all() or (distances >= 1e9).any():
        raise ValueError('Invalid or unreachable road distances')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / 'special_instance.pkl'
    with dataset_path.open('wb') as handle:
        pickle.dump([sample], handle, protocol=4)
    # Retain the exact road graph with the same basename for automatic visualization.
    graph_path = output / 'special_instance.graphml'
    if graph_path.resolve() != args.graphml.resolve():
        graph_path.write_bytes(args.graphml.read_bytes())
    with (output / 'node_mapping.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    groups = {}
    for row in report:
        groups.setdefault(str(row['osmid']), []).append(row['task_index'])
    metadata = {
        'source_csv': str(args.csv.resolve()),
        'csv_sha256': hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        'source_graphml': str(args.graphml.resolve()),
        'graphml_sha256': hashlib.sha256(args.graphml.read_bytes()).hexdigest(),
        'dataset': str(dataset_path.resolve()), 'graph_size': args.graph_size,
        'num_instances': 1, 'num_vehicles': 2, 'original_total_nodes': len(mapping),
        'snap_method': 'nearest graph node by WGS84 geodesic distance',
        'max_snap_distance_m': max(row['snap_distance_m'] for row in report),
        'mean_snap_distance_m': float(np.mean([row['snap_distance_m'] for row in report])),
        'shared_road_nodes': {node: ids for node, ids in groups.items() if len(ids) > 1},
        'nodes': report,
    }
    (output / 'dataset_manifest.json').write_text(json.dumps(metadata, indent=2))
    print(f'Dataset: {dataset_path}\nRoad graph: {graph_path}')
    print(f"Snap distances: mean={metadata['mean_snap_distance_m']:.2f}m, max={metadata['max_snap_distance_m']:.2f}m")
    print(f"Shared road nodes: {metadata['shared_road_nodes']}")


if __name__ == '__main__':
    main()
