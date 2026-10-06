"""Offline checks for solution integrity, graph persistence and map rendering."""
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vis_osm_real as vis


def fixture():
    graph = nx.MultiDiGraph(crs='EPSG:4326')
    for n, (x, y) in enumerate([(-80.10, 26.35), (-80.09, 26.35), (-80.09, 26.36)], 10):
        graph.add_node(n, x=x, y=y)
    for u, v in [(10, 11), (11, 12), (12, 10)]:
        graph.add_edge(u, v, length=1000.)
    sample = {'node2osmid': [10, 11, 12], 'pairs': [(1, 2)],
              'path_lookup': {(0, 1): [10, 11], (1, 2): [11, 12], (2, 0): [12, 10]}}
    return graph, sample


class RealMapTests(unittest.TestCase):
    def setUp(self):
        self.G, self.sample = fixture()

    def test_decode_and_integrity(self):
        self.assertEqual(vis.decode_solution([1, 2, 0], 3), [0, 1, 2, 0])
        for adj in ([1, 0, 2], [1, 2, 1], [1, 2, 3], [1, 0], [1, 2, 0.0]):
            with self.subTest(adj=adj), self.assertRaises(ValueError):
                vis.decode_solution(adj, 3)
        vis.validate_pairs(self.sample)
        self.sample['pairs'] = [(1, 1)]
        with self.assertRaises(ValueError):
            vis.validate_pairs(self.sample)

    def test_paths_and_shortest_path_fallback(self):
        route = [0, 1, 2, 0]
        self.assertEqual(vis.reconstruct_route(self.G, self.sample, route), [10, 11, 12, 10])
        self.sample['path_lookup'] = {}
        self.assertEqual(vis.reconstruct_route(self.G, self.sample, route), [10, 11, 12, 10])
        self.G.remove_edge(12, 10)
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            vis.reconstruct_route(self.G, self.sample, route)

    def test_missing_node_edge_and_bad_endpoints(self):
        route = [0, 1, 2, 0]
        self.G.remove_edge(11, 12)
        with self.assertRaisesRegex(ValueError, 'directed OSM edge'):
            vis.reconstruct_route(self.G, self.sample, route)
        self.G.remove_node(12)
        with self.assertRaisesRegex(ValueError, 'Task node 2'):
            vis.reconstruct_route(self.G, self.sample, route)
        self.G, self.sample = fixture()
        self.sample['path_lookup'][0, 1] = [11, 10]
        with self.assertRaisesRegex(ValueError, 'endpoints'):
            vis.reconstruct_route(self.G, self.sample, route)

    def test_projection_and_bounds(self):
        points = vis.project_nodes(self.G, [10, 11, 12])
        self.assertTrue(np.isfinite(points).all())
        self.assertLess(points[0, 0], -8e6)
        self.assertLess(points[0, 0], points[1, 0])
        low, high = vis.display_bounds(points)
        self.assertTrue((low < points.min(axis=0)).all())
        self.assertTrue((high > points.max(axis=0)).all())
        low, high = vis.display_bounds(np.array([[1., 2.]]))
        np.testing.assert_allclose(high - low, [240, 240])

    def test_graph_roundtrip_and_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            pkl = Path(folder) / 'val.pkl'
            path = pkl.with_suffix('.graphml')
            with patch.object(vis, 'build_drive_graph', return_value=self.G) as build:
                graph, output, new = vis.load_graph(None, pkl, 'test')
                self.assertTrue(new)
                self.assertEqual(output, path)
                ox.save_graphml(graph, filepath=path)
                graph2, _, new = vis.load_graph(None, pkl, 'test')
                self.assertFalse(new)
                self.assertEqual(build.call_count, 1)
            self.assertEqual(set(graph2.edges), set(self.G.edges))
            for n in graph2:
                self.assertEqual(graph2.nodes[n]['x'], self.G.nodes[n]['x'])
                self.assertEqual(graph2.nodes[n]['y'], self.G.nodes[n]['y'])
            self.assertEqual(vis.reconstruct_route(graph2, self.sample, [0, 1, 2, 0]), [10, 11, 12, 10])
            with self.assertRaises(OSError):
                vis.load_graph(Path(folder) / 'missing.graphml', pkl, 'test')

    def test_main_render_sources_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pkl, results = root / 'val.pkl', root / 'results.json'
            with pkl.open('wb') as f:
                pickle.dump([self.sample], f)
            results.write_text(json.dumps({'instances': [{'best_path': [1, 2, 0], 'best_cost': 3000}]}))
            ox.save_graphml(self.G, filepath=pkl.with_suffix('.graphml'))
            args = vis.make_parser().parse_args(['--results', str(results), '--val_dataset', str(pkl),
                '--output_dir', str(root / 'images'), '--tile_cache_dir', str(root / 'tiles'), '--dpi', '50'])
            sources = []
            def fake_map(ax, **kwargs):
                sources.append(kwargs['source']['name'])
                self.assertEqual(kwargs['crs'], 'EPSG:3857')
                self.assertGreater(ax.get_xlim()[1], ax.get_xlim()[0])
            with patch.dict('os.environ', {'CARTO_BASEMAP_API_KEY': 'test-key'}), patch.object(vis.cx, 'add_basemap', side_effect=fake_map):
                first = vis.main(args)
                args.basemap = 'positron'
                args.extent = 'place'
                second = vis.main(args)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())
            self.assertEqual(sources, ['OpenStreetMap.Mapnik', 'CartoDB.Positron'])
            args.index = -1
            with self.assertRaises(ValueError):
                vis.main(args)
            args.index = 1
            with self.assertRaises(ValueError):
                vis.main(args)
            args.index = 0
            args.output_dir = str(root / 'failed')
            with patch.dict('os.environ', {'CARTO_BASEMAP_API_KEY': 'test-key'}), patch.object(vis.cx, 'add_basemap', side_effect=OSError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'Failed to load'):
                    vis.main(args)
            self.assertEqual(list((root / 'failed').glob('*.png')), [])

    def test_positron_requires_key(self):
        args = vis.make_parser().parse_args(['--basemap', 'positron'])
        with patch.dict('os.environ', {}, clear=True), patch.object(vis.cx, 'add_basemap') as fetch:
            with self.assertRaisesRegex(ValueError, 'CARTO_BASEMAP_API_KEY'):
                vis.plot_solution(self.G, self.sample, [10, 11, 12, 10], 3000, 0, args, 'unused.png')
            fetch.assert_not_called()

    def test_generator_saves_actual_graph(self):
        import torch
        import create_osm_val_dataset as generator
        sample = dict(self.sample, coordinates=torch.zeros((3, 2)), dist=torch.zeros((3, 3)),
                      capacity=3, multi_start=4, disable_geo_aug=True)
        class Dataset:
            G = self.G
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return sample
        with tempfile.TemporaryDirectory() as folder, patch.object(generator, 'OSMOnlinePDPSDataset', return_value=Dataset()):
            output = Path(folder) / 'new.pkl'
            generator.create_val_dataset(num_samples=1, output_file=str(output))
            with output.open('rb') as f:
                saved = pickle.load(f)
            self.assertEqual(set(saved[0]), set(sample))
            graph = ox.load_graphml(filepath=output.with_suffix('.graphml'))
            self.assertEqual(set(graph.edges), set(self.G.edges))
            self.assertEqual(vis.reconstruct_route(graph, saved[0], [0, 1, 2, 0]), [10, 11, 12, 10])


if __name__ == '__main__':
    unittest.main()
