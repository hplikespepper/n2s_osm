"""Run with: python -m unittest discover -s tests -p test_monte_carlo.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from MonteCarlo_mvpdp import solve_instance
from problems.problem_mvpdtsp import MVPDTSP


class MonteCarloTests(unittest.TestCase):
    def test_candidate_selection_uses_requested_objective(self):
        # The three objectives must prefer different candidates.
        candidates = [(10., 9.), (20., 4.), (15., 5.)]
        routes = [[0, 5, 6, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        for objective, expected in [('distance', 10.), ('makespan', 4.),
                                    ('distance+makespan', 35.)]:
            with self.subTest(objective=objective), patch(
                'MonteCarlo_mvpdp.generate_random_solution', return_value=(list(range(7)), routes)
            ), patch.object(MVPDTSP, 'compute_cost_components', side_effect=[
                (torch.tensor([d]), torch.tensor([m])) for d, m in candidates
            ]):
                _, _, cost = solve_instance(np.zeros((7, 2)), 5, 3, objective)
                self.assertEqual(cost, expected)

    def test_serial_parallel_reproducibility_and_feasibility(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for workers in (1, 2):
                output = Path(tmp) / f'{workers}.json'
                subprocess.run([
                    sys.executable, str(root / 'MonteCarlo_mvpdp.py'),
                    '--val_dataset', str(root / 'datasets/pdp_100.pkl'),
                    '--graph_size', '100', '--num_vehicles', '5',
                    '--val_size', '4', '--num_mc_samples', '100',
                    '--num_workers', str(workers), '--output', str(output),
                ], check=True, capture_output=True, text=True, timeout=120)
                outputs.append(json.loads(output.read_text()))
            serial, parallel = outputs
            self.assertEqual(len({x['worker_pid'] for x in parallel['instances']}), 2)
            problem = MVPDTSP(100, num_vehicles=5, with_assert=True, use_makespan=True)
            for a, b in zip(serial['instances'], parallel['instances']):
                self.assertEqual(a['best_rec'], b['best_rec'])
                self.assertEqual(a['best_cost'], b['best_cost'])
                self.assertEqual(b['best_cost'], b['best_distance_cost'] + 4 * b['best_makespan_cost'])
                problem.check_feasibility(torch.tensor([b['best_rec']]))
                cost = problem.get_costs({'coordinates': torch.tensor([b['coordinates']])},
                                         torch.tensor([b['best_rec']])).item()
                self.assertAlmostEqual(cost, b['best_cost'], delta=1e-5)


if __name__ == '__main__':
    unittest.main()
