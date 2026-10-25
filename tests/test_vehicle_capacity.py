import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options import get_options
from agent.ppo import PPO, _checkpoint_state, _copy_optimal_checkpoint, _restore_checkpoint_state
from problems.problem_mvpdtsp import MVPDTSP
from problems.problem_mvpdtsp_fixed import MVPDTSPFixedAssignment


def route_record(routes, num_pairs):
    total_nodes = len(routes) + 2 * num_pairs
    rec = list(range(total_nodes))
    for vehicle, route in enumerate(routes):
        cycle = [vehicle] + route + [vehicle]
        for current, nxt in zip(cycle, cycle[1:]):
            rec[current] = nxt
    return torch.tensor([rec], dtype=torch.long)


class VehicleCapacityTests(unittest.TestCase):
    def setUp(self):
        # Three tasks: 2->5, 3->6, 4->7. Vehicle 0 reaches load 2.
        self.rec = route_record([[2, 3, 5, 6], [4, 7]], 3)
        self.order_map = torch.zeros((1, 8, 8), dtype=torch.bool)
        self.selected = torch.tensor([4])

    def mask(self, problem):
        return problem.get_swap_mask(self.selected, self.order_map, rec=self.rec)[0]

    def test_homogeneous_and_heterogeneous_boundaries(self):
        shared = MVPDTSP(6, num_vehicles=2, capacity=2)
        mixed = MVPDTSP(6, num_vehicles=2, capacity=[2, 1])
        self.assertEqual(shared.vehicle_capacities, [2, 2])
        self.assertEqual(mixed.vehicle_capacities, [2, 1])
        self.assertEqual(mixed.get_vehicle_peak_loads(self.rec).tolist(), [[2, 1]])
        shared.check_feasibility(self.rec)
        mixed.check_feasibility(self.rec)
        with self.assertRaisesRegex(AssertionError, 'capacity'):
            MVPDTSP(6, num_vehicles=2, capacity=[1, 2]).check_feasibility(self.rec)

    def test_initial_solutions_are_feasible_for_both_methods(self):
        batch = {'coordinates': torch.rand(3, 25, 2)}
        for method in ('random', 'greedy'):
            problem = MVPDTSP(20, num_vehicles=5,
                              capacity=[7, 7, 7, 3, 3], init_val_met=method)
            rec = problem.get_initial_solutions(batch)
            problem.check_feasibility(rec)
            self.assertTrue((problem.get_vehicle_peak_loads(rec) <= 1).all())

    def test_mask_checks_entire_interval_and_same_anchor(self):
        problem = MVPDTSP(6, num_vehicles=2, capacity=[2, 1])
        mask = self.mask(problem)
        self.assertTrue(mask[0, 5])  # Load 2 occurs between these anchors.
        self.assertTrue(mask[3, 3])  # Pickup and delivery share a full-load anchor.
        self.assertFalse(mask[0, 0])  # Same anchor with spare capacity.
        self.assertFalse(mask[5, 5])
        self.assertFalse(mask[1, 1])  # Empty vehicle after removing its task.
        self.assertTrue(mask[0, 1])  # Anchors cannot cross vehicles.

    def test_cross_vehicle_move_and_invalid_action(self):
        problem = MVPDTSP(6, num_vehicles=2, capacity=[2, 1])
        moved = problem.insert_star(self.rec, self.selected.view(1, 1),
                                    torch.tensor([[0]]), torch.tensor([[0]]))
        problem.check_feasibility(moved)
        self.assertEqual(problem.get_vehicle_peak_loads(moved).tolist(), [[2, 0]])
        with self.assertRaisesRegex(AssertionError, 'capacity'):
            problem.insert_star(self.rec, self.selected.view(1, 1),
                                torch.tensor([[3]]), torch.tensor([[3]]))

    def test_unconstrained_legacy_behavior(self):
        problem = MVPDTSP(6, num_vehicles=2)
        self.assertIsNone(problem.vehicle_capacities)
        self.assertFalse(self.mask(problem)[3, 3])
        overloaded = problem.insert_star(self.rec, self.selected.view(1, 1),
                                         torch.tensor([[3]]), torch.tensor([[3]]))
        self.assertEqual(problem.get_vehicle_peak_loads(overloaded).tolist(), [[3, 0]])
        problem.check_feasibility(overloaded)
        fixed = MVPDTSPFixedAssignment(6, num_vehicles=2, capacity=1)
        self.assertIsNone(fixed.vehicle_capacities)

    def test_direct_capacity_validation(self):
        for capacity in (0, -1, [2], [2, 0], [2, 1, 1], 2.5):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                MVPDTSP(6, num_vehicles=2, capacity=capacity)

    def test_optimal_checkpoint_is_an_extra_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_epoch = root / 'epoch-0.pt'
            first_epoch.write_bytes(b'first checkpoint')
            first_optimal = Path(_copy_optimal_checkpoint(str(first_epoch), directory, 0))
            self.assertEqual(first_optimal.name, 'optimal-0.pt')
            self.assertEqual(first_optimal.read_bytes(), first_epoch.read_bytes())

            later_epoch = root / 'epoch-7.pt'
            later_epoch.write_bytes(b'better checkpoint')
            later_optimal = Path(_copy_optimal_checkpoint(
                str(later_epoch), directory, 7, str(first_optimal)
            ))
            self.assertFalse(first_optimal.exists())
            self.assertEqual(later_optimal.read_bytes(), later_epoch.read_bytes())
            self.assertTrue(first_epoch.exists())
            self.assertTrue(later_epoch.exists())

    def test_resume_restores_ranking_and_discards_uncommitted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            retained = root / 'epoch-4.pt'
            optimal = root / 'optimal-4.pt'
            retained.write_bytes(b'retained')
            optimal.write_bytes(b'best')
            state = _checkpoint_state([(1.25, str(retained))], 1.25, str(optimal))
            (root / 'epoch-5.pt').write_bytes(b'uncommitted')
            (root / 'optimal-5.pt').write_bytes(b'uncommitted')

            ranking, best_score, best_path = _restore_checkpoint_state(directory, state, 4)
            self.assertEqual(ranking, [(1.25, str(retained))])
            self.assertEqual(best_score, 1.25)
            self.assertEqual(best_path, str(optimal))
            self.assertFalse((root / 'epoch-5.pt').exists())
            self.assertFalse((root / 'optimal-5.pt').exists())

    def test_atomic_resume_checkpoint_contains_epoch_and_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = PPO.__new__(PPO)
            agent.actor = torch.nn.Linear(2, 2)
            agent.critic = torch.nn.Linear(2, 1)
            agent.optimizer = torch.optim.Adam(
                list(agent.actor.parameters()) + list(agent.critic.parameters())
            )
            agent.opts = SimpleNamespace(save_dir=directory)
            state = {'top_checkpoints': [], 'optimal_epoch': None,
                     'best_validation_score': None}
            agent.save(7, filename='resume-latest.pt', checkpoint_state=state)
            saved = torch.load(Path(directory) / 'resume-latest.pt', map_location='cpu')
            self.assertEqual(saved['epoch'], 7)
            self.assertEqual(saved['checkpoint_state'], state)
            self.assertFalse((Path(directory) / 'resume-latest.pt.tmp').exists())

    @patch('options.torch.cuda.device_count', return_value=0)
    def test_cli_capacity_validation_and_defaults(self, _):
        args = ['--problem', 'mvpdtsp', '--num_vehicles', '2']
        self.assertIsNone(get_options(args).capacity)
        self.assertEqual(get_options(args + ['--capacity', '3']).capacity, 3)
        self.assertEqual(get_options(args + ['--vehicle_capacities', '7', '3']).vehicle_capacities,
                         [7, 3])
        self.assertEqual(get_options(['--problem', 'mvpdtsp_osm']).capacity, 3)
        for extra in (['--capacity', '0'], ['--vehicle_capacities', '7'],
                      ['--vehicle_capacities', '7', '0'],
                      ['--capacity', '3', '--vehicle_capacities', '3', '3']):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                get_options(args + extra)


if __name__ == '__main__':
    unittest.main()
