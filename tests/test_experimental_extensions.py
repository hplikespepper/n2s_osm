import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.ppo import PPO
from nets.actor_network import Actor
from options import get_options
from problems.problem_mvpdtsp import MVPDTSP
from problems.problem_mvpdtsp_fixed import MVPDTSPFixedAssignment


class ObjectiveTests(unittest.TestCase):
    def test_three_objectives_and_k1_degeneracy(self):
        distance = torch.tensor([10.0])
        makespan = torch.tensor([4.0])
        expected = {
            'distance': 10.0,
            'only_makespan': 4.0,
            'proposed': 26.0,
        }
        for objective, value in expected.items():
            problem = MVPDTSP(4, num_vehicles=5, objective=objective)
            self.assertEqual(problem.objective_cost(distance, makespan).item(), value)
        k1 = MVPDTSP(4, num_vehicles=1, objective='proposed')
        self.assertEqual(k1.objective_cost(distance, makespan).item(), 10.0)

    @patch('options.torch.cuda.device_count', return_value=0)
    def test_objective_cli_and_legacy_alias(self, _):
        opts = get_options(['--problem', 'mvpdtsp', '--makespan'])
        self.assertEqual(opts.objective, 'proposed')
        opts = get_options(['--problem', 'mvpdtsp', '--objective', 'only_makespan'])
        self.assertEqual(opts.objective, 'only_makespan')
        with self.assertRaises(SystemExit):
            get_options([
                '--problem', 'mvpdtsp', '--makespan',
                '--objective', 'distance',
            ])


class FixedAssignmentTests(unittest.TestCase):
    def setUp(self):
        # vehicle 0: 0 -> 2 -> 4 -> 0; vehicle 1: 1 -> 3 -> 5 -> 1
        self.rec = torch.tensor([[2, 3, 4, 5, 0, 1]], dtype=torch.long)
        self.visited_order_map = torch.zeros((1, 6, 6), dtype=torch.bool)

    def test_cross_vehicle_moves_only_masked_in_fixed_variant(self):
        base = MVPDTSP(4, num_vehicles=2, objective='proposed')
        fixed = MVPDTSPFixedAssignment(4, num_vehicles=2, objective='proposed')
        selected = torch.tensor([2])
        base_mask = base.get_swap_mask(selected, self.visited_order_map, rec=self.rec)
        fixed_mask = fixed.get_swap_mask(selected, self.visited_order_map, rec=self.rec)
        newly_masked = (~base_mask) & fixed_mask
        self.assertTrue(newly_masked.any())

    def test_repeated_legal_moves_preserve_initial_assignment(self):
        problem = MVPDTSPFixedAssignment(
            4, num_vehicles=2, objective='proposed', with_assert=True
        )
        rec = self.rec.clone()
        initial_vehicle = problem.get_vehicle_id(rec)[:, problem.get_pickup_indices()]
        for pair_offset in (0, 1, 0, 1):
            selected = torch.tensor([problem.pickup_start + pair_offset])
            mask = problem.get_swap_mask(selected, self.visited_order_map, rec=rec)
            legal = (~mask[0]).nonzero(as_tuple=False)
            self.assertGreater(len(legal), 0)
            first, second = legal[-1]
            rec = problem.insert_star(
                rec,
                selected.view(1, 1),
                first.view(1, 1),
                second.view(1, 1),
            )
            problem.check_feasibility(rec)
        final_vehicle = problem.get_vehicle_id(rec)[:, problem.get_pickup_indices()]
        self.assertTrue(torch.equal(initial_vehicle, final_vehicle))


class ComponentHistoryTests(unittest.TestCase):
    def test_history_steps_monotonicity_and_components(self):
        opts = SimpleNamespace(
            num_vehicles=2,
            embedding_dim=32,
            hidden_dim=32,
            actor_head_num=4,
            critic_head_num=4,
            n_encode_layers=1,
            normalization='layer',
            v_range=6.0,
            eval_only=True,
            use_cuda=False,
            distributed=False,
            device=torch.device('cpu'),
            T_max=3,
            no_progress_bar=True,
            record_component_history=True,
            history_interval=2,
        )
        problem = MVPDTSP(4, num_vehicles=2, init_val_met='random', objective='proposed')
        agent = PPO(problem.NAME, problem.size, opts)
        agent.eval()
        batch = {'coordinates': torch.rand(2, 6, 2)}
        output = agent.rollout(problem, 1, batch, do_sample=False)
        history = output[-1]
        self.assertEqual(history['steps'], [0, 2, 3])
        self.assertTrue((history['objective'][:, 1:] <= history['objective'][:, :-1] + 1e-7).all())
        recomputed = history['distance'] + history['makespan']
        self.assertTrue(torch.allclose(history['objective'], recomputed, atol=1e-6))

    def test_history_is_disabled_by_default(self):
        opts = SimpleNamespace(
            num_vehicles=2,
            embedding_dim=32,
            hidden_dim=32,
            actor_head_num=4,
            critic_head_num=4,
            n_encode_layers=1,
            normalization='layer',
            v_range=6.0,
            eval_only=True,
            use_cuda=False,
            distributed=False,
            device=torch.device('cpu'),
            T_max=1,
            no_progress_bar=True,
            record_component_history=False,
            history_interval=100,
        )
        problem = MVPDTSP(4, num_vehicles=2, init_val_met='random', objective='proposed')
        agent = PPO(problem.NAME, problem.size, opts)
        agent.eval()
        output = agent.rollout(problem, 1, {'coordinates': torch.rand(2, 6, 2)})
        self.assertIsNone(output[-1])


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_pdtsp_checkpoint_loads_for_k1_mvpdtsp(self):
        checkpoint = ROOT / 'pre-trained/pdtsp/20/epoch-156.pt'
        if not checkpoint.is_file():
            self.skipTest('Bundled PDTSP checkpoint is unavailable')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['actor']
        actor = Actor(
            problem_name='mvpdtsp', embedding_dim=128, hidden_dim=128,
            n_heads_actor=4, n_layers=3, normalization='layer',
            v_range=6.0, seq_length=21,
        )
        actor.load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()
