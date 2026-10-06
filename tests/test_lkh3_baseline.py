"""Run with: python -m unittest discover -s tests -p test_lkh3_baseline.py."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lkh3_baseline import (
    DEFAULT_LKH,
    build_rec,
    compute_costs,
    lkh_coordinates,
    objective_cost,
    solve_instance,
    write_pdptw_problem,
)
from problems.problem_mvpdtsp import MVPDTSP


class LKH3BaselineTests(unittest.TestCase):
    def setUp(self):
        # Two depot clones, three pickups, then their three deliveries.
        self.coords = np.array([
            [0.5, 0.5], [0.5, 0.5],
            [0.1, 0.2], [0.8, 0.1], [0.2, 0.9],
            [0.2, 0.3], [0.7, 0.2], [0.3, 0.8],
        ], dtype=np.float32)

    def test_problem_writer_removes_depot_clones_and_pairs_nodes(self):
        compact = lkh_coordinates(self.coords, 2)
        self.assertEqual(compact.shape, (7, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instance.pdptw"
            write_pdptw_problem(path, self.coords, 2, 1e6)
            text = path.read_text()
        self.assertIn("TYPE : PDPTW", text)
        self.assertIn("DIMENSION : 7", text)
        self.assertIn("VEHICLES : 2", text)
        self.assertIn("2 1 0 ", text)
        self.assertIn(" 0 0 5", text)
        self.assertIn("5 -1 0 ", text)
        self.assertIn(" 0 2 0", text)

    def test_build_rec_handles_empty_vehicle(self):
        routes = [[0, 2, 5, 0], [1, 1]]
        rec = build_rec(2, 2, routes)
        self.assertEqual(rec, [2, 1, 5, 3, 4, 0])

    @unittest.skipUnless(DEFAULT_LKH.is_file(), "LKH-3.0.14 executable not found")
    def test_real_lkh_solution_is_mvn2s_feasible(self):
        (rec, routes, distance, makespan, vehicle_costs, selected_seed,
         selected_limit, route_limits, failures, candidate_details) = solve_instance(
            coords=self.coords,
            graph_size=6,
            num_vehicles=2,
            lkh_path=DEFAULT_LKH,
            scale=1e6,
            seeds=[101, 102],
            max_trials=100,
            start_workers=2,
        )
        self.assertEqual(len(routes), 2)
        self.assertEqual(sorted(node for route in routes for node in route[1:-1]), list(range(2, 8)))
        self.assertIn(selected_seed, (101, 102))
        self.assertEqual(len(route_limits), 2)
        self.assertIsNone(route_limits[0])
        self.assertIn(selected_limit, route_limits)
        self.assertLess(len(failures), 2)
        self.assertEqual(len(candidate_details), 2)
        self.assertEqual(sum(item["selected"] for item in candidate_details), 1)
        problem = MVPDTSP(6, num_vehicles=2, with_assert=True, use_makespan=True)
        problem.check_feasibility(torch.tensor([rec]))
        d2, m2, vehicle2 = compute_costs(problem, self.coords, rec)
        self.assertAlmostEqual(distance, d2)
        self.assertAlmostEqual(makespan, m2)
        self.assertEqual(vehicle_costs, vehicle2)
        self.assertAlmostEqual(objective_cost(distance, makespan, 2), distance + makespan)


if __name__ == "__main__":
    unittest.main()
