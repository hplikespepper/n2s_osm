"""Run in the ortools environment: python -m unittest discover -s tests -p test_ortools_baseline.py."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from problems.problem_mvpdtsp import MVPDTSP


class OrtoolsTests(unittest.TestCase):
    def test_serial_parallel_timing_and_feasibility(self):
        root = Path(__file__).resolve().parents[1]
        problem = MVPDTSP(100, num_vehicles=5, with_assert=True, use_makespan=True)
        with tempfile.TemporaryDirectory() as tmp:
            for workers in (1, 2):
                with self.subTest(workers=workers):
                    # A bare output filename also exercises output directory handling.
                    filename = f"result_{workers}.json"
                    subprocess.run([
                        sys.executable, str(root / "ortools_baseline.py"),
                        "--val_dataset", str(root / "datasets/pdp_100.pkl"),
                        "--graph_size", "100", "--num_vehicles", "5",
                        "--val_size", "4", "--time_limit", "2",
                        "--num_workers", str(workers), "--output", filename,
                    ], cwd=tmp, check=True, capture_output=True, text=True, timeout=120)
                    result = json.loads((Path(tmp) / filename).read_text())
                    self.assertEqual(result["num_workers"], workers)
                    self.assertEqual(result["makespan_weight"], 4)
                    self.assertEqual(result["time_limit"], 2)
                    instances = result["instances"]
                    self.assertEqual([x["instance_id"] for x in instances], list(range(4)))
                    self.assertEqual(len({x["worker_pid"] for x in instances}), workers)
                    self.assertGreaterEqual(result["solve_wall_time"], max(x["solve_time"] for x in instances))
                    for x in instances:
                        self.assertGreater(x["solve_time"], 0)
                        self.assertEqual(x["best_cost"], x["best_distance_cost"] + 4 * x["best_makespan_cost"])
                        self.assertEqual(len(x["vehicle_routes"]), 5)
                        self.assertEqual(sorted(n for r in x["vehicle_routes"] for n in r[1:-1]), list(range(5, 105)))
                        cost = problem.get_costs(
                            {"coordinates": torch.tensor([x["coordinates"]])},
                            torch.tensor([x["best_rec"]]),
                        ).item()
                        self.assertAlmostEqual(cost, x["best_cost"], delta=2e-5)
                    # Time-limited search need not choose identical routes in parallel.


if __name__ == "__main__":
    unittest.main()
