import csv
import json
from pathlib import Path
import tempfile
import unittest

from baseline_summary import summarize_directory, summarize_result


class BaselineSummaryTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "graph_size": 50, "num_vehicles": 2, "val_size": 2,
            "solve_wall_time": 5,
            "instances": [
                {"best_cost": 6, "best_distance_cost": 4,
                 "best_makespan_cost": 2, "solve_time": 3},
                {"best_cost": 10, "best_distance_cost": 6,
                 "best_makespan_cost": 4, "solve_time": 4},
            ],
        }

    def test_statistics_and_timing(self):
        summary = summarize_result(self.result)
        self.assertEqual([summary[k] for k in ("mean_cost", "std_cost", "min_cost", "max_cost")], [8, 2, 6, 10])
        self.assertEqual(summary["mean_distance"], 5)
        self.assertEqual(summary["mean_makespan"], 3)
        self.assertEqual(summary["mean_instance_solve_time_s"], 3.5)
        self.assertEqual(summary["wall_time_per_instance_s"], 2.5)

    def test_backfill_is_idempotent_and_preserves_raw_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(self.result))
            original = path.read_bytes()
            first = summarize_directory(tmp)
            self.assertEqual(summarize_directory(tmp), first)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads((Path(tmp) / "experiment_summary.json").read_text()), first)
            with (Path(tmp) / "experiment_summary.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["mean_cost"]), 8)

    def test_incomplete_results_are_rejected(self):
        self.result["instances"].pop()
        with self.assertRaises(ValueError):
            summarize_result(self.result)


if __name__ == "__main__":
    unittest.main()
