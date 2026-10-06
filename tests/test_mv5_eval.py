"""Shell orchestration tests with a fake evaluator; never load models or use GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class EvaluationScriptTests(unittest.TestCase):
    def test_both_sizes_latest_checkpoints_and_overrides(self):
        source = Path(__file__).resolve().parents[1] / 'mv_5_eval.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(source, root / source.name)
            (root / 'datasets').mkdir()
            for size, stamp in [(50, '20260506T152110'), (100, '20260521T060808')]:
                (root / 'datasets' / f'pdp_{size}.pkl').touch()
                directory = root / f'outputs/mvpdtsp_{size}/mvpdtsp{size}_mv5_makespanw4_log_{stamp}'
                directory.mkdir(parents=True)
                for epoch in (9, 20, 198):
                    (directory / f'epoch-{epoch}.pt').touch()
            fake = root / 'fake_python'
            fake.write_text('''#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
directory = pathlib.Path(args[args.index('--results_dir') + 1])
(directory / 'received_args.json').write_text(json.dumps(args))
print('FAKE evaluation statistics')
''')
            fake.chmod(0o755)
            env = dict(os.environ, PYTHON=str(fake), T_MAX='7', T_MAX_100='11',
                       VAL_SIZE='4', VAL_BATCH_SIZE='2', PRINT_SOLUTION='1', CUDA_VISIBLE_DEVICES='')
            for key in ('MODEL_PATH_50', 'MODEL_PATH_100', 'T_MAX_50'):
                env.pop(key, None)
            subprocess.run(['bash', str(root / source.name)], env=env, cwd='/',
                           check=True, capture_output=True, text=True)
            runs = list((root / 'result_n2s').iterdir())
            self.assertEqual(len(runs), 1)
            for size, steps in [(50, '7'), (100, '11')]:
                directory = runs[0] / f'n2s_{size}_mv5'
                args = json.loads((directory / 'received_args.json').read_text())
                for name, value in [('graph_size', str(size)), ('num_vehicles', '5'),
                                    ('T_max', steps), ('val_size', '4'), ('val_batch_size', '2')]:
                    self.assertEqual(args[args.index('--' + name) + 1], value)
                self.assertTrue(args[args.index('--load_path') + 1].endswith('epoch-198.pt'))
                self.assertIn('--makespan', args)
                self.assertIn('--print_solution', args)
                self.assertIn('FAKE evaluation statistics', (directory / 'evaluation.log').read_text())
                self.assertTrue((directory / 'command.txt').is_file())
            # tee must not hide a failed evaluator; the second configuration must not run.
            fake.write_text('#!/usr/bin/env bash\nexit 17\n')
            failed = subprocess.run(['bash', str(root / source.name)], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 17)
            self.assertNotIn('Evaluating 100 nodes', failed.stdout)


if __name__ == '__main__':
    unittest.main()
