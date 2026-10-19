import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.utils import _atomic_torch_save


class AtomicValidationShardTests(unittest.TestCase):
    def test_final_path_is_hidden_until_archive_is_complete(self):
        with TemporaryDirectory() as tmp_dir:
            final_path = Path(tmp_dir) / 'validation_rank_1.pt'
            write_started = threading.Event()
            allow_finish = threading.Event()
            errors = []
            real_torch_save = torch.save

            def delayed_save(obj, path):
                with open(path, 'wb') as handle:
                    handle.write(b'incomplete archive')
                    handle.flush()
                write_started.set()
                if not allow_finish.wait(timeout=5):
                    raise RuntimeError('test timed out waiting to finish save')
                real_torch_save(obj, path)

            def save_in_thread():
                try:
                    _atomic_torch_save({'rank': 1, 'value': torch.tensor([3])}, str(final_path))
                except Exception as exc:
                    errors.append(exc)

            with patch('agent.utils.torch.save', side_effect=delayed_save):
                worker = threading.Thread(target=save_in_thread)
                worker.start()
                self.assertTrue(write_started.wait(timeout=5))
                self.assertFalse(final_path.exists())
                allow_finish.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(torch.load(str(final_path), map_location='cpu')['rank'], 1)
            self.assertEqual(list(Path(tmp_dir).glob('*.tmp.*')), [])

    def test_failed_save_removes_temporary_file(self):
        with TemporaryDirectory() as tmp_dir:
            final_path = Path(tmp_dir) / 'validation_rank_1.pt'

            def failed_save(_obj, path):
                Path(path).write_bytes(b'incomplete archive')
                raise OSError('simulated write failure')

            with patch('agent.utils.torch.save', side_effect=failed_save):
                with self.assertRaisesRegex(OSError, 'simulated write failure'):
                    _atomic_torch_save({'rank': 1}, str(final_path))

            self.assertFalse(final_path.exists())
            self.assertEqual(list(Path(tmp_dir).glob('*.tmp.*')), [])


if __name__ == '__main__':
    unittest.main()
