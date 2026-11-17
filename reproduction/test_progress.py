"""Integrity regressions: stale processes and corrupt receipts cannot pass."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from progress import collect_progress, live_training


class ProgressTests(unittest.TestCase):
    def test_absent_pid_is_not_a_live_job(self):
        self.assertFalse(live_training(None, Path('/no-project'), 'LSTM'))

    def test_checkpoint_corruption_and_wrong_signature_are_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            here = root / 'reproduction'
            folder = here / 'results'
            folder.mkdir(parents=True)
            (here / 'protocol.json').write_text(json.dumps({'seeds': [0, 1, 2, 3, 4]}))
            (folder / 'manifest.json').write_text(json.dumps({'signature': 'correct', 'inputs': {'source': {}}}))
            data = b'frozen checkpoint fixture'
            for kind, signature in [('standard', 'correct'), ('selected', 'wrong')]:
                (folder / f'{kind}-seed0.pt').write_bytes(data)
                (folder / f'{kind}-seed0.receipt.json').write_text(json.dumps({
                    'signature': signature, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}))
            progress = collect_progress(root)
            self.assertEqual(progress['LSTM']['completed_fits'], 1)
            self.assertEqual(progress['LSTM']['invalid_checkpoints'], ['selected-seed0.pt'])
            self.assertFalse(progress['GRU']['live'])
            self.assertEqual(progress['GRU']['completed_fits'], 0)
            (folder / 'standard-seed0.pt').write_bytes(b'corrupt')
            progress = collect_progress(root)
            self.assertEqual(progress['LSTM']['completed_fits'], 0)
            self.assertEqual(len(progress['LSTM']['invalid_checkpoints']), 2)

    def test_pid_reuse_for_another_script_is_rejected(self):
        with patch.object(Path, 'read_text', return_value='1 (python) S 0 0'), \
             patch.object(Path, 'read_bytes', return_value=b'python\0/another/train.py\0--backbone\0LSTM\0'):
            self.assertFalse(live_training(1, Path('/project'), 'LSTM'))


if __name__ == '__main__':
    unittest.main()
