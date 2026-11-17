"""Data gates and GPU resource recovery without starting GPU or remote jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import supervise


class SupervisorRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        here = root / 'reproduction'
        (here / 'results').mkdir(parents=True)
        (root / 'validation.json').write_text(json.dumps({'complete': True, 'summary': {'integrity_error_count': 0}}))
        args = argparse.Namespace(root=root, poll_seconds=0, device='cuda', threads=2,
                                  minimum_cuda_free_mib=2800)
        return here, args

    def test_waits_for_capacity_then_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, args = self.fixture(Path(tmp))
            calls = []
            def run(cmd, **kwargs):
                calls.append(Path(cmd[1]).name)
                if calls[-1] == 'train.py':
                    (here / 'results/summary.json').write_text('{"complete":true}')
                return subprocess.CompletedProcess(cmd, 0)
            with patch.object(supervise, 'write_report'), patch.object(supervise, 'other_replication_running', return_value=False), \
                 patch.object(supervise, 'cuda_free_mib', side_effect=[100, 3000]) as memory, \
                 patch.object(supervise.time, 'sleep') as sleep, patch.object(supervise.subprocess, 'run', side_effect=run):
                self.assertEqual(supervise.main(args), 0)
            self.assertEqual(calls, ['build_tensors.py', 'train.py'])
            self.assertEqual(memory.call_count, 2)
            sleep.assert_called_once()
            self.assertTrue(json.loads((here / 'pipeline_status.json').read_text())['complete'])

    def test_retries_only_cuda_oom_and_preserves_completed_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, args = self.fixture(Path(tmp))
            completed = here / 'results/completed-seed.pt'
            completed.write_bytes(b'completed fit fixture')
            attempts = []
            def run(cmd, **kwargs):
                if Path(cmd[1]).name == 'train.py':
                    attempts.append(1)
                    self.assertEqual(completed.read_bytes(), b'completed fit fixture')
                    if len(attempts) == 1:
                        (here / 'results/status.json').write_text('{"phase":"failed","error_type":"OutOfMemoryError"}')
                        return subprocess.CompletedProcess(cmd, 1)
                    (here / 'results/summary.json').write_text('{"complete":true}')
                return subprocess.CompletedProcess(cmd, 0)
            with patch.object(supervise, 'write_report'), patch.object(supervise, 'other_replication_running', return_value=False), \
                 patch.object(supervise, 'cuda_free_mib', return_value=3000), patch.object(supervise.time, 'sleep'), \
                 patch.object(supervise.subprocess, 'run', side_effect=run):
                self.assertEqual(supervise.main(args), 0)
            self.assertEqual(len(attempts), 2)

    def test_does_not_repeat_nonresource_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, args = self.fixture(Path(tmp))
            (here / 'results/status.json').write_text('{"phase":"failed","error_type":"ValueError"}')
            calls = []
            def run(cmd, **kwargs):
                calls.append(Path(cmd[1]).name)
                return subprocess.CompletedProcess(cmd, 2 if calls[-1] == 'train.py' else 0)
            with patch.object(supervise, 'write_report'), patch.object(supervise, 'other_replication_running', return_value=False), \
                 patch.object(supervise, 'cuda_free_mib', return_value=3000), patch.object(supervise.subprocess, 'run', side_effect=run):
                self.assertEqual(supervise.main(args), 2)
            self.assertEqual(calls, ['build_tensors.py', 'train.py'])
            self.assertEqual(json.loads((here / 'pipeline_status.json').read_text())['phase'], 'failed')

    def test_incomplete_failed_data_never_launches_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, args = self.fixture(Path(tmp))
            (args.root / 'validation.json').write_text('{"complete":false}')
            (args.root / 'run_status.json').write_text('{"state":"failed"}')
            with patch.object(supervise, 'write_report'), patch.object(supervise.subprocess, 'run') as run:
                self.assertEqual(supervise.main(args), 1)
            run.assert_not_called()

    def test_backbone_results_are_separate_and_each_comparison_must_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            here, args = self.fixture(Path(tmp))
            args.backbones = 'LSTM,GRU'
            fitted = []
            def run(cmd, **kwargs):
                if Path(cmd[1]).name == 'train.py':
                    name = cmd[cmd.index('--backbone') + 1]
                    fitted.append(name)
                    output = here / 'results'
                    if name != 'LSTM':
                        output = output / 'backbones' / name.lower()
                    output.mkdir(parents=True, exist_ok=True)
                    (output / 'summary.json').write_text('{"complete":true}')
                return subprocess.CompletedProcess(cmd, 0)
            with patch.object(supervise, 'write_report'), patch.object(supervise, 'other_replication_running', return_value=False), \
                 patch.object(supervise, 'cuda_free_mib', return_value=3000), patch.object(supervise.subprocess, 'run', side_effect=run):
                self.assertEqual(supervise.main(args), 0)
            self.assertEqual(fitted, ['LSTM', 'GRU'])
            final = json.loads((here / 'pipeline_status.json').read_text())
            self.assertEqual(final['completed_backbones'], ['LSTM', 'GRU'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
