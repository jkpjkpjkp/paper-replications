"""Recovery and audit guards for the resumable data coordinator; no network."""
import json
from pathlib import Path
from subprocess import CompletedProcess
import tempfile
import unittest
from unittest.mock import patch

import run_full_download as runner


class SupervisorRecovery(unittest.TestCase):
    def execute(self, outcomes, complete):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'validation.json').write_text(json.dumps({'complete':complete}))
            with patch.object(runner,'ROOT',root), patch.object(runner.time,'sleep'), \
                 patch.object(runner.subprocess,'run',side_effect=[CompletedProcess([],code) for code in outcomes]) as run:
                code=runner.main(['--workers','3'])
                state=json.loads((root/'run_status.json').read_text())
                calls=[call.args[0] for call in run.call_args_list]
        return code,state,calls

    def test_failed_parallel_attempt_reuses_download_at_lower_concurrency(self):
        code,state,calls=self.execute([1,0,0],True)
        self.assertEqual(code,0)
        self.assertEqual([call[-1] for call in calls[:2]],['3','2'])
        self.assertTrue(calls[-1][-1].endswith('validate_download.py'))
        self.assertEqual(state['state'],'complete')

    def test_no_audit_or_success_claim_after_all_download_attempts_fail(self):
        code,state,calls=self.execute([1,1,1],True)
        self.assertEqual(code,1)
        self.assertEqual([call[-1] for call in calls],['3','2','1'])
        self.assertEqual(state['state'],'failed')

    def test_zero_exit_code_does_not_override_incomplete_independent_audit(self):
        code,state,calls=self.execute([0,0],False)
        self.assertEqual(code,1)
        self.assertFalse(state['complete'])
        self.assertEqual(state['state'],'validation_failed')


if __name__=='__main__':
    unittest.main()
