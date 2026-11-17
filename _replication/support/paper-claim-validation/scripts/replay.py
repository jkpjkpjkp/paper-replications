#!/usr/bin/env python3
"""Reuse a project's verifier, redirecting only its report into this audit.

Inputs/checkpoints stay in their original locations. This is checkpoint replay,
not new training and not evidence of agreement with a published table.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper-claim-validation/results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', choices=['prism', 'graph', 'limt', 'hermes'])
    args = parser.parse_args()
    projects = {
        'prism': ('prism-vq-reproduction', 'scripts/verify.py'),
        'graph': ('financial-graph-reproduction', 'scripts/verify_results.py'),
        'limt': ('limt-reproduction', 'scripts/verify.py'),
        'hermes': ('hermes-reproduction', 'scripts/verify_results.py'),
    }
    folder, script = projects[args.project]
    project = ROOT / folder
    sys.path[:0] = [str(project), str(project / 'scripts')]
    os.chdir(project)
    path = project / script
    start = time.time()
    result = dict(project=args.project, verifier=str(path),
                  verifier_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  kind='saved_empirical_forecast_and_checkpoint_replay',
                  published_claim_replicated=False)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            module = runpy.run_path(str(path), run_name='audit_import')
            function = module['verify'] if args.project == 'limt' else module['main']
            env = function.__globals__
            if args.project == 'prism':
                env['save_json'] = lambda _, value: result.update(verifier_result=value)
            else:
                # The other verifiers write one JSON report through Path.write_text.
                # Intercept just that exact report, keeping every input path intact.
                target = project / ({'graph':'results/final_verification.json',
                                     'limt':'results/main/verification.json',
                                     'hermes':'results/verification.json'}[args.project])
                original = Path.write_text
                def redirected(self, data, *a, **kw):
                    if self.resolve() == target.resolve():
                        result['verifier_result'] = json.loads(data)
                        return len(data)
                    raise RuntimeError(f'Unexpected verifier write: {self}')
                Path.write_text = redirected
            function()
        result['passed'] = True
    except BaseException:
        result['passed'] = False
        result['error'] = traceback.format_exc()
    finally:
        if args.project != 'prism' and 'original' in locals():
            Path.write_text = original
        result['elapsed_seconds'] = time.time() - start
        (OUT / f'{args.project}_replay.log').write_text(log.getvalue())
        (OUT / f'{args.project}_replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
