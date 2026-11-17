"""Check archive preservation and zero-copy input staging without research data."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import run
from run import copy_generated, copy_source, link, overlay, prepare


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive'
        self.output = self.root / 'workspace'
        self.archive.mkdir()
        self.output.mkdir()

    def put(self, name, text='original'):
        p = self.archive / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def stage(self, **kwargs):
        overlay(self.archive, self.output, fresh_inputs=kwargs.get('fresh_inputs', False),
                reuse_results=kwargs.get('reuse_results', False))

    def test_inputs_linked_and_source_provenance_available(self):
        self.put('data/input.bin')
        self.put('sources/source-verification.json', '{}')
        self.stage()
        self.assertTrue((self.output / 'data').is_symlink())
        self.assertEqual((self.output / 'data/input.bin').resolve(), self.archive / 'data/input.bin')
        self.assertEqual((self.output / 'sources/source-verification.json').read_text(), '{}')
        self.assertFalse((self.output / 'sources/source-verification.json').is_symlink())

    def test_fresh_mode_omits_prepared_cache(self):
        self.put('data/input.bin')
        self.put('sources/paper.txt')
        self.put('upstream/author/data/raw.npy')
        self.put('reference-rsr/data/prices.csv')
        self.stage(fresh_inputs=True)
        self.assertFalse((self.output / 'data').exists())
        self.assertTrue((self.output / 'sources/paper.txt').exists())
        self.assertTrue((self.output / 'upstream/author/data').is_symlink())
        self.assertTrue((self.output / 'reference-rsr/data').is_symlink())

    def test_fresh_run_has_no_historical_output_links(self):
        paths = ['analysis/summary.csv', 'baseline/model.txt', 'results/predictions.csv',
                 'STATUS.json', 'REPORT.txt', 'verification.json', 'training.log',
                 'run.lock', 'experiments/extension/status.json']
        for name in paths:
            self.put(name)
        self.stage()
        for name in paths:
            self.assertFalse((self.output / name).exists(), name)

    def test_fresh_inputs_rejects_an_existing_archive_cache_link(self):
        self.put('data/input.bin')
        self.stage()
        with self.assertRaisesRegex(ValueError, 'new --workspace'):
            self.stage(fresh_inputs=True)
        self.assertTrue((self.output / 'data').is_symlink())

    def test_replay_outputs_are_independent_writable_copies(self):
        for name in ['analysis/summary.csv', 'baseline/model.txt', 'STATUS.json']:
            self.put(name)
        self.stage(reuse_results=True)
        for name in ['analysis/summary.csv', 'baseline/model.txt', 'STATUS.json']:
            dst = self.output / name
            self.assertFalse(dst.is_symlink())
            dst.write_text('changed')
            self.assertEqual((self.archive / name).read_text(), 'original')

    def test_output_merge_preserves_staged_code_and_current_results(self):
        self.put('analysis/report.py', 'archived code')
        self.put('analysis/summary.csv')
        (self.output / 'analysis').mkdir()
        (self.output / 'analysis/report.py').write_text('current code')
        self.stage(reuse_results=True)
        self.assertEqual((self.output / 'analysis/report.py').read_text(), 'current code')
        (self.output / 'analysis/summary.csv').write_text('current result')
        self.stage(reuse_results=True)
        self.assertEqual((self.output / 'analysis/summary.csv').read_text(), 'current result')

    def test_replayed_symlink_cannot_write_to_archive(self):
        target = self.put('saved_metric.txt')
        (self.archive / 'results').mkdir()
        (self.archive / 'results/metric.txt').symlink_to(target)
        self.stage(reuse_results=True)
        copied = self.output / 'results/metric.txt'
        self.assertFalse(copied.is_symlink())
        copied.write_text('new metric')
        self.assertEqual(target.read_text(), 'original')

    def test_editable_metadata_does_not_mutate_archived_queue(self):
        original = self.put('queue.json', '{"stage": "old"}')
        self.stage()
        staged = self.output / 'queue.json'
        self.assertFalse(staged.is_symlink())
        staged.write_text('{"stage": "new"}')
        self.assertEqual(original.read_text(), '{"stage": "old"}')

    def test_reference_result_inputs_remain_available_without_result_reuse(self):
        self.put('release/results/reference.csv')
        self.put('upstream/checkpoints/author.pt')
        self.put('official/results/benchmark.csv')
        self.put('results/current.csv')
        self.stage()
        for name in ['release/results', 'upstream/checkpoints', 'official/results']:
            self.assertTrue((self.output / name).is_symlink(), name)
        self.assertFalse((self.output / 'results').exists())

    def test_helpers_refuse_to_write_through_archival_parent_link(self):
        sentinel = self.put('keep.txt')
        (self.output / 'linked').symlink_to(self.archive, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'linked parent'):
            link(sentinel, self.output / 'linked/new.txt')
        with self.assertRaisesRegex(ValueError, 'linked parent'):
            overlay(self.archive, self.output / 'linked/child', fresh_inputs=False, reuse_results=False)
        with self.assertRaisesRegex(ValueError, 'linked parent'):
            copy_generated(sentinel, self.output / 'linked/copy.txt')
        self.assertEqual(sorted(p.name for p in self.archive.iterdir()), ['keep.txt'])

    def test_source_symlink_inputs_are_linked_never_copied(self):
        source = self.root / 'source'
        source.mkdir()
        raw = self.put('raw/values.bin')
        (source / 'input-file.bin').symlink_to(raw)
        (source / 'input-dir').symlink_to(raw.parent, target_is_directory=True)
        with patch.object(run, 'ROOT', self.root):
            copy_source(source, self.output / 'paper', self.output, [])
        self.assertTrue((self.output / 'paper/input-file.bin').is_symlink())
        self.assertTrue((self.output / 'paper/input-dir').is_symlink())
        self.assertEqual((self.output / 'paper/input-file.bin').resolve(), raw)

    def test_replay_merge_rejects_nested_output_symlinks(self):
        self.put('results/new.txt')
        (self.output / 'results').mkdir()
        (self.output / 'results/linked').symlink_to(self.archive, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'output contains an input link'):
            self.stage(reuse_results=True)
        self.assertFalse((self.archive / 'new.txt').exists())

    def test_extracted_package_uses_external_archive_and_relocates_own_nested_root(self):
        package = self.root / 'package'
        data_root = self.root / 'shared-data'
        workspace = package / '.replication-local/workspace'
        original = 'paper-reproduction-audit/workspaces/103-healthcare-insurance-fraud'
        source = package / 'healthcare'
        source.mkdir(parents=True)
        code = ("ROOT = '/home/jkp/data/" + original + "'\n"
                "RAW = '/home/jkp/data/crsp-daily/parquet'\n"
                "OTHER = '/home/jkp/data/paper-reproduction-audit/workspaces/999-other'\n"
                "LOCAL = '" + str(package) + "/" + original + "'\n")
        (source / 'run.py').write_text(code)
        registry = {'healthcare': {'source': 'healthcare', 'archive': original,
                                   'aliases': [original]}}
        (package / '_replication').mkdir()
        (package / '_replication/projects.json').write_text(json.dumps(registry))
        external = data_root / '.replication-local/artifacts'
        (external / original / 'data').mkdir(parents=True)
        (external / original / 'data/panel.bin').write_text('cached input')
        support = package / '_replication/support/paper-reproduction-audit'
        support.mkdir(parents=True)
        (support / 'helper.py').write_text('# coordinator code\n')
        (data_root / 'crsp-daily/parquet').mkdir(parents=True)
        args = argparse.Namespace(project='healthcare', workspace=workspace,
                                  data_root=data_root, archive_root=None,
                                  fresh_inputs=False, reuse_results=False)
        with patch.object(run, 'ROOT', package), patch.object(run, 'LOCAL', package / '.replication-local'):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(args)
        staged = (workspace / 'healthcare/run.py').read_text()
        self.assertIn("ROOT = '" + str(workspace / 'healthcare') + "'", staged)
        self.assertIn("LOCAL = '" + str(workspace / 'healthcare') + "'", staged)
        self.assertIn(str(workspace / 'crsp-daily/parquet'), staged)
        self.assertIn(str(workspace / 'paper-reproduction-audit/workspaces/999-other'), staged)
        self.assertTrue((workspace / 'healthcare/data').is_symlink())
        self.assertEqual((workspace / 'healthcare/data').resolve(), external / original / 'data')
        self.assertTrue((workspace / 'paper-reproduction-audit/workspaces').is_symlink())
        self.assertFalse((external / original / 'run.py').exists())
        receipt = json.loads((workspace / '.prepare-healthcare.json').read_text())
        self.assertEqual(receipt['archive_root'], str(external))
        self.assertEqual(receipt['omitted_archival_aliases'], [original])


if __name__ == '__main__':
    unittest.main()
