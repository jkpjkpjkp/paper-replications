"""Checks for exclusions and publication isolation; no external services."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_upload as upload


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'paper'
        self.repo.mkdir()
        upload.init(self.repo, 'original')
        upload.git(self.repo, 'config', 'user.name', 'Upload test')
        upload.git(self.repo, 'config', 'user.email', 'test@example.invalid')

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, content='source\n'):
        p = self.repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def policy(self, **extra):
        policy = (Path(upload.__file__).parent / 'upload.gitignore').read_text()
        with patch.object(upload, 'ROOT', self.root):
            upload.install_rules({'path': 'paper', **extra}, policy)

    def test_data_excluded_but_source_packages_preserved(self):
        self.policy(source_data_dirs=['upstream/alphagen/data'])
        for p in ['data/raw.txt', 'deep/datasets/raw.txt', 'LLM/a.py', 'CRYPTO/a.py',
                  'weights/model.pth', 'x.npy', '.env', 'results/report.json', '.venv/a.py',
                  'upstream/alphagen/data/prices.parquet']:
            self.file(p)
        self.file('upstream/alphagen/data/expression.py')
        self.file('upstream/alphagen/rl/env/core.py')
        self.file('train.py')
        paths, _, _ = upload.selected(self.repo)
        self.assertEqual(set(paths), {'.gitignore', 'train.py',
                         'upstream/alphagen/data/expression.py', 'upstream/alphagen/rl/env/core.py'})

    def test_snapshot_preserves_existing_git_and_omits_history(self):
        self.file('data/old.csv', 'old proprietary input\n')
        self.file('train.py', 'old source\n')
        upload.git(self.repo, 'add', '.')
        upload.git(self.repo, 'commit', '-qm', 'Original commit with data')
        self.file('train.py', 'staged edit\n')
        upload.git(self.repo, 'add', 'train.py')
        self.file('train.py', 'unstaged edit\n')
        upload.git(self.repo, 'remote', 'add', 'origin', 'https://example.invalid/upstream')
        self.policy()
        nested = self.repo / 'upstream'
        nested.mkdir()
        upload.init(nested, 'main')
        self.file('upstream/model.py')
        os.symlink('/tmp', self.repo / 'external')
        paths, links, _ = upload.selected(self.repo)
        before_head = upload.textgit(self.repo, 'rev-parse', 'HEAD')
        before_index = (self.repo / '.git/index').read_bytes()
        before_remote = upload.textgit(self.repo, 'remote', '-v')
        sha = upload.snapshot(self.repo, 'paper/test', paths, False)
        self.assertEqual(before_head, upload.textgit(self.repo, 'rev-parse', 'HEAD'))
        self.assertEqual(before_index, (self.repo / '.git/index').read_bytes())
        self.assertEqual(before_remote, upload.textgit(self.repo, 'remote', '-v'))
        self.assertEqual(upload.textgit(self.repo, 'rev-list', '--count', sha), '1')
        tree = upload.textgit(self.repo, 'ls-tree', '-r', sha)
        self.assertNotIn('160000', tree)
        self.assertNotIn('data/old.csv', tree)
        self.assertIn('upstream/model.py', tree)
        self.assertIn('external', links)
        self.assertEqual(sha, upload.snapshot(self.repo, 'paper/test', paths, False))
        self.assertEqual(upload.textgit(self.repo, 'show', sha + ':train.py'), 'unstaged edit')

    def test_inspection_rejects_large_files_and_tokens(self):
        self.file('secret.txt', 'ghp_' + 'a' * 36)
        with self.assertRaisesRegex(RuntimeError, 'credential'):
            upload.inspect(self.repo, ['secret.txt'])
        p = self.file('large.txt')
        with p.open('r+b') as f:
            f.truncate(25 * 1024 * 1024)
        with self.assertRaisesRegex(RuntimeError, 'large source'):
            upload.inspect(self.repo, ['large.txt'])

    def test_active_publication_branch_keeps_clean_index(self):
        upload.git(self.repo, 'symbolic-ref', 'HEAD', 'refs/heads/paper/test')
        self.file('code.py', 'first\n')
        upload.snapshot(self.repo, 'paper/test', ['code.py'], True)
        self.file('code.py', 'second\n')
        upload.snapshot(self.repo, 'paper/test', ['code.py'], False)
        self.assertEqual(upload.textgit(self.repo, 'status', '--porcelain'), '')
        self.file('code.py', 'staged\n')
        upload.git(self.repo, 'add', 'code.py')
        staged = (self.repo / '.git/index').read_bytes()
        self.file('code.py', 'third\n')
        upload.snapshot(self.repo, 'paper/test', ['code.py'], False)
        self.assertEqual(staged, (self.repo / '.git/index').read_bytes())


if __name__ == '__main__':
    unittest.main()
