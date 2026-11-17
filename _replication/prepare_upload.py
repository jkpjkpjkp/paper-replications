#!/usr/bin/env python3
"""Prepare independent source-only paper branches for one GitHub repository.

Never pushes. Original branches, staged work, remotes and history are preserved.
Publication commits use an isolated index and have no original-history parents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / '_replication/upload_projects.json'
BEGIN = '# BEGIN managed upload rules'
END = '# END managed upload rules'
PRUNE = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
SECRET = re.compile(rb'(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
                    rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|'
                    rb'AKIA[0-9A-Z]{16}|xox[baprs]-[0-9A-Za-z-]{20,})')


def git(repo, *args, data=None, env=None, check=True):
    p = subprocess.run(['git', '-C', str(repo), *args], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if check and p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace').strip())
    return p


def textgit(repo, *args, **kw):
    return git(repo, *args, **kw).stdout.decode().strip()


def init(repo, branch):
    existed = (repo / '.git').exists()
    if not existed:
        git(repo, 'init', '-q', '-b', branch)
    return existed


def install_rules(project, policy):
    repo = ROOT / project['path']
    f = repo / '.gitignore'
    previous = f.read_text() if f.exists() else ''
    previous = re.sub(r'(?ms)^' + re.escape(BEGIN) + r'\n.*?^' +
                      re.escape(END) + r'\n?', '', previous).rstrip()
    extra = project.get('ignore', [])
    allow = []
    for path in project.get('source_data_dirs', []):
        # These exact directories contain Python modules/configuration, not inputs.
        allow.extend(['!/' + path + '/', '!/' + path + '/**/'])
        for suffix in ('py', 'pyx', 'pxd', 'cpp', 'h', 'yaml', 'yml', 'json', 'toml', 'md', 'txt'):
            allow.append('!/' + path + '/**/*.' + suffix)
    allow += ['!/' + p for p in project.get('source_files', [])]
    block = '\n'.join([BEGIN, policy.rstrip(), *extra, *allow, END]) + '\n'
    f.write_text((previous + '\n\n' if previous else '') + block)


def ignored(repo, names):
    if not names:
        return set()
    p = git(repo, 'check-ignore', '--no-index', '-z', '--stdin',
            data=b'\0'.join(os.fsencode(n) for n in names) + b'\0', check=False)
    if p.returncode not in (0, 1):
        raise RuntimeError(p.stderr.decode())
    return {os.fsdecode(n) for n in p.stdout.split(b'\0') if n}


def selected(repo):
    """Honor every .gitignore, including previously tracked paths; flatten Git dirs."""
    files, omitted_links, omitted_dirs = [], [], []
    for parent, ds, fs in os.walk(repo, followlinks=False):
        parent = Path(parent)
        dirs = []
        for name in ds:
            p = parent / name
            rel = p.relative_to(repo).as_posix()
            if p.is_symlink():
                omitted_links.append(rel)
            elif name in PRUNE or name.startswith('.venv'):
                omitted_dirs.append(rel)
            else:
                dirs.append(rel + '/')
        excluded = ignored(repo, dirs)
        omitted_dirs.extend(sorted(excluded))
        ds[:] = [Path(d.rstrip('/')).name for d in dirs if d not in excluded]
        names = [(parent / n).relative_to(repo).as_posix() for n in fs if n != '.git']
        excluded = ignored(repo, names)
        for name in names:
            p = repo / name
            if name in excluded:
                continue
            if p.is_symlink():
                omitted_links.append(name)
            elif p.is_file():
                files.append(name)
    return sorted(files), sorted(omitted_links), sorted(omitted_dirs)


def inspect(repo, files):
    records = []
    for name in files:
        p = repo / name
        size = p.stat().st_size
        if size >= 25 * 1024 * 1024:
            raise RuntimeError(f'Unexpected large source file: {p} ({size} bytes)')
        raw = p.read_bytes()
        if SECRET.search(raw):
            raise RuntimeError(f'Potential credential in source selection: {p}')
        records.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return records


def snapshot(repo, branch, files, new_repo):
    """Hash explicit files: nested upstream .git directories never become gitlinks."""
    gitdir = Path(textgit(repo, 'rev-parse', '--absolute-git-dir'))
    with tempfile.TemporaryDirectory(prefix='upload-index-', dir=gitdir) as tmp:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(tmp) / 'index')}
        git(repo, 'read-tree', '--empty', env=env)
        quoted = ''.join(json.dumps(str(repo / n), ensure_ascii=False) + '\n' for n in files)
        hashes = git(repo, 'hash-object', '-w', '--no-filters', '--stdin-paths',
                     data=quoted.encode()).stdout.decode().splitlines()
        entries = b''
        for name, sha in zip(files, hashes, strict=True):
            mode = '100755' if (repo / name).stat().st_mode & 0o111 else '100644'
            entries += f'{mode} {sha}\t'.encode() + os.fsencode(name) + b'\0'
        git(repo, 'update-index', '-z', '--index-info', data=entries, env=env)
        tree = textgit(repo, 'write-tree', env=env)
        ref = 'refs/heads/' + branch
        old = textgit(repo, 'rev-parse', '--verify', ref, check=False)
        if old and textgit(repo, 'rev-parse', ref + '^{tree}') == tree:
            return old
        # Keep a clean index on the active publication branch clean. Never
        # replace staged work or the index of an original/upstream branch.
        sync_index = new_repo
        if old and textgit(repo, 'symbolic-ref', '--quiet', 'HEAD', check=False) == ref:
            check_index = Path(tmp) / 'existing-index'
            if (gitdir / 'index').exists():
                shutil.copyfile(gitdir / 'index', check_index)
                check_env = {**os.environ, 'GIT_INDEX_FILE': str(check_index)}
                sync_index = textgit(repo, 'write-tree', env=check_env) == textgit(repo, 'rev-parse', old + '^{tree}')
        parents = ['-p', old] if old else []
        commit = textgit(repo, 'commit-tree', tree, *parents,
                         data=f'Prepare source-only upload: {branch}\n'.encode())
        git(repo, 'update-ref', ref, commit, old or '0' * 40)
        if sync_index:
            git(repo, 'read-tree', commit)
        return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true', help='Inspect current selections without creating commits')
    args = parser.parse_args()
    config = json.loads(MANIFEST.read_text())
    projects = config['projects']
    policy = (ROOT / '_replication/upload.gitignore').read_text()
    root_new = not (ROOT / '.git').exists()
    init(ROOT, 'main')
    report = {'repository': config['repository'], 'projects': [], 'excluded_roots': config['excluded_roots']}
    for project in projects:
        repo = ROOT / project['path']
        if not repo.is_dir() or repo.is_symlink():
            raise RuntimeError(f'Missing or linked project: {repo}')
        branch = project['branch']
        new_repo = not (repo / '.git').exists()
        init(repo, branch)
        if not args.audit_only:
            install_rules(project, policy)
        files, links, dirs = selected(repo)
        records = inspect(repo, files)
        item = {**project, 'file_count': len(files), 'source_bytes': sum(r['bytes'] for r in records),
                'omitted_symlinks': links, 'ignored_directories': dirs, 'files': records}
        if not args.audit_only:
            item['commit'] = snapshot(repo, branch, files, new_repo)
            git(ROOT, 'fetch', '--no-tags', '--quiet', str(repo),
                f'refs/heads/{branch}:refs/heads/{branch}')
        report['projects'].append(item)
        print(f"{branch}: {len(files)} files, {item['source_bytes'] / 1e6:.2f} MB", flush=True)
    files, links, dirs = selected(ROOT)
    # The main catalogue must never acquire a paper or dataset through an ignore edit.
    allowed = set(config['main_paths'])
    if any(Path(n).parts[0] not in allowed for n in files):
        raise RuntimeError('Unexpected path in main catalogue selection')
    records = inspect(ROOT, files)
    report['main'] = {'file_count': len(files), 'source_bytes': sum(r['bytes'] for r in records), 'files': records}
    if not args.audit_only:
        report['main']['commit'] = snapshot(ROOT, 'main', files, root_new)
    report_path = ROOT / '.replication-local/upload/report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Local audit: {report_path}')
    print('No remote was created and nothing was pushed.')


if __name__ == '__main__':
    main()
