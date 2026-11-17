#!/usr/bin/env python3
"""Stage a runnable workspace without putting inputs or outputs in source folders."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.replication-local'
RAW = ['Chen_Zimmermann_Factor_Zoo_2026_10', 'compustat', 'crsp-daily',
       'crsp-monthly', 'csi', 'factors-daily', 'global-market-data', 'ibes',
       'ravenpack', 'sp500-earnings-transcripts', 'wrds-crsp-download', 'song26l-us']
SKIP = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
OUTPUTS = {'results', 'logs', 'checkpoints', 'figures', 'outputs', 'reports',
           'report', 'artifacts', 'analysis-results', 'analysis', 'baseline', 'tmp'}
CACHES = {'data', 'cache', 'asset_data', 'inputs', 'raw', 'bars', 'support'}
GENERATED_NAMES = {
    'status.json', 'status.txt', 'status.md', 'report.txt', 'report.md',
    'report.html', 'results.txt', 'results.md', 'results.json', 'result.txt',
    'run_status.json', 'completion.json', 'complete.json', '.complete',
    '.suite-complete', 'preflight.json', 'locked_before_training.json',
    'locked-before-training.json', 'locked_before_evaluation.json',
}
REFERENCE_DIRS = {'sources', 'release', 'reference', 'official', 'author-code'}
METADATA_SUFFIXES = {'.json', '.yaml', '.yml', '.toml', '.conf', '.cfg', '.ini', '.md'}
TEXT = {'.py', '.sh', '.json', '.toml', '.yaml', '.yml', '.conf', '.cfg', '.ini', '.txt', '.md', '.service'}
ALIASES = {
    'error-propagation-reproduction': 'MDP Pricing',
    'expanding-chaos-reproduction': 'PDE', '2505.11243': 'Set Sequence',
    'DRO': 'DRO', 'portfoliopilot-reproduction': 'Portfolio Pilot',
    'alpha-research-reproduction': 'alpha-research',
    'stable-reproduction': 'stable',
    'sparse-tangent-original-reproduction': 'blocked/sparse-tangent',
}


def catalogue():
    path = ROOT / '_replication/projects.json'
    return json.loads(path.read_text()) if path.exists() else {}


def symlink_directory(path):
    """Find a directory link through which a write would escape the workspace."""
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            return candidate
    return None


def require_real_parent(destination):
    linked_parent = symlink_directory(destination.parent)
    if linked_parent is not None:
        raise ValueError(f'Refusing to write through linked parent {linked_parent}: {destination}')


def link(source, destination):
    if destination.exists() or destination.is_symlink():
        return
    require_real_parent(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def copy_source(source, destination, workspace, changes, *, own_archive=None, own_aliases=()):
    """Copy small source files; never walk shared data, environments or archives."""
    if source.is_symlink():
        link(source, destination)
        return
    require_real_parent(destination)
    if destination.is_symlink():
        raise ValueError(f'Project destination is an input link; use a new workspace: {destination}')
    for parent, dirs, files in os.walk(source):
        if '.git' in dirs:
            link(Path(parent) / '.git', destination / Path(parent).relative_to(source) / '.git')
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith('.venv')]
        for directory in list(dirs):
            src = Path(parent) / directory
            if src.is_symlink():
                link(src, destination / src.relative_to(source))
                dirs.remove(directory)
        for name in files:
            src = Path(parent) / name
            if src.is_symlink():
                # Source symlinks may point to private inputs. Restore them only
                # in the runtime, without copying their targets into the release.
                link(src, destination / src.relative_to(source))
                continue
            dst = destination / src.relative_to(source)
            if dst.exists() or dst.is_symlink():
                continue  # never overwrite a staged experiment silently
            require_real_parent(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            frozen_config = source.name == 'arboreal' and src.name == 'config.json'
            if not frozen_config and (src.suffix in TEXT or src.name == 'Makefile'):
                try:
                    old = src.read_text()
                except UnicodeError:
                    continue
                prefixes = [str(ROOT), '/mnt/data/home/jkp/data',
                            '/mnt/data/home/jkp', '/home/jkp/data']
                replacements = {prefix: str(workspace) for prefix in prefixes}
                # A nested paper used to live below a coordinator's workspaces/
                # tree. Redirect only this paper's own absolute legacy root;
                # generic root replacement would otherwise point its writers
                # into the archived coordinator tree restored as input links.
                nested_roots = [own_archive, *own_aliases]
                for relative in nested_roots:
                    if relative and '/' in relative:
                        for prefix in prefixes:
                            replacements[prefix.rstrip('/') + '/' + relative] = str(destination)
                # One substitution pass prevents a destination nested below
                # ROOT from being rewritten a second time into itself.
                pattern = '|'.join(re.escape(prefix) for prefix in sorted(replacements, key=len, reverse=True))
                new = re.sub('(?:' + pattern + r')(?![A-Za-z0-9_.-])',
                             lambda match: replacements[match.group(0)], old)
                if new != old:
                    dst.write_text(new)
                    changes.append(str(src.relative_to(ROOT)))


def generated_file(path, depth):
    """Keep mutable run records out of a fresh workspace.

    Source receipts and manifests are inputs, including verification metadata
    under sources/. Root verification documents are historical run outputs.
    """
    name = path.name.lower()
    if path.suffix.lower() in {'.log', '.lock', '.pid', '.tmp'}:
        return True
    if name in GENERATED_NAMES or name.endswith('-status.json'):
        return True
    return depth == 0 and (
        'verification' in name or 'validation' in name or
        name.startswith(('report', 'results', 'completion', 'final-checks'))
    )


def copy_generated(source, destination):
    """Restore writable output copies, preserving all archived bytes.

    Only generated paths use copies; raw/prepared inputs remain links. Existing
    staged files are never replaced. Reflinks share disk extents where available.
    Dereferencing output symlinks prevents writes through them into the archive.
    """
    if destination.is_symlink():
        raise ValueError(f'Output destination is an input link; use a new workspace: {destination}')
    require_real_parent(destination)
    if source.is_dir():
        if destination.exists():
            for parent, dirs, files in os.walk(destination, followlinks=False):
                for name in dirs + files:
                    if (Path(parent) / name).is_symlink():
                        raise ValueError(f'Existing output contains an input link; use a new workspace: {Path(parent) / name}')
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-a', '--dereference', '--no-clobber', '--reflink=auto',
                        str(source) + '/.', str(destination)], check=True)
    elif not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-a', '--dereference', '--no-clobber', '--reflink=auto',
                        str(source), str(destination)], check=True)


def metadata_file(path):
    """Small editable metadata, distinct from bulk/raw input payloads."""
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            return False
    except OSError:
        return False
    if path.suffix.lower() in METADATA_SUFFIXES:
        return True
    return path.suffix.lower() == '.txt' and any(
        word in path.name.lower() for word in
        ['manifest', 'requirements', 'environment', 'provenance', 'protocol',
         'readme', 'source', 'commit', 'receipt', 'config']
    )


def overlay(archive, destination, *, fresh_inputs, reuse_results, depth=0, reference_inputs=False):
    if not archive.exists():
        return
    require_real_parent(destination)
    # Already linked inputs are complete subtrees. Never recurse through them.
    if destination.is_symlink():
        return
    for src in archive.iterdir():
        dst = destination / src.name
        if src.name in {'__pycache__', '.pytest_cache', '.ruff_cache'}:
            continue
        if src.name in OUTPUTS:
            if reference_inputs:
                if not dst.exists() and not dst.is_symlink():
                    link(src, dst)
                elif dst.is_dir() and not dst.is_symlink():
                    overlay(src, dst, fresh_inputs=fresh_inputs, reuse_results=False,
                            depth=depth+1, reference_inputs=True)
            elif reuse_results:
                copy_generated(src, dst)
            continue
        if not reference_inputs and not src.is_dir() and generated_file(src, depth):
            # Stale lock/PID files are never useful replay inputs.
            if reuse_results and src.suffix.lower() not in {'.lock', '.pid', '.tmp'}:
                copy_generated(src, dst)
            continue
        if fresh_inputs and src.name in CACHES and not reference_inputs:
            if dst.is_symlink():
                raise ValueError(
                    f'Existing cached-input link {dst}; --fresh-inputs requires a new --workspace'
                )
            continue
        if src.is_dir() and not src.is_symlink():
            if src.name.startswith('.venv') or src.name == '.git':
                link(src, dst)
            elif src.name in CACHES or src.name in {'models', 'vendor', 'packages', 'deps', 'vendor-py311'}:
                link(src, dst)
            elif not dst.exists() and src.name in {'workspaces', 'reproductions', 'reviews', 'evidence', 'control-receipts'}:
                link(src, dst)
            elif dst.is_symlink():
                continue
            else:
                require_real_parent(dst)
                dst.mkdir(parents=True, exist_ok=True)
                is_reference = reference_inputs or src.name in REFERENCE_DIRS or src.name.startswith(('reference-', 'upstream', 'official-'))
                overlay(src, dst, fresh_inputs=fresh_inputs, reuse_results=reuse_results,
                        depth=depth+1, reference_inputs=is_reference)
        elif not dst.exists() and not dst.is_symlink():
            if metadata_file(src):
                require_real_parent(dst)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                link(src, dst)


def prepare(args):
    projects = catalogue()
    if args.project not in projects:
        raise SystemExit(f'Unknown project {args.project!r}; use list')
    workspace = args.workspace.resolve()
    if workspace == ROOT or ROOT in workspace.parents and '.replication-local' not in workspace.parts:
        raise SystemExit('Use .replication-local/workspace or a directory outside the source root')
    external_archive = args.data_root.resolve() / '.replication-local/artifacts'
    archive_root = getattr(args, 'archive_root', None)
    archive_root = Path(archive_root).resolve() if archive_root else (
        external_archive if external_archive.exists() else (LOCAL / 'artifacts').resolve()
    )
    protected = [archive_root, (LOCAL / 'artifacts').resolve()]
    protected += [(args.data_root / name).resolve() for name in RAW if (args.data_root / name).exists()]
    if workspace == args.data_root.resolve() or any(workspace == p or p in workspace.parents for p in protected):
        raise SystemExit('Shared inputs and historical archives cannot be used as a writable workspace')
    workspace.mkdir(parents=True, exist_ok=True)
    changes = []
    omitted_aliases = []
    # Stage all clean sibling implementations, since historical runners import
    # sibling evaluation/model modules. This copies code only.
    for name, item in projects.items():
        source = ROOT / item.get('source', name)
        target = workspace / name
        if source.exists():
            copy_source(source, target, workspace, changes,
                        own_archive=item.get('archive', name), own_aliases=item.get('aliases', []))
        archive = archive_root / item.get('archive', name)
        overlay(archive, target, fresh_inputs=args.fresh_inputs,
                reuse_results=args.reuse_results and name == args.project)
        # Some relocated projects originally occupied a nested path.
        for alias in item.get('aliases', []):
            if alias != name:
                # These coordinator containers are restored as archive input
                # links later. Do not materialize aliases inside those trees.
                if alias.startswith(('paper-reproduction-audit/', 'paper-reproductions/')) or symlink_directory((workspace / alias).parent):
                    omitted_aliases.append(alias)
                else:
                    link(target, workspace / alias)
    for name in RAW:
        src = args.data_root / name
        if src.exists():
            link(src, workspace / name)
    for name in ['protocol.txt', 'reproduce.txt', 'reference.xlsx', 'paperlist.txt']:
        src = args.data_root / name
        if not src.exists():
            src = archive_root / 'root' / name
        if src.exists():
            link(src, workspace / name)
    support = ROOT / '_replication/support'
    if support.exists():
        for source in support.iterdir():
            if source.is_dir():
                copy_source(source, workspace / source.name, workspace, changes)
                overlay(archive_root / source.name, workspace / source.name,
                        fresh_inputs=args.fresh_inputs, reuse_results=args.reuse_results)
    for old, new in ALIASES.items():
        if (workspace / new).exists():
            link(workspace / new, workspace / old)
    for name in projects:
        if '/' not in name and (workspace / name).exists():
            link(workspace / name, workspace / (name + '-reproduction'))
    receipt = {'project': args.project, 'data_root': str(args.data_root.resolve()),
               'archive_root': str(archive_root),
               'workspace': str(workspace), 'rewritten_path_files': sorted(set(changes)),
               'omitted_archival_aliases': omitted_aliases,
               'fresh_inputs': args.fresh_inputs, 'reuse_results': args.reuse_results}
    receipt_path = workspace / ('.prepare-' + args.project.replace('/', '_') + '.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(workspace / args.project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('list', help='List the clean paper folders')
    p = sub.add_parser('prepare', help='Stage source and link existing shared inputs')
    p.add_argument('--project', required=True)
    p.add_argument('--workspace', type=Path, default=LOCAL / 'workspace')
    p.add_argument('--data-root', type=Path, default=ROOT)
    p.add_argument('--archive-root', type=Path,
                   help='Existing artifact archive; defaults to DATA_ROOT/.replication-local/artifacts when present')
    p.add_argument('--fresh-inputs', action='store_true', help='Omit local prepared data/cache links; rebuild from shared raw inputs')
    p.add_argument('--reuse-results', action='store_true', help='Restore archived generated results for replay, using reflinks where available')
    args = parser.parse_args()
    if args.action == 'list':
        for name in catalogue():
            print(name)
    else:
        prepare(args)


if __name__ == '__main__':
    main()
