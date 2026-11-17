# Upload and maintain this collection

The collection uses one GitHub repository, `jkpjkpjkp/paper-replications`.
`main` is the catalogue. Each entry in `_replication/upload_projects.json`
has an independent `paper/<name>` branch and a local Git repository in its
original folder. No submodules or extra GitHub repositories are needed.

## Refresh the upload

From the collection root:

```bash
python _replication/prepare_upload.py
python _replication/prepare_upload.py --audit-only
```

The first command installs the shared ignore policy in each paper and creates
source-only commits. It gathers those branches into the root repository.
Existing upstream repositories retain their current HEAD, index, remotes and old
history. Their upload branch is separate, so earlier data/checkpoint commits
cannot reach GitHub through this workflow. New repositories start on their
paper branch. Later refreshes append a commit only if the source tree changes.
Staged edits are preserved. A clean index on an active publication branch follows
its refreshed commit; indexes on original upstream branches are unchanged. Use
the helper to refresh publication branches rather than committing from an old
upstream branch.

Nested upstream repositories are flattened into regular source files in the
upload commit; their local Git directories are preserved. Symlinks are omitted,
never followed. Files of 25 MiB or larger and recognizable credential tokens
stop preparation. The local report at `.replication-local/upload/report.json`
records every included file's size/hash and skipped directories/symlinks.
This is a limited automated credential check, not a guarantee about all content.

`_replication/upload.gitignore` excludes datasets, binary arrays, model weights,
results, caches, environments, credentials, and `LLM/` and `CRYPTO/` at any depth.
Exact source-code packages named `data`/`dataset` are exceptions in the manifest;
the fit-grid configuration CSV in Set Sequence is also retained. Actual data
folders stay ignored. To change selection, edit the manifest or shared policy,
then rerun preparation. Existing project ignore rules are preserved.

## Push once the remote exists

```bash
git remote add origin git@github.com:jkpjkpjkp/paper-replications.git
git push origin main 'refs/heads/paper/*:refs/heads/paper/*'
```

If `origin` already exists, inspect `git remote -v` and skip the add command when
it has the intended URL. Push only `main` and the `paper/*` refs. Avoid `--mirror`,
`--all`, force pushes, or pushing original project histories. This keeps existing
upstream branches/tags and previously committed data local. No remote is created
or pushed by the preparation script itself.

The main catalogue's ordinary `git status` shows changes as files are edited;
the helper's commit uses a separate index to preserve any existing staging.
Inspect the upload with `git show main:README.md` and
`git ls-tree -r paper/alphaqcm`, or clone a paper branch into a new directory.

## Source archive

`python _replication/package.py` packages the same selected paper sources and
main files in their original folder layout, honoring the upload ignore policy.
Datasets and the excluded collections are absent from both Git and the archive.
