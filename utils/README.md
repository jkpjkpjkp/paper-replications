# Status tracker

`status_daemon.py` reads only `STATUS.txt` in the root's direct subfolders and
maintains the grouped folder list in the root's `STATUS.txt`. It copies the leading
integer, without assessing results or modifying source files or
`REPLICATION_STATUS.txt`. Nested folders are not scanned. Direct hidden folders
and directory symlinks are included if they contain a `STATUS.txt`.

The script uses Python's standard library. It polls every five seconds, picks up
added, changed, renamed and removed status files or folders, and replaces the
output atomically only when its contents change. Groups 0–3 are always shown;
other leading integers get their own groups. Files without a leading integer
are listed separately. Read errors are logged and preserve the previous output
until the next successful scan.

From `/home/jkp/data`, refresh once or run in the foreground:

```sh
python3 utils/status_daemon.py --once
python3 utils/status_daemon.py --interval 5
```

The default root is the parent of `utils`, regardless of the working directory.
Use `--root /path/to/folder` to track a different root.

Install and start the persistent user service on this machine:

```sh
systemctl --user enable --now /home/jkp/data/utils/status-tracker.service
```

The service starts with the user's systemd manager and restarts after failures.
To inspect it, read its logs, or stop it and disable future automatic starts:

```sh
systemctl --user status status-tracker.service
journalctl --user -u status-tracker.service -n 30
systemctl --user disable --now status-tracker.service
```

If the repository moves, update the service's paths before reinstalling it.
