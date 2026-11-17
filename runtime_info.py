"""Best-effort Linux resource telemetry; no dependency on a named service."""
from pathlib import Path


def cgroup_directory():
    try:
        entry = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        return Path("/sys/fs/cgroup") / entry.split(":", 2)[2].lstrip("/")
    except (OSError, StopIteration):
        return None


def resource_value(group, name):
    try:
        return (group / name).read_text().strip() if group is not None else None
    except OSError:
        return None


def resource_limits(group):
    return {key: resource_value(group, key) for key in ["memory.max", "memory.swap.max", "cpu.max", "pids.max"]}


def peak_memory(group):
    value = resource_value(group, "memory.peak")
    return int(value) if value is not None and value.isdigit() else None
