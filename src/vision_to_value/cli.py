"""Portable command-line runner with explicit inputs and auditable outputs."""

import argparse
import importlib.metadata
from importlib.resources import files
import json
from pathlib import Path
import platform
import sys
from urllib.error import URLError

from . import __version__
from .data import acquire, sha256, specification
from .table1 import compare, compute


def run(args):
    inputs, receipts = acquire(args.cache_dir, args.upstream_dir, args.offline)
    table = compute(inputs)
    comparison = compare(table)
    matched = int(comparison.matches_3dp.sum())
    success = matched == 48
    out = args.output_dir
    if args.upstream_dir is not None and (out.resolve() == args.upstream_dir.resolve() or args.upstream_dir.resolve() in out.resolve().parents):
        raise ValueError("Output directory must be outside the read-only upstream checkout")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table1.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    summary = {
        "experiment": "Table 1, January 25 2026 SSRN preprint, page 11",
        "status": "pass" if success else "mismatch",
        "matched_cells": matched, "total_cells": len(comparison),
        "precision_decimal_places": 3,
        "start_month": "2001-01", "end_month": "2024-12", "n_months": 288,
        "hac_lags": 6, "hac_kernel": "bartlett", "finite_sample_correction": False,
        "scope": "Released sorted portfolios to Table 1; no raw CRSP processing or model training",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    package = files("vision_to_value")
    code_files = ["__init__.py", "__main__.py", "cli.py", "data.py", "table1.py", "inputs.json", "table1_expected.csv"]
    manifest = {
        "replication_version": __version__, "upstream": specification(),
        "python": platform.python_version(),
        "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions())),
        "inputs": receipts,
        "code_sha256": {name: sha256(package.joinpath(name).read_bytes()) for name in code_files},
        "output_sha256": {name: sha256((out / name).read_bytes()) for name in ["table1.csv", "comparison.csv", "summary.json"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Table 1: {matched}/48 numeric cells match at 3 decimals (288 months, HAC lag 6).")
    print(f"Outputs: {out}")
    return 0 if success else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, help="Read inputs from an existing pinned upstream checkout")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vision-to-value"), help="Downloaded CSV cache (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/table1"), help="Result directory; same-named result files are overwritten (default: %(default)s)")
    parser.add_argument("--offline", action="store_true", help="Require local/cached inputs; never download")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, URLError) as error:
        print(f"Replication failed: {error}", file=sys.stderr)
        return 2
