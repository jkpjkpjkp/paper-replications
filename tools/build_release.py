"""Build a data-free source archive from an explicit allowlist, without git."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
# Enumerate files, never recursively archive the mixed local research workspace.
RELEASE_FILES = [
    "README.md", "pyproject.toml", "uv.lock", ".gitignore", "LICENSE",
    "CITATION.cff", "THIRD_PARTY_NOTICES.md", "docs/UPSTREAM_LICENSE",
    "docs/VALIDATION.md", ".github/workflows/replication.yml",
    "src/vision_to_value/__init__.py", "src/vision_to_value/__main__.py",
    "src/vision_to_value/cli.py", "src/vision_to_value/data.py",
    "src/vision_to_value/table1.py", "src/vision_to_value/inputs.json",
    "src/vision_to_value/table1_expected.csv", "tests/test_replication.py",
    "tools/build_release.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/vision-to-value-replication-0.1.0.tar.gz")
    args = parser.parse_args()
    payloads = {}
    for name in RELEASE_FILES:
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing/unsafe release member: {name}")
        payloads[name] = path.read_bytes()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    payloads["RELEASE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, content in sorted(payloads.items()):
            info = tarfile.TarInfo(f"vision-to-value-replication-0.1.0/{name}")
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{digest}  {args.output.name}\n")
    print(f"Built {args.output}: {len(payloads)} files; SHA-256 {digest}")


if __name__ == "__main__":
    main()
