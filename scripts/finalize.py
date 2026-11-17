"""Attach self-contained feature/preprocessing context and seal artifact hashes."""
from __future__ import annotations
import hashlib
import json
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--root", default="artifacts/benchmark")
root = Path(p.parse_args().root)
settings = json.loads((root / "suite_config.json").read_text())
paths = sorted(root.glob("*/seed*/result.json"))
assert len(paths) == len(settings["datasets"]) * len(settings["seeds"]), "Suite incomplete"
for path in paths:
    dest = path.parent
    config = json.loads((dest / "config.json").read_text())
    metadata = config["data_metadata"]
    rules_path = dest / "rules.json"
    rules = json.loads(rules_path.read_text())
    rules["input_feature_names"] = metadata["feature_names"]
    rules["preprocessing"] = {
        k: metadata[k] for k in ("minimum_imputation_train", "label_mapping",
                                "target_mean_train", "target_std_train", "target_transform")}
    rules["prediction"] = ("sigmoid(sum(tree_margin) + base_margin)"
                           if metadata["task"] == "classification" else
                           "(sum(tree_value) + base_margin) * target_std_train + target_mean_train")
    rules["feature_reference"] = "f<N> indexes input_feature_names[N] when a tree has no named features"
    rules_path.write_text(json.dumps(rules, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


files = [p for dirname in ("arboreal", "scripts", "tests", "sources")
         for p in Path(dirname).rglob("*") if p.is_file()
         and p.suffix in {".py", ".pdf", ".md", ".txt"}]
files += [Path(name) for name in ("README.md", "PROTOCOL.md", "pyproject.toml", "requirements.lock.txt") if Path(name).is_file()]
Path("sources/SHA256SUMS.json").write_text(json.dumps({str(p): digest(p) for p in sorted(files)}, indent=2) + "\n")
artifact_files = [p for p in root.rglob("*") if p.is_file()
                  and p.name not in {".run.lock", "SHA256SUMS.json"}]
(root / "SHA256SUMS.json").write_text(json.dumps({str(p.relative_to(root)): digest(p)
                                                for p in sorted(artifact_files)}, indent=2) + "\n")
print(f"Finalized {len(paths)} runs; hashed {len(artifact_files)} benchmark artifacts")
