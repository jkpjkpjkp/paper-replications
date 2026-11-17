"""Repeatable executable evidence for defects in the pinned author release."""
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official"))
import PDUArch
from evaluator import evaluate


def main():
    out = {"commit": subprocess.check_output(["git", "-C", str(ROOT / "official"), "rev-parse", "HEAD"], text=True).strip()}
    proc = subprocess.run([sys.executable, "main.py", "--help"], cwd=ROOT / "official", capture_output=True, text=True)
    out["original_main"] = {"returncode": proc.returncode, "stderr": proc.stderr}
    out["missing_modules"] = [name for name in ["load_data.py", "PUD.py"] if not (ROOT / "official" / name).exists()]
    out["backbone_forward_signature"] = str(inspect.signature(PDUArch.PDU.forward))
    backbone = PDUArch.PDU(torch.device("cpu"), 1, 2, 1, 32, 32, 1, 64, 4, 12, 512, 64, 64, 8)
    x, edges = torch.randn(108, 4), torch.tensor([[0, 1], [1, 0]])
    for name, kwargs in [("unsupported_keywords", {"mask": "train", "e": 0}), ("convolution_dimensions", {})]:
        try:
            backbone(x, edges, **kwargs)
            out[name] = "unexpected success"
        except Exception as exc:
            out[name] = f"{type(exc).__name__}: {exc}"
    # Same aligned stock records, only permuted IDs: legitimate ranking metrics
    # must remain equal when there are no score or label ties.
    rng = np.random.default_rng(2025)
    pred, truth = rng.normal(size=(1026, 4)), rng.normal(0, .02, size=(1026, 4))
    mask = np.ones_like(pred)
    p = rng.permutation(1026)
    original, permuted = evaluate(pred, truth, mask), evaluate(pred[p], truth[p], mask[p])
    out["evaluator"] = {"returned_keys": list(original),
        "missing_requested_keys": [k for k in ["sharpe5", "mrrt"] if k not in original],
        "ndcg_original": original["realndcg5"], "ndcg_after_stock_permutation": permuted["realndcg5"],
        "ndcg_changes_when_only_ids_change": not np.isclose(original["realndcg5"], permuted["realndcg5"])}
    offsets = np.arange(756)
    target = offsets + 4
    out["training_split"] = {"shuffle_pool": len(offsets), "steps_used_per_epoch": 752,
        "validation_label_offsets_in_pool": offsets[target >= 756].tolist(),
        "validation_target_days_in_pool": target[target >= 756].tolist()}
    (ROOT / "results/release_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
