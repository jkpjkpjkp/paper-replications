from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from arboreal.data import file_sha256, load
from arboreal.experiment import metrics
from arboreal.model import ArborealNetwork, SymbolicTree


def topology(node):
    """Retain every symbolic field except trainable thresholds/leaf values."""
    return {k: (True if k == "leaf" else
                [topology(c) for c in v] if k == "children" else v)
            for k, v in node.items() if k != "split_condition"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="artifacts/benchmark")
    p.add_argument("--data-root", default="data")
    p.add_argument("--allow-partial", action="store_true",
                   help="Check completed runs during training without certifying suite completeness")
    args = p.parse_args()
    root = Path(args.root)
    settings = json.loads((root / "suite_config.json").read_text())
    expected = len(settings["datasets"]) * len(settings["seeds"])
    paths = sorted(root.glob("*/seed*/result.json"))
    complete = len(paths) == expected
    if not complete and not args.allow_partial:
        raise AssertionError(f"Incomplete suite: {len(paths)}/{expected}")
    torch.set_num_threads(2)
    checked = []
    observed = set()
    configured = {(name, seed) for name in settings["datasets"] for seed in settings["seeds"]}
    for path in paths:
        result = json.loads(path.read_text())
        dest = path.parent
        identity = (result["dataset"], result["seed"])
        assert identity in configured and identity not in observed
        assert dest.name == f"seed{result['seed']}" and dest.parent.name == result["dataset"]
        observed.add(identity)
        data_path = (Path(args.data_root) / "processed" /
                     f"{result['dataset']}-split{settings['split_seed']}.npz")
        config = json.loads((dest / "config.json").read_text())
        assert file_sha256(data_path) == config["data_sha256"]
        data = load(data_path)
        ids = [set(data[f"row_ids_{s}"].tolist()) for s in ("train", "val", "test")]
        assert all(not ids[i].intersection(ids[j]) for i in range(3) for j in range(i))
        with np.load(dest / "predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["row_ids"], data["row_ids_test"])
            np.testing.assert_array_equal(archive["y"], data["y_test"])
            for method, values in result["test"].items():
                recomputed = metrics(archive["y"], archive[method], result["task"],
                                     data["metadata"]["target_std_train"])
                for metric, value in values.items():
                    np.testing.assert_allclose(recomputed[metric], value, rtol=1e-10, atol=1e-12)
            # Reload checkpoint and exported symbolic rules independently.
            model = ArborealNetwork.load(dest / "arbnn.pt")
            X = data["X_test"][:128]
            np.testing.assert_allclose(model.predict_numpy(X), archive["arbnn_soft"][:128], atol=2e-5)
            booster = xgb.Booster(model_file=dest / "xgboost.json")
            np.testing.assert_allclose(booster.predict(xgb.DMatrix(X), output_margin=True),
                                       archive["xgboost"][:128], atol=2e-5)
            rules = json.loads((dest / "rules.json").read_text())
            trees = [SymbolicTree(t["root"], t["feature_names"]) for t in rules["trees"]]
            assert len(trees) == len(model.trees)
            assert all(topology(original.root) == topology(refined.root)
                       for original, refined in zip(model.trees, trees))
            hard = sum((t.predict(X).astype(np.float64) for t in trees), np.zeros(len(X)))
            hard += rules["base_margin"]
            np.testing.assert_allclose(hard, archive["decompiled_hard"][:128], atol=2e-5)
            assert set(dict(model.named_parameters())) == {"f", "v"}
        checked.append(str(dest))
    if complete:
        assert observed == configured
    payload = {"status": "passed" if complete else "passed_partial", "complete": complete,
               "runs_verified": len(checked), "runs_expected": expected,
               "checkpoint_replay_rows_per_run": 128,
               "checks": ["suite completeness", "disjoint split IDs", "held-out row identity",
                          "dataset SHA-256", "all saved metrics re-scored",
                          "ArbNN and XGBoost checkpoint prediction replay",
                          "independent symbolic rule evaluation", "feature/topology preservation",
                          "only f/v trainable"]}
    if complete:
        (root / "verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
