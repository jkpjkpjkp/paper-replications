"""Small, deterministic counterexamples for paper ambiguities; no test tuning."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from arboreal.model import ArborCell, SymbolicTree

torch.set_num_threads(1)
tree = SymbolicTree({"nodeid": 0, "split": "f0", "split_condition": 0.0,
                     "yes": 1, "no": 2, "missing": 1,
                     "children": [{"nodeid": 1, "leaf": -2.0},
                                  {"nodeid": 2, "leaf": 3.0}]})
correct = ArborCell(tree, 1)
literal = ArborCell(tree, 1, routing="equation2")
far = np.array([[-1.0], [1.0]], dtype=np.float32)
near = np.array([[-1e-7], [1e-7]], dtype=np.float32)
scale = np.array([[-1e-3], [1e-3]], dtype=np.float32)
payload = {
    "routing_sign_audit": {
        "input": far.tolist(), "original_hard": tree.predict(far).tolist(),
        "algorithm1_soft": correct.predict_numpy(far).tolist(),
        "literal_equation2_soft": literal.predict_numpy(far).tolist(),
    },
    "finite_temperature_counterexample": {
        "input": near.tolist(), "hard": correct.predict_numpy(near, hard=True).tolist(),
        "soft": correct.predict_numpy(near).tolist(),
        "conclusion": "Exact hard/soft prediction equivalence does not hold at finite tau",
    },
    "scale_counterexample": {
        "input": scale.tolist(), "soft": correct.predict_numpy(scale).tolist(),
        "input_scaled_1000": (scale * 1000).tolist(),
        "soft_scaled_1000": correct.predict_numpy(scale * 1000).tolist(),
        "hard_unchanged": bool(np.array_equal(tree.predict(scale), tree.predict(scale * 1000))),
        "conclusion": "Fixed-temperature soft routing is sensitive to feature scale",
    },
}
dest = Path("artifacts/method_audit.json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
