"""Algorithm 1 and Eqs. 1, 3--6 of Arboreal Neural Network.

The main-text Eq. 2 conflicts with Algorithm 1. The default follows Algorithm 1:
left=-1, right=+1, and weight=2**(-internal-node depth). See PROTOCOL.md.
P is stored in [tree, internal_node, leaf] order, the transpose of the appendix.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _feature_index(name, feature_names):
    if feature_names is not None and name in feature_names:
        return feature_names.index(name)
    if name.startswith("f") and name[1:].isdigit():
        return int(name[1:])
    raise ValueError(f"Cannot map split feature {name!r} to an input column")


@dataclass
class SymbolicTree:
    """Portable numeric XGBoost tree; right branch is x >= split threshold."""

    root: dict
    feature_names: list[str] | None = None

    def arrays(self, n_features: int, routing="algorithm1"):
        if routing not in {"algorithm1", "equation2"}:
            raise ValueError("routing must be algorithm1 or equation2")
        internals, leaves, paths = [], [], []

        def dfs(node, path):
            if "leaf" in node:
                leaves.append(node)
                paths.append(path)
                return
            if not np.isscalar(node["split_condition"]):
                raise ValueError("Native categorical splits are unsupported; one-hot encode first")
            idx = len(internals)
            internals.append(node)
            children = {c["nodeid"]: c for c in node["children"]}
            # XGBoost yes means x < threshold; no means x >= threshold.
            dfs(children[node["yes"]], path + [(idx, -1)])
            dfs(children[node["no"]], path + [(idx, +1)])

        dfs(self.root, [])
        features = np.array([_feature_index(n["split"], self.feature_names)
                             for n in internals], dtype=np.int64)
        if len(features) and (features.min() < 0 or features.max() >= n_features):
            raise ValueError("Split feature outside input dimension")
        thresholds = np.array([n["split_condition"] for n in internals], dtype=np.float32)
        values = np.array([n["leaf"] for n in leaves], dtype=np.float32)
        P = np.zeros((len(internals), len(leaves)), dtype=np.float32)
        for leaf, path in enumerate(paths):
            for depth, (idx, direction) in enumerate(path):
                if routing == "algorithm1":
                    P[idx, leaf] = direction * 2.0 ** (-depth)
                else:
                    # Literal main-text Eq. 2, retained only for the discrepancy audit.
                    P[idx, leaf] = -direction * 2.0 ** (1 - len(path))
        return features, thresholds, P, values, internals, leaves

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)

        def visit(node, rows):
            if not len(rows):
                return
            if "leaf" in node:
                out[rows] = node["leaf"]
                return
            feature = _feature_index(node["split"], self.feature_names)
            values = X[rows, feature]
            next_ids = np.where(np.isnan(values), node["missing"],
                                np.where(values < node["split_condition"],
                                         node["yes"], node["no"]))
            children = {c["nodeid"]: c for c in node["children"]}
            for child in (node["yes"], node["no"]):
                visit(children[child], rows[next_ids == child])

        visit(self.root, np.arange(len(X)))
        return out

    def rules(self):
        result = []

        def visit(node, conditions):
            if "leaf" in node:
                result.append({"if": conditions, "then": node["leaf"],
                               "leaf_node_id": node["nodeid"]})
                return
            children = {c["nodeid"]: c for c in node["children"]}
            for child, op in ((node["yes"], "<"), (node["no"], ">=")):
                visit(children[child], conditions + [{
                    "feature": node["split"], "operator": op,
                    "threshold": node["split_condition"],
                    "missing_follows_branch": node["missing"] == child}])

        visit(self.root, [])
        return result


class ArborealNetwork(nn.Module):
    """Vectorized forest of ArborCells; only f (thresholds) and v are trainable.

    Index gathering computes X W^T exactly for the fixed one-hot W, avoiding
    a dense input-to-node projection. Padding masks support unequal tree sizes
    and constant (root-only) trees without changing their probabilities.
    """

    def __init__(self, trees: list[SymbolicTree], n_features: int,
                 base_margin=0.0, tau1=100.0, tau2=10.0,
                 routing="algorithm1"):
        super().__init__()
        if not trees or n_features < 1 or tau1 <= 0 or tau2 <= 0:
            raise ValueError("Need nonempty trees, positive input dimension and temperatures")
        self.trees = copy.deepcopy(trees)
        self.n_features = int(n_features)
        self.tau1, self.tau2, self.routing = float(tau1), float(tau2), routing
        parsed = [t.arrays(n_features, routing) for t in trees]
        T, N, L = len(trees), max(1, max(len(a[0]) for a in parsed)), max(len(a[3]) for a in parsed)
        features = torch.zeros((T, N), dtype=torch.long)
        thresholds = torch.zeros(T, N)
        P = torch.zeros(T, N, L)
        values = torch.zeros(T, L)
        leaf_mask = torch.zeros(T, L, dtype=torch.bool)
        node_mask = torch.zeros(T, N, dtype=torch.bool)
        self.node_ids, self.leaf_ids = [], []
        for t, (feat, f, p, v, nodes, leaves) in enumerate(parsed):
            ni, nl = len(feat), len(v)
            features[t, :ni] = torch.from_numpy(feat)
            thresholds[t, :ni] = torch.from_numpy(f)
            P[t, :ni, :nl] = torch.from_numpy(p)
            values[t, :nl] = torch.from_numpy(v)
            node_mask[t, :ni] = True
            leaf_mask[t, :nl] = True
            self.node_ids.append([n["nodeid"] for n in nodes])
            self.leaf_ids.append([n["nodeid"] for n in leaves])
        self.register_buffer("features", features)
        self.register_buffer("P", P)
        self.register_buffer("leaf_mask", leaf_mask)
        self.register_buffer("node_mask", node_mask)
        self.register_buffer("base_margin", torch.tensor(float(base_margin)))
        self.f = nn.Parameter(thresholds)
        self.v = nn.Parameter(values)

    def _check_input(self, X):
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"Expected [batch, {self.n_features}] input")
        if not torch.isfinite(X).all():
            raise ValueError("Soft routing needs finite inputs; fit the imputer on training data")

    def affinities(self, X):
        self._check_input(X)
        d = torch.tanh(self.tau1 * (X[:, self.features] - self.f))
        return torch.bmm(d.transpose(0, 1), self.P).transpose(0, 1)

    def leaf_probabilities(self, X):
        logits = self.tau2 * self.affinities(X)
        return torch.softmax(logits.masked_fill(~self.leaf_mask, -torch.inf), dim=-1)

    def forward(self, X):
        alpha = self.leaf_probabilities(X)
        return (alpha * self.v).sum(dim=(-1, -2)) + self.base_margin

    def decompile(self):
        """Copy the exact topology/features and replace split thresholds and leaves.

        This preserves tree parameters; soft and hard predictions need not match.
        """
        trees = copy.deepcopy(self.trees)
        f, v = self.f.detach().cpu().numpy(), self.v.detach().cpu().numpy()
        for t, tree in enumerate(trees):
            by_id = {}

            def visit(node):
                by_id[node["nodeid"]] = node
                for child in node.get("children", []):
                    visit(child)

            visit(tree.root)
            for idx, node_id in enumerate(self.node_ids[t]):
                by_id[node_id]["split_condition"] = float(f[t, idx])
            for idx, node_id in enumerate(self.leaf_ids[t]):
                by_id[node_id]["leaf"] = float(v[t, idx])
        return trees

    @torch.no_grad()
    def predict_numpy(self, X, batch_size=2048, hard=False):
        X = np.asarray(X, dtype=np.float32)
        if hard:
            return (sum((t.predict(X).astype(np.float64) for t in self.decompile()),
                        np.zeros(len(X))) + self.base_margin.item()).astype(np.float32)
        self.eval()
        device = self.f.device
        return np.concatenate([self(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
                               for i in range(0, len(X), batch_size)])

    @classmethod
    def from_xgboost(cls, booster, X_reference, **kwargs):
        """Compile a numeric gbtree booster, retaining its actual intercept.

        The intercept is inferred from training examples and verified constant;
        this supports both old 0.5 and newer fitted/vector base_score formats.
        Dumped leaf values already include boosting shrinkage: do not rescale.
        """
        import xgboost as xgb

        config = json.loads(booster.save_config())["learner"]
        objective = config["objective"]["name"]
        if objective not in {"binary:logistic", "reg:squarederror"}:
            raise ValueError(f"Unsupported objective {objective}; binary and regression only")
        if config["gradient_booster"]["name"] != "gbtree":
            raise ValueError("Only gbtree boosters are supported")
        trees = [SymbolicTree(json.loads(tree), booster.feature_names)
                 for tree in booster.get_dump(dump_format="json")]
        X_reference = np.asarray(X_reference, dtype=np.float32)
        if not np.isfinite(X_reference).all() or not len(X_reference):
            raise ValueError("Provide finite, pre-imputed training reference data")
        raw = booster.predict(xgb.DMatrix(X_reference, feature_names=booster.feature_names),
                              output_margin=True).astype(np.float64)
        summed = sum((t.predict(X_reference).astype(np.float64) for t in trees),
                     np.zeros(len(X_reference)))
        bias = float(np.median(raw - summed))
        if not np.allclose(raw - summed, bias, atol=2e-4, rtol=1e-5):
            raise ValueError("Nonconstant base margin; booster cannot be compiled faithfully")
        return cls(trees, X_reference.shape[1], base_margin=bias, **kwargs)

    def export_rules(self, path):
        payload = {"format": "arboreal-symbolic-forest-v1", "comparison": "x < threshold",
                   "base_margin": self.base_margin.item(),
                   "n_features": self.n_features,
                   "trees": [{"root": t.root, "feature_names": t.feature_names,
                              "rules": t.rules()} for t in self.decompile()]}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "n_features": self.n_features,
                    "trees": [{"root": t.root, "feature_names": t.feature_names} for t in self.trees],
                    "tau1": self.tau1, "tau2": self.tau2, "routing": self.routing}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls([SymbolicTree(**t) for t in payload["trees"]], payload["n_features"],
                    tau1=payload["tau1"], tau2=payload["tau2"], routing=payload["routing"])
        model.load_state_dict(payload["state_dict"])
        return model.to(device)


class ArborCell(ArborealNetwork):
    def __init__(self, tree: SymbolicTree, n_features: int, **kwargs):
        super().__init__([tree], n_features, **kwargs)
