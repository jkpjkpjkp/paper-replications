"""PyTorch Frame TabularBenchmark sources with explicit, persisted splits."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# These names match PyTorch Frame's TabularBenchmark class, not the similarly
# named multiclass Jannis or continuous house_16H datasets in other collections.
DATASETS = {
    "MagicTelescope": ("clf_num", 0.974, 0.947),
    "house_16H": ("clf_num", 0.987, 0.951),
    "credit": ("clf_num", 0.884, 0.862),
    "jannis": ("clf_num", 0.923, 0.868),
    "Diabetes130US": ("clf_num", 0.657, 0.653),
    "MiniBooNE": ("clf_num", 0.992, 0.986),
    "california": ("clf_num", 0.986, 0.966),
    "yprop_4_1": ("reg_num", 0.832, 0.857),
    "houses": ("reg_num", 0.359, 0.401),
    "california_housing": ("yandex", 0.406, 0.418),
    "medical_charges": ("reg_cat", 0.130, 0.140),
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    tmp.replace(path)


def prepare(name, root="data", split_seed=0):
    root = Path(root)
    dest = root / "processed" / f"{name}-split{split_seed}.npz"
    if dest.exists():
        return dest
    category, _, _ = DATASETS[name]
    if category == "yandex":
        # The only continuous California Housing collection explicitly available
        # in Frame is Yandex; preserve its supplied train/val/test arrays.
        import zipfile

        raw = root / "raw" / f"{name}.zip"
        url = f"https://data.pyg.org/datasets/tables/revisiting_data/{name}.zip"
        _download(url, raw)
        arrays = {}
        with zipfile.ZipFile(raw) as z:
            for member in z.namelist():
                key = Path(member).stem
                if key in {f"{kind}_{s}" for kind in ("N", "y") for s in ("train", "val", "test")}:
                    with z.open(member) as stream:
                        arrays[key] = np.load(stream, allow_pickle=False)
        X_all = np.concatenate([arrays[f"N_{s}"] for s in ("train", "val", "test")])
        y_all = np.concatenate([arrays[f"y_{s}"] for s in ("train", "val", "test")])
        cuts = np.cumsum([0] + [len(arrays[f"y_{s}"]) for s in ("train", "val", "test")])
        split_ids = [np.arange(cuts[i], cuts[i + 1]) for i in range(3)]
        X_df = pd.DataFrame(X_all)
        split_description = "Official Yandex/PyTorch Frame train/val/test arrays"
    else:
        base = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"
        url = f"{base}/{category}/{name}.csv"
        raw = root / "raw" / f"{name}.csv"
        _download(url, raw)
        df = pd.read_csv(raw)
        X_df, y_all = df.iloc[:, :-1], df.iloc[:, -1].to_numpy()
        rows = np.arange(len(df))
        stratify = y_all if category.startswith("clf") else None
        trainval, test = train_test_split(rows, test_size=0.2,
                                         random_state=split_seed, stratify=stratify)
        train, val = train_test_split(trainval, test_size=0.25,
                                     random_state=split_seed,
                                     stratify=y_all[trainval] if stratify is not None else None)
        split_ids = [train, val, test]
        split_description = "60/20/20 random split; stratified for classification; split seed fixed across model seeds"
    train, val, test = split_ids
    task = "classification" if category.startswith("clf") else "regression"
    label_mapping = None
    if task == "classification":
        labels = sorted(pd.unique(y_all[train]).tolist(), key=str)
        if len(labels) != 2:
            raise ValueError(f"Expected binary classification; {name} has {labels}")
        label_mapping = {str(label): idx for idx, label in enumerate(labels)}
        y = np.array([label_mapping[str(value)] for value in y_all], dtype=np.float32)
        y_mean, y_std = 0.0, 1.0
    else:
        y = np.asarray(y_all, dtype=np.float64)
        y_mean, y_std = float(y[train].mean()), float(y[train].std())
        if y_std <= 0:
            raise ValueError("Constant regression target")
        # Table 1's small RMSEs suggest normalized targets, but the paper never
        # specifies the transform. Report both native and standardized RMSE.
        y = ((y - y_mean) / y_std).astype(np.float32)
    numeric = [c for c in X_df if pd.api.types.is_numeric_dtype(X_df[c])]
    categorical = [c for c in X_df if c not in numeric]
    blocks, feature_names = [], []
    minimums = {}
    if numeric:
        # pandas 3 copy-on-write may return a read-only view of float32 inputs.
        arr = X_df[numeric].to_numpy(dtype=np.float32, copy=True)
        arr[~np.isfinite(arr)] = np.nan
        train_arr = arr[train]
        minimum = np.array([np.nanmin(train_arr[:, j]) if np.isfinite(train_arr[:, j]).any() else 0
                            for j in range(arr.shape[1])], dtype=np.float32)
        arr = np.where(np.isnan(arr), minimum, arr)
        blocks.append(arr)
        feature_names.extend(map(str, numeric))
        minimums = dict(zip(map(str, numeric), map(float, minimum)))
    if categorical:
        arr = X_df[categorical].fillna("__MISSING__").astype(str)
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        enc.fit(arr.iloc[train])
        blocks.append(enc.transform(arr))
        feature_names.extend(enc.get_feature_names_out(list(map(str, categorical))).tolist())
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite preprocessed dataset")
    metadata = {"dataset": name, "task": task, "url": url,
                "raw_sha256": file_sha256(raw), "split_seed": split_seed,
                "split": split_description, "n_rows": len(X), "n_features": X.shape[1],
                "split_sizes": dict(zip(("train", "val", "test"), map(len, split_ids))),
                "target_mean_train": y_mean, "target_std_train": y_std,
                "target_transform": "none" if task == "classification" else "training mean/std z-score",
                "feature_names": feature_names, "minimum_imputation_train": minimums,
                "label_mapping": label_mapping,
                "paper_arbnn": DATASETS[name][1], "paper_xgboost": DATASETS[name][2]}
    payload = {"metadata": np.array(json.dumps(metadata))}
    for split, rows in zip(("train", "val", "test"), split_ids):
        payload[f"X_{split}"] = X[rows]
        payload[f"y_{split}"] = y[rows]
        payload[f"row_ids_{split}"] = rows
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **payload)
    dest.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return dest


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files if k != "metadata"}
        data["metadata"] = json.loads(str(archive["metadata"]))
    return data
