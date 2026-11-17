import io
import zipfile

import numpy as np
import pandas as pd

from arboreal.data import load, prepare


def test_yandex_preserves_supplied_splits_and_train_only_target_transform(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    arrays = {
        "N_train": np.array([[1, np.nan], [2, 10], [3, 20]], dtype=np.float32),
        "N_val": np.array([[4, np.nan]], dtype=np.float32),
        "N_test": np.array([[1000, -1000]], dtype=np.float32),
        "y_train": np.array([1, 2, 3], dtype=np.float32),
        "y_val": np.array([10], dtype=np.float32),
        "y_test": np.array([1000], dtype=np.float32),
    }
    with zipfile.ZipFile(raw / "california_housing.zip", "w") as archive:
        for key, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value)
            archive.writestr(key + ".npy", stream.getvalue())
    data = load(prepare("california_housing", tmp_path))
    assert data["metadata"]["target_mean_train"] == 2
    np.testing.assert_allclose(data["y_train"].mean(), 0, atol=1e-7)
    np.testing.assert_allclose(data["y_train"].std(), 1)
    assert data["X_train"][0, 1] == 10
    assert data["X_val"][0, 1] == 10
    assert data["X_test"][0, 1] == -1000
    np.testing.assert_array_equal(data["row_ids_test"], [4])


def test_random_split_is_disjoint_and_preprocessing_uses_training_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame({"numeric": np.arange(100, dtype=float),
                          "categorical": ["unique_" + str(i) for i in range(100)],
                          "target": ["negative", "positive"] * 50})
    frame.loc[[0, 20, 40], "numeric"] = np.nan
    frame.to_csv(raw / "credit.csv", index=False)
    data = load(prepare("credit", tmp_path))
    train, val, test = [data[f"row_ids_{s}"] for s in ("train", "val", "test")]
    assert [len(train), len(val), len(test)] == [60, 20, 20]
    assert len(set(train) | set(val) | set(test)) == 100
    assert not (set(train) & set(val) or set(train) & set(test) or set(test) & set(val))
    expected_min = frame.iloc[train]["numeric"].min()
    assert data["metadata"]["minimum_imputation_train"]["numeric"] == expected_min
    # Unseen validation/test categories must not acquire fitted indicator columns.
    assert data["X_train"].shape[1] == 61
    assert (data["X_val"][:, 1:] == 0).all()
    assert (data["X_test"][:, 1:] == 0).all()
    for split in ("train", "val", "test"):
        assert set(np.unique(data[f"y_{split}"])) == {0, 1}
        assert np.isfinite(data[f"X_{split}"]).all()
