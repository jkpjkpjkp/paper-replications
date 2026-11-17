"""Offline numerical and failure-path tests; fixtures contain synthetic returns."""

from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from importlib.resources import files
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vision_to_value.cli import main
from vision_to_value.data import acquire, sha256
from vision_to_value.table1 import MONTHS, compare, compute, mean_hac, read_portfolios


def fixture():
    rows = []
    for t, month in enumerate(MONTHS):
        for size in [1, 2, 3]:
            for signal in [1, 2, 3]:
                value = 0.002 * signal + 0.001 * size + np.sin(t / 3) * (0.01 + signal * 0.002)
                rows.append([str(month.start_time.date()), size, signal, value])
    return pd.DataFrame(rows, columns=["date", "size_decile_3", "signal_decile_3", "Cap_VW_return"])


def encode(frame):
    return frame.to_csv(index=False).encode()


def fake_spec(content):
    return {"commit": "a" * 40, "files": [{"model": model,
            "path": f"sdf_analysis/sorted_portfolio/{model}_3by3.csv",
            "bytes": len(content), "sha256": sha256(content)} for model in ["vit", "cnn"]]}


class NumericalTests(unittest.TestCase):
    def test_hac_against_independent_bartlett_formula(self):
        y = np.random.default_rng(42).normal(0.01, 0.03, 288)
        y[1:] += 0.6 * y[:-1]
        residuals = y - y.mean()
        meat = residuals @ residuals
        for lag in range(1, 7):
            meat += 2 * (1 - lag / 7) * (residuals[lag:] @ residuals[:-lag])
        standard_error = np.sqrt(meat) / len(y)
        result = mean_hac(y)
        self.assertAlmostEqual(result["t_stat"], y.mean() / standard_error, places=12)
        self.assertAlmostEqual(result["mean_pct"], 100 * y.mean(), places=12)

    def test_shuffled_rows_align_high_minus_low_by_month(self):
        source = fixture().sample(frac=1, random_state=4)
        actual = compute({model: encode(source) for model in ["vit", "cnn"]})
        self.assertEqual(len(actual), 24)
        self.assertTrue(actual.n_months.eq(288).all())
        spread = actual.query("model == 'vit' and size == 'Small' and signal == 'H-L'").iloc[0]
        expected = 0.004 + np.sin(np.arange(288) / 3) * 0.004
        self.assertAlmostEqual(spread.mean_pct, mean_hac(expected)["mean_pct"], places=12)
        self.assertAlmostEqual(spread.t_stat, mean_hac(expected)["t_stat"], places=12)

    def test_all_published_cells_checked_and_mismatch_detected(self):
        with files("vision_to_value").joinpath("table1_expected.csv").open("rb") as stream:
            table = pd.read_csv(stream)
        self.assertEqual(int(compare(table).matches_3dp.sum()), 48)
        table.loc[0, "mean_pct"] += 0.001
        self.assertEqual(int(compare(table).matches_3dp.sum()), 47)
        with self.assertRaises(ValueError):
            compare(table.iloc[1:])
        with self.assertRaises(ValueError):
            compare(pd.concat([table, table.iloc[:1]]))

    def test_invalid_hac_series_rejected(self):
        for series in [[1, 2], [1.0] * 8 + [np.nan], np.ones((10, 2))]:
            with self.subTest(series=series), self.assertRaises(ValueError):
                mean_hac(series)


class InputValidationTests(unittest.TestCase):
    def test_full_panel(self):
        self.assertEqual(len(read_portfolios(encode(fixture()))), 2592)

    def test_bad_panel_rejected(self):
        good = fixture()
        variants = {
            "duplicate": pd.concat([good, good.iloc[:1]]),
            "missing_cell": good.iloc[1:],
            "missing_column": good.drop(columns="Cap_VW_return"),
        }
        for name, column, value in [
            ("missing_return", "Cap_VW_return", np.nan),
            ("infinite_return", "Cap_VW_return", np.inf),
            ("return_below_minus_one", "Cap_VW_return", -1.1),
            ("invalid_size", "size_decile_3", 4),
            ("invalid_signal", "signal_decile_3", 0),
            ("invalid_date", "date", "wrong"),
            ("extra_month", "date", "2025-01-01"),
        ]:
            frame = good.copy()
            frame.loc[0, column] = value
            variants[name] = frame
        for name, frame in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_portfolios(encode(frame))

    def test_checksum_and_offline_cache(self):
        content = b"synthetic fixture"
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.data.specification", return_value=fake_spec(content)), patch("vision_to_value.data.urlopen") as network:
            cache = Path(directory)
            for model in ["vit", "cnn"]:
                (cache / f"{model}_3by3.csv").write_bytes(content)
            inputs, receipts = acquire(cache, offline=True)
            self.assertEqual(inputs["vit"], content)
            self.assertEqual(receipts[0]["origin"], "cache")
            (cache / "vit_3by3.csv").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                acquire(cache)
            network.assert_not_called()

    def test_missing_offline_input_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.data.urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                acquire(Path(directory), offline=True)
            network.assert_not_called()

    def test_download_verification_before_cache_write(self):
        content = b"synthetic fixture"
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.data.specification", return_value=fake_spec(content)), patch("vision_to_value.data.urlopen") as network:
            network.return_value.__enter__.return_value = BytesIO(b"invalid response")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                acquire(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_successful_download_then_offline_reuse(self):
        content = b"synthetic fixture"
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.data.specification", return_value=fake_spec(content)), patch("vision_to_value.data.urlopen", side_effect=lambda *a, **kw: BytesIO(content)) as network:
            inputs, receipts = acquire(Path(directory))
            self.assertEqual(network.call_count, 2)
            self.assertEqual(receipts[0]["origin"], "download")
            cached, _ = acquire(Path(directory), offline=True)
            self.assertEqual(inputs, cached)
            self.assertEqual(network.call_count, 2)

    def test_local_checkout_is_read_only(self):
        content = b"synthetic fixture"
        spec = fake_spec(content)
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.data.specification", return_value=spec), patch("vision_to_value.data.urlopen") as network:
            root = Path(directory)
            for entry in spec["files"]:
                path = root / entry["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            before = {str(p): p.read_bytes() for p in root.rglob("*.csv")}
            acquire(root / "unused-cache", upstream_dir=root)
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*.csv")})
            self.assertFalse((root / "unused-cache").exists())
            network.assert_not_called()


class CommandLineTests(unittest.TestCase):
    def test_missing_inputs_exit_two_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            root = Path(directory)
            result = main(["--offline", "--cache-dir", str(root / "cache"), "--output-dir", str(root / "out")])
            self.assertEqual(result, 2)
            self.assertFalse((root / "out").exists())

    def test_mismatch_exit_one_and_writes_diagnostics(self):
        content = encode(fixture())
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.cli.acquire", return_value=({"vit": content, "cnn": content}, [])), redirect_stdout(StringIO()):
            root = Path(directory)
            result = main(["--output-dir", str(root / "out")])
            self.assertEqual(result, 1)
            self.assertEqual({p.name for p in (root / "out").iterdir()}, {"table1.csv", "comparison.csv", "summary.json", "manifest.json"})

    def test_upstream_output_rejected(self):
        content = encode(fixture())
        with tempfile.TemporaryDirectory() as directory, patch("vision_to_value.cli.acquire", return_value=({"vit": content, "cnn": content}, [])), redirect_stderr(StringIO()):
            result = main(["--upstream-dir", directory, "--output-dir", str(Path(directory) / "out")])
            self.assertEqual(result, 2)
            self.assertFalse((Path(directory) / "out").exists())


if __name__ == "__main__":
    unittest.main()
